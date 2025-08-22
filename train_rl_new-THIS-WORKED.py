# train_ac_corr.py
"""
Actor–Critic trainer: Tetris board → DNA sequence
Reward = per-sample contribution to Pearson correlation(design, game-score)
Black-box constraint: PWM scorers are used only under torch.no_grad().
"""

# ───────────────────────── Imports ─────────────────────────
import json, math, csv
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.onnx import export as onnx_export

# ───────────────────── Hyper-parameters ────────────────────
LR              = 2e-3
BATCH_SIZE      = 64
EPOCHS          = 300
CONV_CHANNELS   = 32
NOISE_STD       = 0.1
DNA_LEN         = 30 #196
K_TETRIS_MAX    = 300
VAL_SPLIT       = 0.05
TEMP_ENABLED    = True
TEMP_RATE       = 0.8
TEMP_MAX        = 5.0
ENTROPY_COEF    = 0.0001 #1e-2
# GRAD_CLIP       = 1.0

ONNX_PATH  = "actor_corr.onnx"
ACTOR_PATH = "actor_corr.pt"
CRITIC_PATH= "critic_corr.pt"
DEVICE     = "cpu"   # set "cuda" if available

# ───────────────────── PWM loading (black box) ─────────────
MOTIF_DIR = Path("motifs")
PPAR_ID, NFKB_ID = "MA0065.2", "MA0105.4"

def _parse_meme(fp: Path):
    lines = fp.read_text().splitlines()
    i = next(i for i,l in enumerate(lines) if l.startswith("letter-probability"))
    w  = int(lines[i].split("w=")[1].split()[0])
    mat = [list(map(float, lines[j].split())) for j in range(i+1,i+1+w)]
    return torch.tensor(mat, dtype=torch.float32)

PPAR_PWM = _parse_meme(MOTIF_DIR/f"{PPAR_ID}.meme").to(DEVICE)
NFKB_PWM = _parse_meme(MOTIF_DIR/f"{NFKB_ID}.meme").to(DEVICE)
PPAR_MAX = float(PPAR_PWM.max(1).values.sum())
NFKB_MAX = float(NFKB_PWM.max(1).values.sum())

conv_ppar = nn.Conv1d(4,1,PPAR_PWM.shape[0],bias=False).to(DEVICE)
conv_nkfb = nn.Conv1d(4,1,NFKB_PWM.shape[0],bias=False).to(DEVICE)
conv_ppar.weight.data.copy_(PPAR_PWM.t().unsqueeze(0))
conv_nkfb.weight.data.copy_(NFKB_PWM.t().unsqueeze(0))
for p in (*conv_ppar.parameters(), *conv_nkfb.parameters()):
    p.requires_grad=False

@torch.no_grad()
def pwm_score(one_hot, conv, mx):
    seq = one_hot.transpose(1,2)           # [B,4,L] → conv
    scr = conv(seq).squeeze(1).max(-1).values
    return torch.clamp(scr / mx, 0, 1)     # [B]

# ───────────────────── Dataset / Collate ───────────────────
class TetrisDataset(Dataset):
    def __init__(self, path:str):
        self.samples = json.load(open(path))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        e = self.samples[idx]
        return (torch.tensor(e["game_matrix"],dtype=torch.float32),
                torch.tensor(e["score"],dtype=torch.float32))

def collate(batch):
    boards,scores = zip(*batch)
    T = max(b.shape[0] for b in boards)
    pads,masks=[],[]
    for b in boards:
        pad = T-b.shape[0]
        if pad: pads.append(torch.cat([b, torch.zeros(pad,10)]))
        else:   pads.append(b)
        masks.append(torch.cat([torch.ones(b.shape[0]), torch.zeros(pad)]))
    return torch.stack(pads), torch.stack(masks), torch.stack(scores)

# ───────────────────── Actor model (CNN) ───────────────────
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(10,CONV_CHANNELS//2,3,padding=1)
        self.c2 = nn.Conv1d(10,CONV_CHANNELS//2,7,padding=3,stride=2)
        self.fc1= nn.Linear(CONV_CHANNELS, 32)
        self.fc2= nn.Linear(32, DNA_LEN*4)
    def forward(self,b,mask):
        x=b.transpose(1,2)
        x1=F.relu(self.c1(x)); m1=mask.unsqueeze(1)
        x2=F.relu(self.c2(x)); m2=mask[:,::2].unsqueeze(1)
        p1=(x1*m1).sum(-1)/m1.sum(-1); p2=(x2*m2).sum(-1)/m2.sum(-1)
        z=torch.cat([p1,p2],1)
        if self.training: z+=torch.randn_like(z)*NOISE_STD
        z=F.relu(self.fc1(z))
        return self.fc2(z).view(-1,DNA_LEN,4)   # logits

# ───────────────────── Critic (shared earlier) ─────────────
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bnet = nn.Sequential(
            nn.Conv1d(10,16,3,padding=1), nn.ReLU(),
            nn.Conv1d(16,16,3,padding=1), nn.ReLU()
        )
        self.dnet = nn.Sequential(
            nn.Conv1d(4,16,5,padding=2), nn.ReLU(),
            nn.Conv1d(16,16,5,padding=2), nn.ReLU()
        )
        self.mix  = nn.Sequential(
            nn.Conv1d(32,32,3,padding=1), nn.ReLU(),
            nn.Conv1d(32,32,3,padding=1), nn.ReLU(),
            nn.Conv1d(32,32,3,padding=1), nn.ReLU()
        )
        self.head = nn.Linear(32,1)   # predicts per-sample reward
    def forward(self,b,mask,dna_onehot):
        bx=self.bnet(b.transpose(1,2))
        pooled=(bx*mask.unsqueeze(1)).sum(-1)/mask.sum(-1,keepdim=True) # [B,16]
        L=dna_onehot.size(1)
        pooled=pooled.unsqueeze(-1).expand(-1,-1,L)                    # [B,16,L]
        dx=self.dnet(dna_onehot.transpose(1,2))                        # [B,16,L]
        x=self.mix(torch.cat([pooled,dx],1)).mean(-1)                  # GAP → [B,32]
        return self.head(x).squeeze(1)                                 # [B]

# ───────────────────── Helper: game→target ─────────────────
def game_to_target(s):
    return torch.clamp(torch.log1p(s)/math.log1p(K_TETRIS_MAX),0,1)    # [B]

# ───────────────────── Training function ───────────────────
def train_model(
    data_path: str = "tetris_data.json",
    log_csv_path: str = "training_log_ac.csv",
):
    # ───────── dataset & loaders ─────────
    ds = TetrisDataset(data_path)
    vn = max(1, int(len(ds) * VAL_SPLIT))
    tn = len(ds) - vn
    tr, vl = random_split(ds, [tn, vn])
    dl_tr = DataLoader(tr, BATCH_SIZE, shuffle=True, collate_fn=collate)
    dl_vl = DataLoader(vl, BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(f"Total: {len(ds)}, train: {tn}, val: {vn}")

    # ───────── models & optims ──────────
    actor  = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)          # outputs one scalar per sample
    opt_a  = torch.optim.Adam(actor.parameters(),  lr=LR)
    opt_c  = torch.optim.Adam(critic.parameters(), lr=LR)

    fields = ["epoch","temp",
              "train_corr","val_corr",
              "train_corr_loss","val_corr_loss",
              "actor_loss","critic_loss","design_var"]
    rows = []

    for ep in range(1, EPOCHS+1):
        # temperature
        temp = 1 + min(ep / EPOCHS / TEMP_RATE, 1) * (TEMP_MAX - 1) \
               if TEMP_ENABLED else 1

        actor.train(); critic.train()
        n_samp = 0
        s_corr = s_var = s_a = s_c = 0.0

        for i, (boards, mask, scores) in enumerate(dl_tr):
            boards, mask, scores = [t.to(DEVICE) for t in (boards, mask, scores)]

            # ── actor forward & sampling ──
            logits = actor(boards, mask)
            dist   = torch.distributions.Categorical(
                        F.softmax(logits / temp, dim=-1))
            sample = dist.sample()                      # [B, L]
            logp   = dist.log_prob(sample).sum(-1)      # [B]
            onehot = F.one_hot(sample, 4).float()       # [B, L, 4]

            # ── black-box design score ──
            with torch.no_grad():
                p = pwm_score(onehot, conv_ppar, PPAR_MAX)  # [B]
                n = pwm_score(onehot, conv_nkfb, NFKB_MAX)
            design = (p - n + 1) / 2                       # [B]
            target = game_to_target(scores)                 # [B]

            # ── batch Pearson correlation (reward) ──
            d_cent = design - design.mean()
            t_cent = target - target.mean()
            denom  = torch.sqrt((d_cent**2).mean() *
                                (t_cent**2).mean()) + 1e-8
            corr   = (d_cent * t_cent).mean() / denom       # scalar
            reward = corr                                   # maximise corr

            # ── critic update ──
            pred_batch = critic(boards, mask, onehot)       # [B]
            baseline   = pred_batch.mean()                  # scalar
            c_loss = F.mse_loss(pred_batch, torch.full_like(pred_batch, reward))
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()

            # ── actor update ──
            advantage = reward - baseline.detach()
            a_loss = -(advantage * logp.mean()) \
                     - ENTROPY_COEF * dist.entropy().mean()
            opt_a.zero_grad(); a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_a.step()
            probs = F.softmax(logits / temp, dim=-1)
            # if i % 200 == 0:
            #     print("logp:\n", probs[0][:5])  # prints 5 rows of 4 logits each
            #     print("onehot: ", onehot[0][:5])
            #     print("Design: ", design)
            #     print("Design var: ",  design.var())
            # ── bookkeeping ──
            bsz          = boards.size(0)
            n_samp      += bsz
            s_corr      += corr * bsz
            s_var       += design.var().item() * bsz
            s_a         += a_loss.item() * bsz
            s_c         += c_loss.item() * bsz

        train_corr       = s_corr / n_samp
        train_corr_loss  = 1 - train_corr
        design_var       = s_var  / n_samp
        actor_loss_ep    = s_a    / n_samp
        critic_loss_ep   = s_c    / n_samp

        # ───── validation ─────
        actor.eval(); val_corr_sum = 0.0; n_val = 0
        with torch.no_grad():
            for boards, mask, scores in dl_vl:
                boards, mask, scores = [t.to(DEVICE) for t in (boards, mask, scores)]
                logits = actor(boards, mask)
                onehot = F.one_hot(logits.argmax(-1), 4).float()
                p = pwm_score(onehot, conv_ppar, PPAR_MAX)
                n = pwm_score(onehot, conv_nkfb, NFKB_MAX)
                design = (p - n + 1) / 2
                target = game_to_target(scores)

                d_cent = design - design.mean()
                t_cent = target - target.mean()
                denom  = torch.sqrt((d_cent**2).mean() *
                                    (t_cent**2).mean()) + 1e-8
                corr_batch = (d_cent * t_cent).mean() / denom

                val_corr_sum += corr_batch.item() * boards.size(0)
                n_val        += boards.size(0)
        val_corr      = val_corr_sum / n_val
        val_corr_loss = 1 - val_corr

        # ───── print & log ─────
        print(f"Ep {ep}/{EPOCHS} temp={temp:.2f}  "
              f"train_corr={train_corr:.3f}  val_corr={val_corr:.3f}  "
              f"train_corr_loss={train_corr_loss:.4f}  val_corr_loss={val_corr_loss:.4f}  "
              f"actor_loss={actor_loss_ep:.4f}  critic_loss={critic_loss_ep:.4f}  "
              f"design_var={design_var:.4f}")

        rows.append(dict(epoch=ep, temp=temp,
                         train_corr=train_corr, val_corr=val_corr,
                         train_corr_loss=train_corr_loss,
                         val_corr_loss=val_corr_loss,
                         actor_loss=actor_loss_ep,
                         critic_loss=critic_loss_ep,
                         design_var=design_var))

    # ───── save artefacts & CSV ─────
    torch.save(actor.state_dict(), ACTOR_PATH)
    torch.save(critic.state_dict(), CRITIC_PATH)
    dummy_b, dummy_m = torch.zeros(1, 30, 10), torch.ones(1, 30)
    onnx_export(actor, (dummy_b, dummy_m), ONNX_PATH,
                opset_version=13,
                input_names=["board", "mask"], output_names=["dna_logits"])
    with open(log_csv_path, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fields)
        csv_writer.writeheader(); csv_writer.writerows(rows)
    print("Saved actor, critic, ONNX model, and training log.")

# ───────────────────── Quick inference util ───────────────
def infer(n=5,data_path="tetris_data.json"):
    actor=Actor().to(DEVICE)
    actor.load_state_dict(torch.load(ACTOR_PATH,map_location=DEVICE)); actor.eval()
    ds=TetrisDataset(data_path)
    top=sorted(ds.samples,key=lambda e:e["score"],reverse=True)[:n]

    for e in top:
        b=torch.tensor(e["game_matrix"],dtype=torch.float32,device=DEVICE).unsqueeze(0)
        m=torch.ones(1,b.size(1),device=DEVICE)
        with torch.no_grad():
            logits=actor(b,m)
        onehot=F.one_hot(logits.argmax(-1),4).float()
        p=pwm_score(onehot,conv_ppar,PPAR_MAX).item()
        n_=pwm_score(onehot,conv_nkfb,NFKB_MAX).item()
        design=(p-n_+1)/2
        norm=min(1., math.log1p(e["score"])/math.log1p(K_TETRIS_MAX))
        dna=''.join("ACGT"[i] for i in logits.argmax(-1).squeeze(0).cpu().numpy())
        print(f"Score {e['score']}  norm {norm:.3f}  design {design:.3f}\n{dna}\n")

