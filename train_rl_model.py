"""
End-to-end Actor-Critic trainer + inference utility for mapping Tetris boards → DNA sequences
and aligning PPARγ activation (good) vs NF-κB activation (bad).
"""
import json
import math
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.onnx import export as onnx_export

# ──────────────────────────── Hyper-parameters ───────────────────────────── #
LR = 2e-3 # LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 300
CONV_CHANNELS = 32
NOISE_STD = 0.1
MLP_HIDDEN = 32
DNA_LEN = 196
DNA_MLP_HIDDEN = 64 ## Was using 256 (this is for old critic)
K_TETRIS_MAX = 300
VAL_SPLIT = 0.05
ONNX_PATH = "model_rl.onnx"
STATE_PATH_ACT = "model_rl.pt"
STATE_PATH_CRIT = "critic_rl.pt"
DEVICE = "cpu"

USE_FULL_CRITIC       = True  # if True: actor and critic both use 3‐head outputs

# USE_PROXY_QUANTIZED = True # Toggle to use the quantized corr loss as additional objective
# USE_CRITIC_SURROGATE = True # Toggle to # Optional critic-guided surrogate loss (decaying)
# USE_SCORE_PRED      = False  # Toggle to enable/disable 2nd critic game score target
# ENABLE_CRITIC_AUGMENTATION = False  # Toggle to enable/disable skipping actor steps
# CRITIC_NOISE_STD = 2.0  # Logit noise during actor-freeze steps

# BETA = 0.02 # Actor Entropy bonus (decaying)
# VAR_WEIGHT = 0.6
VAR_WEIGHT = 1
TEMP_ENABLED = False # Whether to use schedule or not
TEMP_RATE = 0.8
TEMP_MAX = 5.0

# ──────────────────────────── Motif loading ─────────────────────────────── #
MOTIF_DIR = Path("motifs")
PPAR_ID = "MA0065.2"
NFKB_ID = "MA0105.4"

def _parse_meme(filepath: Path):
    lines = filepath.read_text().splitlines()
    i = next((i for i, l in enumerate(lines) if l.startswith("letter-probability")), None)
    if i is None:
        raise RuntimeError(f"No letter-probability matrix in {filepath}")
    w = int(lines[i].split("w=")[1].split()[0])
    matrix = [list(map(float, lines[j].split())) for j in range(i+1, i+1+w)]
    return torch.tensor(matrix, dtype=torch.float32)

PPAR_PWM = _parse_meme(MOTIF_DIR / f"{PPAR_ID}.meme").to(DEVICE)
NFKB_PWM = _parse_meme(MOTIF_DIR / f"{NFKB_ID}.meme").to(DEVICE)
PPAR_MAX = float(PPAR_PWM.max(dim=1).values.sum())
NFKB_MAX = float(NFKB_PWM.max(dim=1).values.sum())
print(f"Loaded PWMs: PPAR width={PPAR_PWM.shape[0]}, NFKB width={NFKB_PWM.shape[0]}")

# ──────────────────────────── Dataset & Collation ───────────────────────── #
class TetrisDataset(Dataset):
    def __init__(self, path):
        self.samples = json.load(open(path))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        e = self.samples[i]
        return torch.tensor(e['game_matrix'], dtype=torch.float32), torch.tensor(e['score'], dtype=torch.float32)


def collate_batch(batch):
    boards, scores = zip(*batch)
    max_len = max(b.shape[0] for b in boards)
    padded, masks = [], []
    for b in boards:
        pad_len = max_len - b.shape[0]
        if pad_len:
            padded.append(torch.cat([b, torch.zeros(pad_len, b.shape[1])], dim=0))
            masks.append(torch.cat([torch.ones(b.shape[0]), torch.zeros(pad_len)], dim=0))
        else:
            padded.append(b)
            masks.append(torch.ones(b.shape[0]))
    return torch.stack(padded), torch.stack(masks), torch.stack(scores)

# ──────────────────────────── Actor Model ───────────────────────────────── #
class TetrisToDNAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, CONV_CHANNELS//2, 3, padding=1)
        self.conv2 = nn.Conv1d(10, CONV_CHANNELS//2, 7, padding=3, stride=2)
        self.fc1   = nn.Linear(CONV_CHANNELS, MLP_HIDDEN)
        self.fc_logits = nn.Linear(MLP_HIDDEN, DNA_LEN*4)
        # always include supervised heads when USE_FULL_CRITIC=True
        self.fc_game   = nn.Linear(MLP_HIDDEN, 1)
        self.fc_design = nn.Linear(MLP_HIDDEN, 1)

    def forward(self, x, mask):
        x1 = F.relu(self.conv1(x.transpose(1,2)))
        x2 = F.relu(self.conv2(x.transpose(1,2)))
        m1 = mask.unsqueeze(1)
        m2 = mask[:, ::2].unsqueeze(1)
        p1 = (x1 * m1).sum(-1) / m1.sum(-1)
        p2 = (x2 * m2).sum(-1) / m2.sum(-1)
        z  = torch.cat([p1, p2], dim=1)
        if self.training:
            z = z + torch.randn_like(z) * NOISE_STD
        h = F.relu(self.fc1(z))
        logits      = self.fc_logits(h).view(-1, DNA_LEN, 4)
        game_pred   = self.fc_game(h).squeeze(-1)
        design_pred = self.fc_design(h).squeeze(-1)
        return logits, game_pred, design_pred

# ──────────────────────────── Critic Model ──────────────────────────────── #
class BoardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(10, CONV_CHANNELS//2, 3, padding=1)
        self.conv2 = nn.Conv1d(10, CONV_CHANNELS//2, 7, padding=3, stride=2)
        self.fc = nn.Linear(CONV_CHANNELS, CONV_CHANNELS)

    def forward(self, x, mask):
        x1 = F.relu(self.conv1(x.transpose(1,2)))
        x2 = F.relu(self.conv2(x.transpose(1,2)))
        m1 = mask.unsqueeze(1)
        m2 = mask[:, ::2].unsqueeze(1)
        p1 = (x1 * m1).sum(-1) / m1.sum(-1)
        p2 = (x2 * m2).sum(-1) / m2.sum(-1)
        z = torch.cat([p1, p2], dim=1)
        if self.training:
            z = z + torch.randn_like(z) * NOISE_STD
        return F.relu(self.fc(z))

class DNAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DNA_LEN*4, DNA_MLP_HIDDEN), nn.ReLU(),
            nn.Linear(DNA_MLP_HIDDEN, CONV_CHANNELS), nn.ReLU()
        )
    def forward(self, dna):
        return self.net(dna.view(dna.size(0), -1))

class CriticOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.be = BoardEncoder()
        self.de = DNAEncoder()
        self.h  = nn.Linear(CONV_CHANNELS * 2, 2)  # Output: [design_pred, score_pred]

    def forward(self, b, m, d):
        z = torch.cat([self.be(b, m), self.de(d)], dim=1)
        return self.h(z)  # Returns shape [B, 2]


# ──────────────────────────── MixedCritic Hyperparameters ────────────────────────────── #
# Board encoder
BOARD_IN_CHANNELS   = 10
BOARD_CONV_CHANNELS = 16   # ↑ from 32
NUM_BOARD_CONVS     = 2    # ↑ from 2

# DNA encoder
DNA_IN_CHANNELS     = 4
DNA_CONV_CHANNELS   = 16   # ↑ from 32
NUM_DNA_CONVS       = 2    # ↑ from 2

# Mixer
MIXER_CHANNELS      = 32   # ↑ from 64
NUM_MIXER_CONVS     = 3    # ↑ from 2

# Final MLP head
MLP_HIDDEN_SIZE     = 32  # ↑ from 128
NUM_MLP_LAYERS      = 2    # ↑ from 2

# Kernel sizes (can be list of ints if you want different sizes per layer)
BOARD_KERNEL_SIZE   = 3
DNA_KERNEL_SIZE     = 5
MIXER_KERNEL_SIZE   = 3

# ──────────────────────────── MixedCritic Definition ─────────────────────── #
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # reuse existing conv encoders and mixer setup
        board_convs = []
        in_ch = 10
        for _ in range(NUM_BOARD_CONVS):
            board_convs += [nn.Conv1d(in_ch, BOARD_CONV_CHANNELS, BOARD_KERNEL_SIZE, padding=BOARD_KERNEL_SIZE//2), nn.ReLU()]
            in_ch = BOARD_CONV_CHANNELS
        self.board_net = nn.Sequential(*board_convs)

        dna_convs = []
        in_ch = DNA_IN_CHANNELS
        for _ in range(NUM_DNA_CONVS):
            dna_convs += [nn.Conv1d(in_ch, DNA_CONV_CHANNELS, DNA_KERNEL_SIZE, padding=DNA_KERNEL_SIZE//2), nn.ReLU()]
            in_ch = DNA_CONV_CHANNELS
        self.dna_net = nn.Sequential(*dna_convs)

        mixer_convs = []
        in_ch = BOARD_CONV_CHANNELS + DNA_CONV_CHANNELS
        for _ in range(NUM_MIXER_CONVS):
            mixer_convs += [nn.Conv1d(in_ch, MIXER_CHANNELS, MIXER_KERNEL_SIZE, padding=MIXER_KERNEL_SIZE//2), nn.ReLU()]
            in_ch = MIXER_CHANNELS
        self.mixer_net = nn.Sequential(*mixer_convs)

        # head reduced to single baseline output
        self.head = nn.Linear(in_ch, 1)

    def forward(self, board, mask, dna_logits):
        b = board.transpose(1,2)
        b = self.board_net(b)
        pooled = (b * mask.unsqueeze(1)).sum(-1) / mask.sum(-1, keepdim=True)
        dna_oh = F.one_hot(dna_logits.argmax(-1), 4).float().transpose(1,2)
        d = self.dna_net(dna_oh)
        cat = torch.cat([pooled.unsqueeze(-1).expand(-1, pooled.size(-1), d.size(-1)), d], dim=1)
        m = self.mixer_net(cat)
        gap = m.mean(-1)
        baseline = self.head(gap).squeeze(-1)
        return baseline
    

# ──────────────────────────── PWM & Loss Utilities ──────────────────────── #
conv_ppar = nn.Conv1d(4,1,PPAR_PWM.shape[0], bias=False)
conv_nkfb = nn.Conv1d(4,1,NFKB_PWM.shape[0], bias=False)
conv_ppar.weight.data.copy_(PPAR_PWM.t().unsqueeze(0))
conv_nkfb.weight.data.copy_(NFKB_PWM.t().unsqueeze(0))
for p in conv_ppar.parameters(): p.requires_grad = False
for p in conv_nkfb.parameters(): p.requires_grad = False

def pwm_score(d, conv, mx, temp=1.0):
    # d: one-hot or logits
    p = F.softmax(d/temp, -1) if d.dim()==3 else d
    return torch.clamp(conv(p.transpose(1,2)).squeeze(1).max(-1).values / mx, 0, 1)

def game_to_target(s):
    return torch.clamp(torch.log1p(s) / math.log1p(K_TETRIS_MAX), 0, 1)

def corr_loss(p, t):
    pc, tc = p - p.mean(), t - t.mean()
    return 1 - (pc*tc).mean() / (torch.sqrt((pc**2).mean() * (tc**2).mean()) + 1e-8)

# ──────────────────────────── Training Function ─────────────────────────── #
def train_model_rl(
    data_path: str = "tetris_data.json",
    log_csv_path: str = "training_log_rl.csv"
):
    ds = TetrisDataset(data_path)
    total = len(ds)
    vn = max(1, int(total * VAL_SPLIT))
    tn = total - vn
    tr, vl = random_split(ds, [tn, vn])
    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    dl_vl = DataLoader(vl, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    actor  = TetrisToDNAModel().to(DEVICE)
    critic = Critic().to(DEVICE)
    print(f"Created actor ({sum(p.numel() for p in actor.parameters()):,} params) "
          f"and critic ({sum(p.numel() for p in critic.parameters()):,} params)")

    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)

    fields = [
        "epoch", "temp",
        "actor_loss", "critic_loss",
        "train_corr", "val_corr",
        "design_var", "train_quant_loss", "val_quant_loss"
    ]
    logs = []

    for e in range(1, EPOCHS + 1):
        actor.train()
        critic.train()

        # temperature schedule
        temp = 1.0 + min((e / EPOCHS) / TEMP_RATE, 1.0) * (TEMP_MAX - 1.0) if TEMP_ENABLED else 1.0

        sum_a_loss = 0.0
        sum_c_loss = 0.0
        sum_var    = 0.0
        sum_tq     = 0.0
        d_all, t_all = [], []

        for boards, mask, scores in dl_tr:
            boards, mask, scores = boards.to(DEVICE), mask.to(DEVICE), scores.to(DEVICE)

            # Actor forward
            logits, a_game, a_design = actor(boards, mask)

            # Sample a DNA sequence (policy)
            probs   = F.softmax(logits / temp, dim=-1)
            dist    = torch.distributions.Categorical(probs)
            samples = dist.sample()                      # (B, L)
            logp    = dist.log_prob(samples).sum(dim=-1) # (B,)

            # One-hot encode the sampled sequence for critic & scoring
            # sampled_oh = F.one_hot(samples, num_classes=4).float()  # (B, L, 4)

            # Compute reward on sampled sequence
            with torch.no_grad():
                p = pwm_score(probs, conv_ppar, PPAR_MAX, temp)
                n = pwm_score(probs, conv_nkfb, NFKB_MAX, temp)
                design = (p - n + 1) / 2
                var_bonus = design.var()
                total_reward = design + VAR_WEIGHT * var_bonus

            # ----- Critic update -----
            # Critic should see exactly the sampled sequence
            baseline = critic(boards, mask, probs)  # (B,)
            c_loss = F.mse_loss(baseline, total_reward)
            opt_c.zero_grad()
            c_loss.backward()
            opt_c.step()

            # ----- Actor update -----
            # Supervised regression heads
            target = game_to_target(scores)
            loss_game   = F.mse_loss(a_game,   target.detach())
            loss_design = F.mse_loss(a_design, design.detach())

            # Policy-gradient loss
            adv = total_reward - baseline.detach()
            pg_loss = -(adv * logp).mean()

            # Combine losses, downweight supervised terms
            actor_loss = pg_loss + 0.1 * loss_game + 0.1 * loss_design

            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            # Accumulate training metrics
            sum_a_loss += actor_loss.item() * boards.size(0)
            sum_c_loss += c_loss.item()       * boards.size(0)
            sum_var    += var_bonus.item()    * boards.size(0)

            # Quantized (“hard”) correlation loss on sampled sequence
            dq = design
            sum_tq += corr_loss(dq, target).item() * boards.size(0)

            d_all.append(design)
            t_all.append(target)

        # Epoch metrics
        train_corr      = 1.0 - corr_loss(torch.cat(d_all), torch.cat(t_all)).item()
        actor_loss_avg  = sum_a_loss  / tn
        critic_loss_avg = sum_c_loss  / tn
        design_var      = sum_var     / tn
        train_quant     = sum_tq      / tn

        # ----- Validation -----
        actor.eval()
        dv, tv = [], []
        val_tq = 0.0
        with torch.no_grad():
            for b, m, s in dl_vl:
                b, m, s = b.to(DEVICE), m.to(DEVICE), s.to(DEVICE)

                logits, a_game, a_design = actor(b, m)
                probs  = F.softmax(logits / temp, dim=-1)

                # Soft design score (expected PWM)
                p_soft = pwm_score(probs, conv_ppar, PPAR_MAX)
                n_soft = pwm_score(probs, conv_nkfb, NFKB_MAX)
                design_soft = (p_soft - n_soft + 1) / 2

                # Quantized design score (argmax)
                one_hot = torch.zeros_like(probs).scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
                p_q = pwm_score(one_hot, conv_ppar, PPAR_MAX)
                n_q = pwm_score(one_hot, conv_nkfb, NFKB_MAX)
                design_q = (p_q - n_q + 1) / 2

                target_v = game_to_target(s)

                dv.append(design_soft)
                tv.append(target_v)
                val_tq += corr_loss(design_q, target_v).item() * b.size(0)

        val_corr  = 1.0 - corr_loss(torch.cat(dv), torch.cat(tv)).item()
        val_quant = val_tq / vn

        print(
            f"Epoch {e}/{EPOCHS}   "
            f"temp={temp:.2f}   "
            f"actor_loss={actor_loss_avg:.4f}   "
            f"critic_loss={critic_loss_avg:.4f}   "
            f"train_corr={train_corr:.3f}   "
            f"val_corr={val_corr:.3f}   "
            f"design_var={design_var:.4f}   "
            f"train_quant_loss={train_quant:.4f}   "
            f"val_quant_loss={val_quant:.4f}"
        )

        logs.append({
            "epoch":               e,
            "temp":                temp,
            "actor_loss":          actor_loss_avg,
            "critic_loss":         critic_loss_avg,
            "train_corr":          train_corr,
            "val_corr":            val_corr,
            "design_var":          design_var,
            "train_quant_loss":    train_quant,
            "val_quant_loss":      val_quant,
        })

    # Save models
    torch.save(actor.state_dict(), STATE_PATH_ACT)
    torch.save(critic.state_dict(), STATE_PATH_CRIT)

    # Export ONNX
    dummy_board = torch.zeros(1, 30, 10)
    dummy_mask  = torch.ones(1, 30)
    onnx_export(
        actor,
        (dummy_board, dummy_mask),
        ONNX_PATH,
        opset_version=13,
        input_names=["board", "mask"],
        output_names=["dna_logits"],
    )

    # Write logs
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(logs)

    print("Saved actor, critic, and ONNX models.")





# ──────────────────────────── Inference Function ────────────────────────── #
def infer_rl(n=100, data_path="tetris_data.json"):
    entries = TetrisDataset(data_path).samples
    top = sorted(entries, key=lambda e: e['score'], reverse=True)[:n]
    actor = TetrisToDNAModel().to(DEVICE)
    actor.load_state_dict(torch.load(STATE_PATH_ACT, map_location=DEVICE))
    total_params = sum(p.numel() for p in actor.parameters())
    print(f"Loaded actor with {total_params:,} parameters")
    actor.eval()

    temp = 5  # fixed temperature for inference

    for e in top:
        b = torch.tensor(e['game_matrix'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        m = torch.ones(1, b.size(1), device=DEVICE)
        with torch.no_grad():
            out = actor(b, m)
            if USE_FULL_CRITIC:
                logits, pred_game, pred_design = out
            else:
                logits = out

            probs = F.softmax(logits / temp, dim=-1)

            # sample DNA string
            dna_str = ''.join(
                np.random.choice(['A','C','G','T'], p=p)
                for p in probs.squeeze(0).cpu().numpy()
            )

            # compute PWM scores with matching temp
            ppar   = pwm_score(logits, conv_ppar, PPAR_MAX, temp=temp).item()
            nfkb   = pwm_score(logits, conv_nkfb, NFKB_MAX, temp=temp).item()
            design = (ppar - nfkb + 1) / 2

        # print results
        print(f"Score {e['score']}: Design {design:.3f}")
        if USE_FULL_CRITIC:
            print(f"Predicted game score: {pred_game.item():.3f}, predicted design score: {pred_design.item():.3f}")
        print("DNA:", dna_str)
        print("-")


#Epoch 60/300   temp=2.00   actor_loss=-0.0717   critic_loss=0.2843   train_corr=0.755   val_corr=0.767   design_var=0.0051   train_quant_loss=0.2476   val_quant_loss=0.2301