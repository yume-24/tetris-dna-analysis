# train_model.py
"""
End-to-end trainer + inference utility for mapping Tetris boards → DNA sequences
and aligning PPARγ activation (good) vs NF-κB activation (bad).
Invoke directly via `python train_model.py`.

────────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────────
Note: For the 30k CNN tetris model, these were the values:
# ──────────────────────────── Hyper-parameters ────────────────────────────── #
LR              = 1e-3      # learning rate
BATCH_SIZE      = 64        # samples per batch
EPOCHS          = 300        # training epochs
CONV_CHANNELS   = 32        # Conv1d output channels # was 32
NOISE_STD       = 0.1       # Added after the global average pool
KERNEL_SIZE     = 5         # Conv1d kernel width
MLP_HIDDEN      = 128       # hidden units in MLP
DNA_LEN         = 196        # output DNA sequence length (>= PWM width)  196 = 14x14
K_TETRIS_MAX    = 300        # for target normalization
VAL_SPLIT       = 0.05       # fraction for validation
ONNX_PATH       = "model.onnx"
STATE_PATH      = "model.pt"
DEVICE          = "cpu"     # or "cuda" if available
Epoch 400/400  temp=5.00   train_loss=1.0046   val_loss=1.0146   train_var=0.0197 
────────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────────
"""
import json
import math
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.onnx import export as onnx_export

# ──────────────────────────── Hyper-parameters ────────────────────────────── #
LR              = 1e-4      # learning rate
BATCH_SIZE      = 64        # samples per batch
EPOCHS          = 300 #300        # training epochs
CONV_CHANNELS   = 32        # Conv1d output channels # was 32
NOISE_STD       = 0.1       # Added after the global average pool
KERNEL_SIZE     = 5         # Conv1d kernel width
MLP_HIDDEN      = 128       # hidden units in MLP
DNA_LEN         = 196        # output DNA sequence length (>= PWM width)  196 = 14x14
K_TETRIS_MAX    = 300        # for target normalization
VAL_SPLIT       = 0.05       # fraction for validation
ONNX_PATH       = "model.onnx"
STATE_PATH      = "model.pt"
DEVICE          = "cpu"     # or "cuda" if available

VAR_WEIGHT = 0.6  # encourage diversity in design scores
USE_REINFORCE = True
USE_SIMPLE_MLP = False  # set to False to use CNN

MLP_HIDDEN_STRIDER = 72
MLP_LAYERS = 6
# MLP_HIDDEN = 10
# MLP_LAYERS = 3





# Temperature annealing for DNA softmax
TEMP_ENABLED    = True
TEMP_RATE       = 0.8       # fraction of epochs to reach TEMP_MAX
TEMP_MAX        = 5.0       # max temperature

# ──────────────────────────── Motif loading ───────────────────────────────── #
MOTIF_DIR = Path("motifs")
PPAR_ID   = "MA0065.2"  # PPARG::RXRA heterodimer
NFKB_ID   = "MA0105.4"  # NFKB1

def _parse_meme(filepath: Path):
    lines = filepath.read_text().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("letter-probability"):
        i += 1
    if i == len(lines):
        raise RuntimeError(f"No letter-probability matrix in {filepath}")
    header = lines[i]
    w = int(header.split("w=")[1].split()[0])
    matrix = []
    for j in range(i+1, i+1+w):
        row = [float(x) for x in lines[j].split()]
        matrix.append(row)
    return torch.tensor(matrix, dtype=torch.float32)

PPAR_PWM = _parse_meme(MOTIF_DIR / f"{PPAR_ID}.meme").to(DEVICE)
NFKB_PWM = _parse_meme(MOTIF_DIR / f"{NFKB_ID}.meme").to(DEVICE)
PPAR_WIDTH = PPAR_PWM.shape[0]
NFKB_WIDTH = NFKB_PWM.shape[0]
print(f"Loaded PPAR PWM length={PPAR_WIDTH}, NF-κB PWM length={NFKB_WIDTH}")
PPAR_MAX = float(PPAR_PWM.max(dim=1).values.sum())
NFKB_MAX = float(NFKB_PWM.max(dim=1).values.sum())

# ──────────────────────────── Dataset ─────────────────────────────────────── #
class TetrisDataset(Dataset):
    def __init__(self, json_path: str):
        raw = json.load(open(json_path))
        self.samples = raw

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        board = torch.tensor(entry["game_matrix"], dtype=torch.float32)
        score = torch.tensor(entry["score"], dtype=torch.float32)
        return board, score

def collate_batch(batch):
    boards, scores = zip(*batch)
    max_len = max(b.shape[0] for b in boards)
    padded, masks = [], []
    for b in boards:
        pad_len = max_len - b.shape[0]
        if pad_len:
            pad = torch.zeros(pad_len, b.shape[1])
            padded.append(torch.cat([b, pad], dim=0))
            masks.append(torch.cat([torch.ones(b.shape[0]), torch.zeros(pad_len)]))
        else:
            padded.append(b)
            masks.append(torch.ones(b.shape[0]))
    return torch.stack(padded), torch.stack(masks), torch.stack(scores)

# ──────────────────────────── Model ───────────────────────────────────────── #
class TetrisToDNAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            10,         # in_channels: number of input feature maps (10 here)
            CONV_CHANNELS // 2,         # out_channels: number of filters/kernels, output feature maps (16)
            3,          # kernel_size: width of the convolution window (3)
            padding=1,  # padding: adds 1 zero on each side to keep output length same as input (for stride=1)
            stride=1    # stride: step size for sliding the kernel (1 means move one element at a time)
        )

        self.conv2 = nn.Conv1d(10, CONV_CHANNELS // 2, 7, padding=3, stride=2)

        self.fc1 = nn.Linear(CONV_CHANNELS, MLP_HIDDEN)  # 16+16 channels concatenated after pooling
        self.fc2 = nn.Linear(MLP_HIDDEN, DNA_LEN * 4)

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # (B,10,N)
        x1 = F.relu(self.conv1(x))  # (B,16,N)
        x2 = F.relu(self.conv2(x))  # (B,16,N//2)
        mask1 = mask.unsqueeze(1)   # (B,1,N)
        mask2 = mask[:, ::2].unsqueeze(1)  # downsampled mask for stride=2 conv

        pooled1 = (x1 * mask1).sum(-1) / mask1.sum(-1)  # (B,16)
        pooled2 = (x2 * mask2).sum(-1) / mask2.sum(-1)  # (B,16)

        x = torch.cat([pooled1, pooled2], dim=1)  # (B,32)

        if self.training:
            x = x + torch.randn_like(x) * NOISE_STD

        x = F.relu(self.fc1(x)) # Hidden layer 1

        return self.fc2(x).view(-1, DNA_LEN, 4)



# GlobalResidualMLP:
# A lightweight, fully-connected architecture for variable-length input perception.
# Each row (10-D Tetris board slice) is first encoded via a shared MLP.
# Then, over multiple layers, we inject global context by computing the masked mean
# across rows and adding it back into each token embedding (residual-style).
# This mechanism allows all positions to interact via shared global feedback
# without using convolution or attention.
# Final outputs are pooled over time and decoded into DNA logits.
import math  # at top if not already imported

class GlobalResidualMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=MLP_HIDDEN_STRIDER, layers=MLP_LAYERS):
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        self.mix_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for _ in range(layers)
        ])
        self.fc1 = nn.Linear(hidden_dim, MLP_HIDDEN_STRIDER)
        self.fc2 = nn.Linear(MLP_HIDDEN_STRIDER, DNA_LEN * 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        x = self.row_mlp(x)
        for layer in self.mix_layers:
            global_avg = (x * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze(-1)
            x = layer(x + global_avg)
            if self.training:
                x = x + torch.randn_like(x) * NOISE_STD
        z = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        z = F.relu(self.fc1(z))
        return self.fc2(z).view(-1, DNA_LEN, 4)






# ──────────────────────────── PWM scorer ─────────────────────────────────── #
conv_ppar = nn.Conv1d(4, 1, PPAR_WIDTH, bias=False)
conv_nkfb = nn.Conv1d(4, 1, NFKB_WIDTH, bias=False)
conv_ppar.weight.data.copy_(PPAR_PWM.t().unsqueeze(0))
conv_nkfb.weight.data.copy_(NFKB_PWM.t().unsqueeze(0))
for p in conv_ppar.parameters(): p.requires_grad = False
for p in conv_nkfb.parameters(): p.requires_grad = False

def pwm_score(dna_logits, conv, max_score, temperature=1.0):
    # dna_logits: (B,L,4)
    probs = F.softmax(dna_logits / temperature, dim=-1)  # (B,L,4)
    seq = probs.transpose(1, 2)                         # (B,4,L)
    scores = conv(seq).squeeze(1).max(-1).values        # (B,)
    return torch.clamp(scores / max_score, 0, 1)

# ──────────────────────────── Loss ────────────────────────────────────────── #
def game_to_target(s):
    return torch.clamp(torch.log1p(s) / math.log1p(K_TETRIS_MAX), 0, 1)

def loss_fn(logits, scores, temperature=1.0):
    p = pwm_score(logits, conv_ppar, PPAR_MAX, temperature)
    n = pwm_score(logits, conv_nkfb, NFKB_MAX, temperature)
    design = (p - n + 1) / 2
    target = game_to_target(scores.to(logits.device))
    # return F.mse_loss(design, target)
    corr_l = corr_loss(design, target)
    var_l = -design.var()  # encourage higher variance in design scores
    # var_weight = 0.3 #For the older ones
    var_weight = 0.6
    return 1 + corr_l + var_weight * var_l


def corr_loss(pred, target):
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    numerator = (pred_centered * target_centered).mean()
    denominator = torch.sqrt((pred_centered ** 2).mean() * (target_centered ** 2).mean())
    corr = numerator / (denominator + 1e-8)
    return 1 - corr

# In the RL setting, we want to maximize the correlation between predicted biological
# design scores and original Tetris game scores. Since true Pearson correlation is not
# decomposable per-sample, we approximate each sample’s contribution using the centered
# product (design_i - mean) * (target_i - mean). This aligns with the gradient of the 
# correlation objective while remaining usable as a REINFORCE reward signal.
# We normalize the rewards (zero mean, unit variance) for stability.


def train_model(data_path="tetris_data.json", log_csv_path="training_log.csv"):
    ds = TetrisDataset(data_path)
    total = len(ds)
    vn = max(1, int(total * VAL_SPLIT))
    tn = total - vn
    print(f"Total samples: {total}, train: {tn}, val: {vn}")
    tr, vl = random_split(ds, [tn, vn])
    dl_tr = DataLoader(tr, BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    dl_vl = DataLoader(vl, BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    model = GlobalResidualMLP().to(DEVICE) if USE_SIMPLE_MLP else TetrisToDNAModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    csv_fields = ["epoch", "temp", "train_loss", "val_loss", "train_var", "train_quant_loss", "val_quant_loss", "train_avg_reward"]
    log_rows = []

    RL_EPOCH_WARM_UP = 20

    for e in range(1, EPOCHS + 1):
        tot_var = 0.0
        total_samples = 0
        train_quant_loss_total = 0.0
        val_quant_loss_total = 0.0
        reward_total = 0.0

        temp = 1.0 + min((e / EPOCHS) / TEMP_RATE, 1.0) * (TEMP_MAX - 1.0) if TEMP_ENABLED else 1.0

        model.train()
        tot_loss = 0.0
        for boards, mask, scores in dl_tr:
            boards, mask, scores = boards.to(DEVICE), mask.to(DEVICE), scores.to(DEVICE)
            optim.zero_grad()
            logits = model(boards, mask)

            #"We found that using the batch-level Pearson correlation as a scalar 
            # reward outperformed noisy per-sample approximations, despite lacking explicit credit assignment."
            if USE_REINFORCE:
                probs = F.softmax(logits / temp, dim=-1)
                dist = torch.distributions.Categorical(probs)
                samples = dist.sample()
                one_hot = F.one_hot(samples, num_classes=4).float()

                with torch.no_grad():
                    ppar = pwm_score(one_hot, conv_ppar, PPAR_MAX)  # (B,)
                    nfkb = pwm_score(one_hot, conv_nkfb, NFKB_MAX)  # (B,)
                    design = (ppar - nfkb + 1) / 2                  # (B,)
                    target = game_to_target(scores.to(logits.device))  # (B,)

                    if e < RL_EPOCH_WARM_UP:
                        reward = design  # (B,)
                    else:
                        design_centered = design - design.mean()
                        target_centered = target - target.mean()
                        reward = (design_centered * target_centered).mean() / (
                            design_centered.std() * target_centered.std() + 1e-8
                        )  # scalar

                log_probs = dist.log_prob(samples).sum(dim=1)  # (B,)
                var_l = -design.var()

                if e < RL_EPOCH_WARM_UP:
                    loss = -((reward - reward.mean()) * log_probs).mean() + VAR_WEIGHT * var_l
                    reward_total += reward.sum().item()
                else:
                    loss = -(reward * log_probs).mean() + VAR_WEIGHT * var_l
                    reward_total += reward.item() * boards.size(0)
            else:
                loss = loss_fn(logits, scores, temperature=temp)

            probs = F.softmax(logits / temp, dim=-1)
            # print(f"[Epoch {e}] Softmax[0]:", probs[0, :5].detach().cpu())
            # print(f"[Epoch {e}] Var: {design.var().item():.6f}")
            loss.backward()
            optim.step()
            tot_loss += loss.item() * boards.size(0)

            with torch.no_grad():
                p = pwm_score(logits, conv_ppar, PPAR_MAX, temperature=temp)
                n = pwm_score(logits, conv_nkfb, NFKB_MAX, temperature=temp)
                design = (p - n + 1) / 2
                tot_var += design.var().item() * boards.size(0)
                total_samples += boards.size(0)

                probs = F.softmax(logits / temp, dim=-1)
                one_hot = torch.zeros_like(probs).scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
                p_q = pwm_score(one_hot, conv_ppar, PPAR_MAX)
                n_q = pwm_score(one_hot, conv_nkfb, NFKB_MAX)
                design_q = (p_q - n_q + 1) / 2
                target = game_to_target(scores)
                train_quant_loss_total += corr_loss(design_q, target).item() * boards.size(0)

        tot_loss /= tn
        avg_var = tot_var / total_samples
        train_quant_loss = train_quant_loss_total / tn
        train_avg_reward = reward_total / tn if USE_REINFORCE else 0.0

        model.eval()
        val_loss = 0.0
        for boards, mask, scores in dl_vl:
            boards, mask, scores = boards.to(DEVICE), mask.to(DEVICE), scores.to(DEVICE)
            with torch.no_grad():
                logits = model(boards, mask)
                val_loss += loss_fn(logits, scores, temperature=temp).item() * boards.size(0)

                probs = F.softmax(logits / temp, dim=-1)
                one_hot = torch.zeros_like(probs).scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
                p_q = pwm_score(one_hot, conv_ppar, PPAR_MAX)
                n_q = pwm_score(one_hot, conv_nkfb, NFKB_MAX)
                design_q = (p_q - n_q + 1) / 2
                target = game_to_target(scores)
                val_quant_loss_total += corr_loss(design_q, target).item() * boards.size(0)

        val_loss /= vn
        val_quant_loss = val_quant_loss_total / vn

        print(f"Epoch {e}/{EPOCHS}  temp={temp:.2f}   train_loss={tot_loss:.4f}   val_loss={val_loss:.4f}   "
              f"train_var={avg_var:.4f}   train_quant_loss={train_quant_loss:.4f}   "
              f"val_quant_loss={val_quant_loss:.4f}   train_avg_reward={train_avg_reward:.4f}")

        log_rows.append({
            "epoch": e,
            "temp": temp,
            "train_loss": tot_loss,
            "val_loss": val_loss,
            "train_var": avg_var,
            "train_quant_loss": train_quant_loss,
            "val_quant_loss": val_quant_loss,
            "train_avg_reward": train_avg_reward
        })

    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(model.state_dict(), STATE_PATH)
    dummy_board = torch.zeros(1, 30, 10)
    dummy_mask = torch.ones(1, 30)
    onnx_export(model, (dummy_board, dummy_mask), ONNX_PATH,
                opset_version=13, input_names=["board", "mask"], output_names=["dna_logits"])
    print("Saved model.pt and exported model.onnx")



def dna_probs_to_string(probs):
    bases = ['A', 'C', 'G', 'T']
    indices = [np.random.choice(4, p=p) for p in probs]
    return ''.join(bases[i] for i in indices), indices


def infer(n=100, data_path="tetris_data.json"):
    # Parameters
    temp = 10
    SCORE_WEIGHT_SOFT = 1
    SCORE_WEIGHT_ONEHOT = 0

    entries = TetrisDataset(data_path).samples
    # top_entries = entries[:n]
    top_entries = sorted(entries, key=lambda e: e["score"], reverse=True)[:n]

    model = GlobalResidualMLP().to(DEVICE) if USE_SIMPLE_MLP else TetrisToDNAModel().to(DEVICE)
    model.load_state_dict(torch.load(STATE_PATH, map_location=DEVICE))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {total_params:,} parameters")
    model.eval()

    print(f"Running inference with temperature={temp}\n")

    results = []

    for entry in top_entries:
        board = entry["game_matrix"]
        score = entry["score"]
        norm_score = min(1.0, math.log1p(score) / math.log1p(K_TETRIS_MAX))

        b = torch.tensor(board, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask = torch.ones(1, b.size(1), device=DEVICE)

        with torch.no_grad():
            logits = model(b, mask)
            probs = F.softmax(logits / temp, dim=-1)
            dna_probs = probs.squeeze(0).cpu().numpy()
            dna_str, sampled_indices = dna_probs_to_string(dna_probs)

            # Build sampled one-hot
            sampled_one_hot = torch.zeros(1, DNA_LEN, 4, device=DEVICE, dtype=logits.dtype)
            for i, idx in enumerate(sampled_indices):
                sampled_one_hot[0, i, idx] = 1.0

            # Score both soft and sampled
            ppar_soft = pwm_score(logits, conv_ppar, PPAR_MAX, temperature=temp).item()
            nfkb_soft = pwm_score(logits, conv_nkfb, NFKB_MAX, temperature=temp).item()

            ppar_hard = pwm_score(sampled_one_hot, conv_ppar, PPAR_MAX).item()
            nfkb_hard = pwm_score(sampled_one_hot, conv_nkfb, NFKB_MAX).item()

            # Weighted score
            ppar = SCORE_WEIGHT_SOFT * ppar_soft + SCORE_WEIGHT_ONEHOT * ppar_hard
            nfkb = SCORE_WEIGHT_SOFT * nfkb_soft + SCORE_WEIGHT_ONEHOT * nfkb_hard
            design_score = (ppar - nfkb + 1) / 2

        results.append({
            "score": score,
            "norm_score": norm_score,
            "ppar": ppar,
            "nfkb": nfkb,
            "design_score": design_score,
            "dna_probs": dna_probs,
            "dna_str": dna_str,
        })

    for r in results:
        print(
            f"Raw Board Score: {r['score']}    "
            f"Normalized: {r['norm_score']:.3f}    "
            f"PPARγ: {r['ppar']:.3f}    "
            f"NF-κB: {r['nfkb']:.3f}    "
            f"Design: {r['design_score']:.3f}"
        )
        print("DNA[0:2] softmax probs:\n", np.around(r['dna_probs'][:2], 3))
        print("DNA string (sampled):", r['dna_str'])
        print("-" * 40)

    print("\nAll DNA strings generated:")
    for r in results:
        print(r['dna_str'])


