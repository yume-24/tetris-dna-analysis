# analyze.py

import matplotlib.pyplot as plt
import pandas as pd

import json, math, numpy as np, onnxruntime as ort, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import entropy as bin_entropy   # binary entropy helper



# File paths
FILES = {
    "Supervised CNN": "training_log-cnn.csv",
    "Supervised MLP": "training_log-mlp2.csv",
    "Actor-Critic": "training_log_ac-2.csv"
}
def plot_validation_loss():
    logs = {label: pd.read_csv(path) for label, path in FILES.items()}

    plt.figure(figsize=(4.2, 4.2))  # or (4, 4) or (4.5, 4.5)
    # NeurIPS standard

    for label, df in logs.items():
        plt.plot(df['epoch'], df['val_quant_loss'], label=label, linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 0.6)  # Clip to highlight convergence

    plt.legend(frameon=False, fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # High-resolution PDF export for LaTeX
    plt.savefig("val_loss_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.show()




# analyze.py
"""
Heat-map:   Δ loss ( CNN – MLP )  across board-length × entropy.
Loss per sample = ( design_score  –  normalized_game_score )²
Design-score reproduces training-time PWM scoring.
"""

# ───────────────────────── Imports ─────────────────────────
import json, math
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F
import onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.stats import entropy as bin_entropy

# ───────────────────────── Constants ───────────────────────
K_TETRIS_MAX = 300
MAX_ROWS     = 30                           # fixed in ONNX export
MOTIF_DIR    = Path("motifs")
PPAR_ID, NFKB_ID = "MA0065.2", "MA0105.4"
ONNX_OUT      = "dna_logits"                # exported output name
TEMP_TEST     = 1.0                         # softmax temperature at eval

# ───────────────────── PWM loading (matches training) ─────
def _parse_meme(fp: Path):
    lines = fp.read_text().splitlines()
    idx   = next(i for i,l in enumerate(lines) if l.startswith("letter-probability"))
    W     = int(lines[idx].split("w=")[1].split()[0])
    mat   = [list(map(float, lines[idx+1+i].split())) for i in range(W)]
    return torch.tensor(mat, dtype=torch.float32)

PPAR_PWM = _parse_meme(MOTIF_DIR/f"{PPAR_ID}.meme")
NFKB_PWM = _parse_meme(MOTIF_DIR/f"{NFKB_ID}.meme")
PPAR_MAX = float(PPAR_PWM.max(1).values.sum())
NFKB_MAX = float(NFKB_PWM.max(1).values.sum())

# ───────────────────────── Helpers ─────────────────────────
def norm_game(score: float) -> float:
    return math.log1p(score) / math.log1p(K_TETRIS_MAX)

def pwm_score_np(logits: np.ndarray, pwm: torch.Tensor, pwm_max: float) -> float:
    """
    logits: (L,4) numpy
    Returns max-pool PWM score ∈[0,1]   (identical to training)
    """
    # softmax w/ temperature
    probs = F.softmax(torch.from_numpy(logits).float() / TEMP_TEST, dim=-1)  # (L,4)
    probs = probs.T.unsqueeze(0)                                             # (1,4,L)
    filt  = pwm.T.unsqueeze(0)                                               # (1,4,W)
    conv  = F.conv1d(probs, filt).squeeze()                                  # (L-W+1)
    return float(conv.max().item() / pwm_max)

def design_score_np(logits: np.ndarray) -> float:
    p = pwm_score_np(logits, PPAR_PWM, PPAR_MAX)
    n = pwm_score_np(logits, NFKB_PWM, NFKB_MAX)
    return (p - n + 1) / 2

def board_entropy(board: list[list[float]]) -> float:
    cols = len(board[0])
    ent_rows=[]
    for r in board:
        p = np.count_nonzero(np.array(r)!=0.0) / cols
        ent_rows.append(0.0 if p in (0,1) else bin_entropy([p,1-p], base=2))
    return float(np.mean(ent_rows))

def pad_or_truncate(board: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (board_arr, mask_arr) both length MAX_ROWS.
    Pads at bottom with zeros; if board is taller than MAX_ROWS, keeps last rows.
    """
    H,W = len(board), len(board[0])
    if H > MAX_ROWS:
        board = board[-MAX_ROWS:]           # keep most recent rows
        H = MAX_ROWS
    arr  = np.zeros((MAX_ROWS, W), dtype=np.float32)
    arr[:H,:] = np.asarray(board, dtype=np.float32)
    mask = np.zeros((MAX_ROWS,), dtype=np.float32)
    mask[:H] = 1.0
    return arr, mask

def run_model(sess: ort.InferenceSession,
              board: list[list[float]],
              score: float) -> tuple[float,float]:
    board_arr, mask_arr = pad_or_truncate(board)
    logits = sess.run([ONNX_OUT], {"board": board_arr[None,...],
                                   "mask":  mask_arr[None,...]})[0][0]
    return design_score_np(logits), norm_game(score)

# ───────────────────── Heat-map main routine ───────────────
# def generate_heatmap(tetris_json="tetris_test.json",
#                      onnx_mlp="model-mlp.onnx",
#                      onnx_cnn="model-cnn-sl.onnx",
#                      bins=(20,20)):
#     # Load ONNX sessions *once*
#     sess_mlp = ort.InferenceSession(onnx_mlp, providers=["CPUExecutionProvider"])
#     sess_cnn = ort.InferenceSession(onnx_cnn, providers=["CPUExecutionProvider"])

#     # Sanity: confirm output name
#     assert any(o.name == ONNX_OUT for o in sess_mlp.get_outputs()), \
#         f"{ONNX_OUT} not in MLP outputs"
#     assert any(o.name == ONNX_OUT for o in sess_cnn.get_outputs()), \
#         f"{ONNX_OUT} not in CNN outputs"

#     samples = json.load(open(tetris_json))
#     lengths, entropies, deltas = [], [], []

#     for e in samples:
#         board, score = e["game_matrix"], e["score"]

#         d_mlp, tgt = run_model(sess_mlp, board, score)
#         d_cnn, _   = run_model(sess_cnn, board, score)

#         delta      = (d_cnn - tgt)**2 - (d_mlp - tgt)**2   # + ⇒ CNN worse
#         lengths.append(min(len(board), MAX_ROWS))          # after trunc
#         entropies.append(board_entropy(board))
#         deltas.append(delta)

#     # 2-D bin mean
#     arr_len = np.asarray(lengths)
#     arr_ent = np.asarray(entropies)
#     arr_del = np.asarray(deltas)

#     xbins = np.linspace(arr_len.min(), arr_len.max(), bins[0]+1)
#     ybins = np.linspace(arr_ent.min(), arr_ent.max(), bins[1]+1)
#     heat  = np.full((bins[1], bins[0]), np.nan)

#     for i in range(bins[0]):
#         for j in range(bins[1]):
#             m = (arr_len>=xbins[i])&(arr_len<xbins[i+1]) & \
#                 (arr_ent>=ybins[j])&(arr_ent<ybins[j+1])
#             if m.any(): heat[j,i] = arr_del[m].mean()

#     # ────────────────  Plot  ────────────────
#     plt.figure(figsize=(5,4))
#     vmax = np.nanmax(np.abs(heat))
#     plt.imshow(heat, origin="lower", aspect="auto",
#                extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
#                cmap="RdBu_r", vmin=-vmax, vmax=vmax)
#     plt.colorbar(label="CNN loss – MLP loss")
#     plt.xlabel("Board length (rows)", fontsize=12)
#     plt.ylabel("Avg. row entropy",  fontsize=12)
#     plt.xticks(fontsize=10); plt.yticks(fontsize=10)
#     plt.tight_layout()
#     plt.savefig("delta_loss_heatmap.pdf", dpi=300, bbox_inches="tight")
#     plt.show()
#     print("✔ Saved delta_loss_heatmap.pdf")

# ##Decent, using row-wise entorpy
# def generate_heatmap(tetris_json="tetris_test.json",
#                      onnx_mlp="model-mlp.onnx",
#                      onnx_cnn="model-cnn-sl.onnx",
#                      bins=(20, 20)):
#     # Load ONNX sessions
#     sess_mlp = ort.InferenceSession(onnx_mlp, providers=["CPUExecutionProvider"])
#     sess_cnn = ort.InferenceSession(onnx_cnn, providers=["CPUExecutionProvider"])

#     # Read data
#     samples = json.load(open(tetris_json))
#     scores, entropies, deltas = [], [], []

#     for e in samples:
#         board, score = e["game_matrix"], e["score"]
#         d_mlp, tgt = run_model(sess_mlp, board, score)
#         d_cnn, _   = run_model(sess_cnn, board, score)

#         # Δ MSE loss
#         delta = (d_cnn - tgt)**2 - (d_mlp - tgt)**2

#         scores.append(min(score, 100))           # cap at 100
#         entropies.append(board_entropy(board))
#         deltas.append(delta)

#     arr_s = np.array(scores)
#     arr_e = np.array(entropies)
#     arr_d = np.array(deltas)

#     # Bin edges
#     x_min, x_max = 0, 100
#     y_min, y_max = arr_e.min(), arr_e.max()
#     xbins = np.linspace(x_min, x_max, bins[0] + 1)
#     ybins = np.linspace(y_min, y_max, bins[1] + 1)

#     # Compute mean delta in each bin
#     heat = np.full((bins[1], bins[0]), np.nan)
#     for i in range(bins[0]):
#         for j in range(bins[1]):
#             mask = (
#                 (arr_s >= xbins[i]) & (arr_s < xbins[i+1]) &
#                 (arr_e >= ybins[j]) & (arr_e < ybins[j+1])
#             )
#             if mask.any():
#                 heat[j, i] = arr_d[mask].mean()

#     # Plot
#     plt.figure(figsize=(5, 4))
#     vmax = np.nanmax(np.abs(heat))
#     plt.imshow(
#         heat,
#         origin='lower',
#         aspect='auto',
#         extent=[x_min, x_max, y_min, y_max],
#         cmap='RdBu_r',
#         vmin=-vmax, vmax=vmax
#     )
#     plt.colorbar(label="CNN loss − MLP loss")
#     plt.xlabel("Score",            fontsize=12)
#     plt.ylabel("Avg. row entropy", fontsize=12)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.xlim(x_min, x_max)
#     plt.tight_layout()
#     plt.savefig("delta_loss_heatmap.pdf", dpi=300, bbox_inches="tight")
#     plt.show()


#######
#######   NOTE: Using MSE here is kinda wack, it makes way more sense to use a batch wise corr... hm...
#######

#This is MSE / ABS:
##I liked (20, 20) bins and (0, 100) score, 0.02 contrast...
# def generate_heatmap(
#                     tetris_json="tetris_test.json",    
#                     # tetris_json="tetris_data-15k.json",
#                      onnx_mlp="model-mlp2.onnx",
#                     #  onnx_mlp="actor_corr-2.onnx",
#                      onnx_cnn="model-cnn-sl.onnx",
#                     #  bins=(32, 32), # This is nice
#                      bins=(20,20),
#                      frag_range=(5, 7),
#                      score_range=(0, 120),
#                     #  contrast=0.01, # This is nice
#                      contrast=0.02,
#                      save_path="delta_loss_heatmap_fragments_fixed.pdf"):
#     """
#     Heatmap of Δ MSE loss (CNN – MLP) over Score × Avg. row-fragment count.
#     Only considers bottom-most filled rows (stopping at first zero row).
    
#     Args:
#         tetris_json: path to Tetris data JSON.
#         onnx_mlp: path to MLP ONNX model.
#         onnx_cnn: path to CNN ONNX model.
#         bins: (x, y) number of bins.
#         frag_range: (min, max) y-axis range.
#         score_range: (min, max) x-axis range.
#         contrast: +/- scale for color exaggeration.
#         save_path: output PDF file name.
#     """

#     def row_fragments(row: list[float]) -> int:
#         segs = 0
#         last = 0.0
#         for v in row:
#             if v != 0.0 and v != last:
#                 segs += 1
#                 last = v
#             elif v == 0.0:
#                 last = 0.0
#         return segs

#     samples  = json.load(open(tetris_json))
#     # samples = samples[:2000]
#     sess_mlp = ort.InferenceSession(onnx_mlp, providers=["CPUExecutionProvider"])
#     sess_cnn = ort.InferenceSession(onnx_cnn, providers=["CPUExecutionProvider"])

#     scores, frags, deltas = [], [], []
#     for e in samples:
#         board, score = e["game_matrix"], e["score"]

#         filled_rows = []
#         for row in reversed(board):
#             if any(cell == 0.0 for cell in row):
#                 break
#             filled_rows.append(row)
#         filled_rows.reverse()

#         avg_frag = float(np.mean([row_fragments(r) for r in filled_rows])) if filled_rows else 0.0

#         d_mlp, tgt = run_model(sess_mlp, board, score)
#         d_cnn, _   = run_model(sess_cnn, board, score)
#         # delta      = (d_cnn - tgt)**2 - (d_mlp - tgt)**2 #Mse
#         delta = abs(d_cnn - tgt) - abs(d_mlp - tgt)


#         scores.append(min(score, score_range[1]))
#         frags.append(avg_frag)
#         deltas.append(delta)

#     arr_s = np.array(scores)
#     arr_f = np.array(frags)
#     arr_d = np.array(deltas)

#     xedges = np.linspace(score_range[0], score_range[1], bins[0] + 1)
#     yedges = np.linspace(frag_range[0], frag_range[1], bins[1] + 1)

#     heat = np.full((bins[1], bins[0]), np.nan)
#     for i in range(bins[0]):
#         for j in range(bins[1]):
#             mask = ((arr_s >= xedges[i]) & (arr_s < xedges[i+1]) &
#                     (arr_f >= yedges[j]) & (arr_f < yedges[j+1]))
#             if mask.any():
#                 heat[j, i] = arr_d[mask].mean()

#     plt.figure(figsize=(5, 4))
#     plt.imshow(
#         heat,
#         origin='lower',
#         aspect='auto',
#         extent=[score_range[0], score_range[1], frag_range[0], frag_range[1]],
#         cmap='coolwarm',
#         vmin=-contrast, vmax=contrast
#     )

#     plt.colorbar(label="CNN loss − MLP loss")
#     plt.xlabel("Game Score", fontsize=12)
#     plt.ylabel("Row Tile Fragmentation", fontsize=12)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.xlim(score_range)
#     plt.ylim(frag_range)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()



#Z scores
def generate_heatmap(
    tetris_json="tetris_test.json",
    onnx_mlp="model-mlp2.onnx",
    onnx_cnn="model-cnn-sl.onnx",
    bins=(28, 28),
    frag_range=(5, 7),
    score_range=(0, 120),
    contrast=0.3,
    save_path="delta_loss_heatmap_fragments_fixed.pdf"
):
    samples = json.load(open(tetris_json))
    sess_mlp = ort.InferenceSession(onnx_mlp, providers=["CPUExecutionProvider"])
    sess_cnn = ort.InferenceSession(onnx_cnn, providers=["CPUExecutionProvider"])

    scores, frags, mlp_preds, cnn_preds, targets = [], [], [], [], []

    def row_fragments(row: list[float]) -> int:
        segs = 0
        last = 0.0
        for v in row:
            if v != 0.0 and v != last:
                segs += 1
                last = v
            elif v == 0.0:
                last = 0.0
        return segs

    for e in samples:
        board, score = e["game_matrix"], e["score"]
        filled_rows = []
        for row in reversed(board):
            if any(cell == 0.0 for cell in row):
                break
            filled_rows.append(row)
        filled_rows.reverse()

        avg_frag = np.mean([row_fragments(r) for r in filled_rows]) if filled_rows else 0.0

        d_mlp, tgt = run_model(sess_mlp, board, score)
        d_cnn, _   = run_model(sess_cnn, board, score)

        scores.append(min(score, score_range[1]))
        frags.append(avg_frag)
        mlp_preds.append(d_mlp)
        cnn_preds.append(d_cnn)
        targets.append(tgt)

    # Z-score normalize predictions and targets
    mlp_preds = np.array(mlp_preds)
    cnn_preds = np.array(cnn_preds)
    targets   = np.array(targets)

    mlp_preds = (mlp_preds - mlp_preds.mean()) / mlp_preds.std()
    cnn_preds = (cnn_preds - cnn_preds.mean()) / cnn_preds.std()
    targets   = (targets   - targets.mean())   / targets.std()

    deltas = np.abs(cnn_preds - targets) - np.abs(mlp_preds - targets)

    arr_s = np.array(scores)
    arr_f = np.array(frags)
    arr_d = np.array(deltas)

    xedges = np.linspace(score_range[0], score_range[1], bins[0] + 1)
    yedges = np.linspace(frag_range[0], frag_range[1], bins[1] + 1)
    heat = np.full((bins[1], bins[0]), np.nan)

    for i in range(bins[0]):
        for j in range(bins[1]):
            mask = ((arr_s >= xedges[i]) & (arr_s < xedges[i+1]) &
                    (arr_f >= yedges[j]) & (arr_f < yedges[j+1]))
            if mask.any():
                heat[j, i] = arr_d[mask].mean()

    plt.figure(figsize=(5, 4))
    plt.imshow(
        heat, origin='lower', aspect='auto',
        extent=[score_range[0], score_range[1], frag_range[0], frag_range[1]],
        cmap='coolwarm', vmin=-contrast, vmax=contrast
    )
    plt.colorbar(label="CNN abs error − MLP abs error (z-scored)")
    plt.xlabel("Game Score", fontsize=12)
    plt.ylabel("Row Tile Fragmentation", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(score_range)
    plt.ylim(frag_range)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
