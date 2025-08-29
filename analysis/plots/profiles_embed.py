# save as analysis/plots/profiles_embed.py (or run in a Python shell)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

npz = np.load("outputs/analysis/pwm_supervised/profiles_wide.npz", allow_pickle=True)
X_p = npz["ppar_best"]; X_n = npz["nfkb_best"]
ds = npz["dataset"];    sid = npz["seq_id"]

# impute NaN (padding) with column means
imp = SimpleImputer(strategy="mean")
Xp = imp.fit_transform(X_p); Xn = imp.fit_transform(X_n)
X = np.hstack([Xp, Xn])

# scale -> PCA
Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
Z = PCA(n_components=2).fit_transform(Xz)

# plot PCA
fig, ax = plt.subplots(figsize=(7,6))
for name, c in [("AI","#1f77b4"), ("Human","#ff7f0e")]:
    m = (ds == name)
    ax.scatter(Z[m,0], Z[m,1], s=30, alpha=0.8, label=f"{name} (n={m.sum()})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA of per-base PWM profiles")
ax.legend(frameon=False); plt.tight_layout()
plt.savefig("outputs/analysis/pwm_supervised/pca_profiles.png", dpi=220)

# t-SNE (optional)
Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(Xz)
fig, ax = plt.subplots(figsize=(7,6))
for name, c in [("AI","#1f77b4"), ("Human","#ff7f0e")]:
    m = (ds == name)
    ax.scatter(Zt[m,0], Zt[m,1], s=30, alpha=0.8, label=f"{name} (n={m.sum()})")
ax.set_title("t-SNE of per-base PWM profiles"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("outputs/analysis/pwm_supervised/tsne_profiles.png", dpi=220)
