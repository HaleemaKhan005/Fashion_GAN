from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # save to a file, no pop-up window
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "comparison"
OUT_DIR.mkdir(exist_ok=True)

# Load the saved losses (one number per epoch)
g_v1 = np.load(BASE_DIR / "outputs" / "g_loss.npy")
d_v1 = np.load(BASE_DIR / "outputs" / "d_loss.npy")
g_v2 = np.load(BASE_DIR / "outputs_v2" / "g_loss.npy")
d_v2 = np.load(BASE_DIR / "outputs_v2" / "d_loss.npy")

print("Epochs found -> baseline:", len(g_v1), "| v2:", len(g_v2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: discriminator (the teacher)
axes[0].plot(range(1, len(d_v1) + 1), d_v1, label="Baseline (v1)")
axes[0].plot(range(1, len(d_v2) + 1), d_v2, label="Stricter discriminator (v2)")
axes[0].axhline(1.386, color="gray", linestyle="--", label="Balance point (1.386)")
axes[0].set_title("Discriminator loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Right: generator (the designer)
axes[1].plot(range(1, len(g_v1) + 1), g_v1, label="Baseline (v1)")
axes[1].plot(range(1, len(g_v2) + 1), g_v2, label="Stricter discriminator (v2)")
axes[1].axhline(0.693, color="gray", linestyle="--", label="Balance point (0.693)")
axes[1].set_title("Generator loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "loss_comparison.png", dpi=120)
print("Saved:", OUT_DIR / "loss_comparison.png")