from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save to a file, no pop-up window
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "comparison"
OUT_DIR.mkdir(exist_ok=True)

# Load the two final grids (they are already saved as images)
img_v1 = plt.imread(BASE_DIR / "outputs" / "epoch_030.png")
img_v2 = plt.imread(BASE_DIR / "outputs_v2" / "epoch_030.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 8.5))

axes[0].imshow(img_v1)
axes[0].set_title("Baseline (v1): repeated shoes", fontsize=13)
axes[0].axis("off")

axes[1].imshow(img_v2)
axes[1].set_title("Stricter discriminator (v2): more variety", fontsize=13)
axes[1].axis("off")

fig.tight_layout()
fig.savefig(OUT_DIR / "grid_comparison.png", dpi=130)
print("Saved:", OUT_DIR / "grid_comparison.png")