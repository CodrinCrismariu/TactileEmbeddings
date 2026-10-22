"""Export transfer comparisons; error bars describe seed variation, not CIs."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = Path(sys.argv[1]); rows = json.loads(path.read_text())["seed_aggregates"]
    for epoch in sorted({r["encoder_epoch"] for r in rows}):
        current = [r for r in rows if r["encoder_epoch"] == epoch]
        heads = sorted({r["head"] for r in current})
        fractions = sorted({r["train_fraction"] for r in current})
        figure, axes = plt.subplots(len(heads), len(fractions), figsize=(14, 4 * len(heads)), squeeze=False)
        for i, head in enumerate(heads):
            for j, fraction in enumerate(fractions):
                values = sorted([r for r in current if r["head"] == head and r["train_fraction"] == fraction],
                                key=lambda r: (r["placement"], r["variant"]))
                labels = [r["placement"].replace("tactile_first", "tactile") + ": " + r["variant"] for r in values]
                axis = axes[i, j]
                axis.barh(range(len(values)), [r["mae_mean"] for r in values],
                          xerr=[r["mae_seed_std"] or 0 for r in values], capsize=2)
                axis.set_yticks(range(len(values)), labels); axis.invert_yaxis()
                axis.set_title(f"{head} · {int(fraction * 100)}% training episodes")
                axis.set_xlabel("Normalized torque MAE; bars ± seed standard deviation")
                axis.grid(axis="x", alpha=.2)
        figure.suptitle(f"MotionTrans transfer · encoder epoch {epoch}")
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(path.parent / f"transfer_epoch_{epoch:03d}.{extension}", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
