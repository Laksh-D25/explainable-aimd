"""Figure: what transfers, and what does not.

The two cross-generator experiments disagree sharply, and the disagreement is
the finding. Holding the real class fixed and swapping the generator (Suno to
Udio) barely dents the detector. Swapping the *real* class's corpus as well
(SONICS to MusicCaps, as FakeMusicCaps requires) collapses it to near chance
with every real clip called fake.

Plotting AUROC beside false-positive rate is what makes that legible: AUROC
alone shows the collapse but not its mechanism, and FPR alone looks like a
threshold problem. Together they say the detector stopped recognising real
music, not that it failed on new generators.

    python scripts/make_transfer_figure.py --out artifacts/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.eval.report import INK, INK_MUTED, SERIES, SURFACE, _plt, _save, _style  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gensplit", type=Path, default=Path("artifacts/gensplit/generator_split.csv"))
    ap.add_argument("--fmc", type=Path, default=Path("artifacts/run2/cross_generator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures"))
    args = ap.parse_args()

    rows = []
    if args.gensplit.exists():
        for _, r in pd.read_csv(args.gensplit).iterrows():
            rows.append({
                "setting": r["direction"].replace("->", " → ").replace("suno", "Suno").replace("udio", "Udio"),
                "real_class": "same corpus",
                "auroc": r["cross_auroc"],
                "fpr": r["cross_fpr"],
            })
    if args.fmc.exists():
        d = json.loads(args.fmc.read_text())
        rows.append({
            "setting": "SONICS → FakeMusicCaps",
            "real_class": "different corpus",
            "auroc": d["metrics"]["auroc"],
            "fpr": d["metrics"]["fpr"],
        })
    if not rows:
        raise SystemExit("no results found")

    data = pd.DataFrame(rows)
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    y = range(len(data))
    # One hue per axis: each panel shows a single measure across settings, not
    # several series, so colouring by row would imply an identity that is absent.
    ax1.barh(list(y), data["auroc"], color=SERIES[0], height=0.5,
             edgecolor=SURFACE, linewidth=1.2)
    ax1.axvline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    # Place the marker low, not at the top edge, where it collided with the title.
    ax1.text(0.5, -0.42, " chance", fontsize=8, color=INK_MUTED, va="center")
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("AUROC on the unseen generator")
    ax1.set_title("Discrimination")

    ax2.barh(list(y), data["fpr"], color=SERIES[1], height=0.5,
             edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("False-positive rate (real music called fake)")
    ax2.set_title("Recognition of real music")

    for ax, col, fmt in ((ax1, "auroc", "{:.3f}"), (ax2, "fpr", "{:.3f}")):
        ax.set_yticks(list(y))
        ax.set_yticklabels(data["setting"])
        _style(ax)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(data[col]):
            ax.text(v + 0.02, i, fmt.format(v), va="center", fontsize=9, color=INK_MUTED)

    fig.suptitle("Transfer survives a new generator, not a new corpus of real music",
                 fontsize=11, color=INK, y=1.02)
    return 0 if _save(fig, args.out / "fig_transfer", data) else 0


if __name__ == "__main__":
    raise SystemExit(main())
