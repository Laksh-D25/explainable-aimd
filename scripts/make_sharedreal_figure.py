r"""Figure: the collapse happens with the real class held fixed.

The paper's earlier transfer figure paired AUROC with false-positive rate to
show that the detector had stopped recognising real music. This one makes the
opposite point, and needs a different pairing: FPR is *constant* here, because
the same recordings supply the real class throughout, so putting it beside
AUROC is what proves the loss is not about real music at all.

    python scripts/make_sharedreal_figure.py --out artifacts/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.eval.report import INK, INK_MUTED, SERIES, SURFACE, _plt, _save, _style  # noqa: E402

LABELS = {
    "audioldm2": "AudioLDM2",
    "stable_audio_open": "StableAudioOpen",
    "musicgen": "MusicGen",
    "musicldm": "MusicLDM",
    "mustango": "Mustango",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("artifacts/sharedreal_ctl/results.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures"))
    args = ap.parse_args()

    d = json.loads(args.results.read_text())
    rows = [{"setting": "Suno/Udio\n(seen)", "auroc": d["in_distribution"]["auroc"],
             "seen": True}]
    rows += [{"setting": LABELS.get(k, k), "auroc": v["auroc"], "seen": False}
             for k, v in sorted(d["per_generator"].items(), key=lambda kv: -kv[1]["auroc"])]
    data = pd.DataFrame(rows)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    # One hue for the generators trained on, another for the unseen ones: the
    # split between them is the whole content of the figure.
    colours = [SERIES[0] if s else SERIES[1] for s in data["seen"]]
    ax.bar(data["setting"], data["auroc"], color=colours, width=0.6,
           edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(len(data) - 0.4, 0.515, "chance", fontsize=8, color=INK_MUTED, ha="right")

    for i, v in enumerate(data["auroc"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=INK_MUTED)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC")
    ax.set_title("Same real recordings throughout; only the generator changes")
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8.5)

    fpr = d["in_distribution"]["fpr"]
    fig.text(0.5, -0.06,
             f"False-positive rate is {fpr:.3f} for every bar: the real class is the "
             f"same {d['in_distribution']['n'] // 2} MusicCaps recordings in all cases.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.suptitle("", y=1.0)
    _save(fig, args.out / "fig_sharedreal", data)
    print(f"wrote {args.out / 'fig_sharedreal.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
