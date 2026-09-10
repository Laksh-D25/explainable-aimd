"""Figures and tables for the write-up.

Four figures carry the project's four claims, so each is built to be read
rather than admired:

* **Reliability diagram** -- the calibration claim. Perfect calibration is the
  diagonal; the gap to it is the result.
* **Robustness** -- F1 drop per held-out condition, sorted, one bar each.
* **Layer profile** -- what the model *weights* (attention, level 1) against
  what actually *moved the decision* (Shapley, level 3). Their disagreement is
  a faithfulness finding, so the figure is built to show it rather than hide it.
* **Faithfulness curves** -- the explanation against its random baseline. The
  gap between the two lines is the claim; the explanation curve alone is not.

Conventions: one measure per axis (never a second y-scale), thin marks,
recessive grid, a legend whenever two series are present, and text in ink
colours rather than series colours so identity is never carried by colour
alone. Every figure writes its underlying numbers to CSV beside it -- that
table is the accessible view, and it is also what goes in an appendix.

Static output for print, so there is no hover layer to ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Validated categorical slots (light surface). Two series is the common case;
# the third is reserved. See the palette validation in the project notes.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#dcdbd6"


def _plt():
    """Import matplotlib with a non-interactive backend.

    Deferred so importing this module never requires a display, and so the
    package remains importable where matplotlib is absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    """Recessive axes: the data should be the only assertive thing present."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK_MUTED)
        label.set_fontsize(10)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)


def _save(fig, out: Path, data: pd.DataFrame | None = None) -> Path:
    """Write PDF (vector, for print) and PNG, plus the numbers as CSV."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    if data is not None:
        data.to_csv(out.with_suffix(".csv"), index=False)
    _plt().close(fig)
    return out.with_suffix(".pdf")


def reliability_diagram(
    labels: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray | None,
    out: Path,
    n_bins: int = 10,
) -> Path:
    """Confidence against observed accuracy, with the perfect-calibration diagonal."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))

    rows = []
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    series = [("Uncalibrated", probs_before, SERIES[0])]
    if probs_after is not None:
        series.append(("Temperature-scaled", probs_after, SERIES[1]))

    for name, probs, colour in series:
        confidence = np.where(probs >= 0.5, probs, 1 - probs)
        correct = (probs >= 0.5).astype(int) == np.asarray(labels).astype(int)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (confidence > lo) & (confidence <= hi)
            if not in_bin.any():
                continue
            xs.append(confidence[in_bin].mean())
            ys.append(correct[in_bin].mean())
            rows.append({"series": name, "confidence": xs[-1], "accuracy": ys[-1],
                         "n": int(in_bin.sum())})
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=colour, label=name)

    ax.plot([0.5, 1], [0.5, 1], linestyle=(0, (4, 3)), linewidth=1.2, color=INK_MUTED,
            label="Perfect calibration", zorder=1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.4, 1.02)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")
    return _save(fig, out, pd.DataFrame(rows))


def robustness_chart(robustness: pd.DataFrame, out: Path) -> Path:
    """F1 drop per held-out condition, sorted, one bar each.

    A single hue: this is one magnitude across categories, not several series,
    so colouring the bars differently would imply an identity that is not there.
    """
    plt = _plt()
    data = robustness.drop(index="clean", errors="ignore").sort_values("f1_drop")
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(data) + 1.6))

    # Thin bars with a surface-coloured edge, so adjacent fills stay separated.
    ax.barh(data.index, data["f1_drop"], color=SERIES[0], height=0.5,
            edgecolor=SURFACE, linewidth=1.2)
    for name, value in zip(data.index, data["f1_drop"]):
        # Direct labels: the exact drop matters more than reading it off an axis.
        ax.text(value + 0.006, name, f"{value:+.3f}", va="center", fontsize=9, color=INK_MUTED)

    ax.set_xlabel("F1 drop from clean audio")
    ax.set_title("Robustness to held-out degradations")
    ax.set_xlim(min(0, data["f1_drop"].min() * 1.15), max(data["f1_drop"].max() * 1.30, 0.02))
    _style(ax)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, data.reset_index())


def layer_profile(attention: np.ndarray, shapley: np.ndarray, out: Path) -> Path:
    """Level 1 against level 3 across MERT's hidden states.

    Both are normalised to sum to 1 in absolute value so they share one axis --
    they measure different quantities, and plotting them on two y-scales would
    manufacture agreement or disagreement from the scaling alone.
    """
    plt = _plt()
    attention = np.asarray(attention).ravel()
    shapley = np.asarray(shapley).ravel()
    norm_shap = np.abs(shapley) / (np.abs(shapley).sum() or 1.0)
    layers = np.arange(len(attention))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # Paired bars need a visible gap between the two fills, not just between pairs.
    width = 0.36
    ax.bar(layers - width / 2 - 0.02, attention, width, color=SERIES[0],
           edgecolor=SURFACE, linewidth=1.0, label="Attention weight (L1)")
    ax.bar(layers + width / 2 + 0.02, norm_shap, width, color=SERIES[1],
           edgecolor=SURFACE, linewidth=1.0, label="|Shapley| share (L3)")

    ax.set_xlabel("MERT hidden state")
    ax.set_ylabel("Share of total")
    ax.set_title("What the model weights vs what moved the decision")
    ax.set_xticks(layers)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    return _save(fig, out, pd.DataFrame({
        "layer": layers, "attention_l1": attention,
        "shapley_l3": shapley, "shapley_share": norm_shap,
    }))


def faithfulness_curves(curve, out: Path, label: str = "Deletion") -> Path:
    """Explanation against its random baseline.

    The gap between the lines is the result. A steep explanation curve alone
    means nothing -- masking any audio degrades a detector somewhat.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    ax.plot(curve.fractions, curve.scores, marker="o", markersize=5, linewidth=2,
            color=SERIES[0], label="Guided by explanation")
    ax.plot(curve.fractions, curve.random_scores, marker="s", markersize=5, linewidth=2,
            color=SERIES[1], linestyle=(0, (5, 3)), label="Random regions")

    ax.set_xlabel("Fraction of audio masked")
    ax.set_ylabel("Binary logit")
    ax.set_title(f"{label} — faithfulness gap {curve.gap:+.3f}")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    return _save(fig, out, pd.DataFrame({
        "fraction": curve.fractions,
        "explanation": curve.scores,
        "random": curve.random_scores,
    }))


def results_markdown(protocols: pd.DataFrame, baseline_f1: float = 0.629) -> str:
    """The headline table, formatted for the write-up.

    The base paper's cross-generator F1 travels with the table so the
    comparison is stated rather than left to the reader.
    """
    lines = ["| Protocol | n | F1 | AUROC | EER | FPR | ECE |", "|---|---|---|---|---|---|---|"]
    for name, row in protocols.iterrows():
        lines.append(
            f"| {name} | {int(row['n_songs'])} | {row['f1']:.3f} | {row['auroc']:.3f} | "
            f"{row['eer']:.3f} | {row['fpr']:.3f} | {row['ece']:.4f} |"
        )
    if "cross_generator" in protocols.index:
        achieved = protocols.loc["cross_generator", "f1"]
        lines += [
            "",
            f"Base paper cross-generator F1: **{baseline_f1:.3f}**. "
            f"This work: **{achieved:.3f}** ({achieved - baseline_f1:+.3f}).",
        ]
    return "\n".join(lines)
