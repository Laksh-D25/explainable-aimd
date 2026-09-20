"""Test whether the classes are separable by trivial signal features.

Validation F1 reached 1.000 at epoch 0 and test AUROC was exactly 1.000. Either
the task is genuinely easy at this scale, or the two classes differ in something
that has nothing to do with how the music was generated -- a codec chain, a
loudness convention, a duration convention.

The test: fit a logistic regression on a handful of cheap global descriptors
(spectral rolloff, centroid, RMS, zero-crossing rate, and so on). MERT never
sees these as features; they are properties of the *recording*. If a linear
model on five numbers separates the classes, a 94 M-parameter transformer will
find the same shortcut immediately, and the reported F1 measures the shortcut.

Per-feature AUROC localises the problem: a single feature near 1.0 names the
confound directly.

    python scripts/diagnose_shortcut.py --manifest ~/sonics/manifest_cached.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.data.audio import load_audio, peak_normalize  # noqa: E402
from aimd.eval.metrics import roc_auc  # noqa: E402

SR = 24_000


def descriptors(wav: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Cheap global properties of a recording, none of them musical content."""
    if len(wav) < 2048:
        return {}
    spectrum = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    total = spectrum.sum() + 1e-12
    cumulative = np.cumsum(spectrum)

    return {
        "rolloff_99": float(freqs[np.searchsorted(cumulative, 0.99 * total)]),
        "rolloff_85": float(freqs[np.searchsorted(cumulative, 0.85 * total)]),
        "centroid": float((freqs * spectrum).sum() / total),
        "rms": float(np.sqrt(np.mean(wav**2))),
        "peak": float(np.abs(wav).max()),
        "crest": float(np.abs(wav).max() / (np.sqrt(np.mean(wav**2)) + 1e-12)),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(wav))) > 0)),
        # Energy above 8 kHz: the band a 16 kHz-sampled encoder cannot carry.
        "hf_ratio": float(spectrum[freqs > 8000].sum() / total),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--normalize", action="store_true",
                    help="apply the dataset's peak normalisation first")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    rng = np.random.default_rng(1337)
    rows = []
    for label, group in manifest.groupby("label"):
        take = group.sample(n=min(args.n_per_class, len(group)), random_state=1337)
        for path in take["path"]:
            try:
                wav = load_audio(path, SR)[: int(args.seconds * SR)]
                if args.normalize:
                    # Match what the dataset feeds the model, rather than what
                    # sits on disk -- otherwise the diagnostic measures a
                    # difference the model never sees.
                    wav = peak_normalize(wav)
            except Exception:
                continue
            feats = descriptors(wav)
            if feats:
                rows.append({**feats, "label": int(label)})

    data = pd.DataFrame(rows)
    if data.empty or data["label"].nunique() < 2:
        raise SystemExit("need both classes with readable audio")
    print(f"{len(data)} songs: {data['label'].value_counts().to_dict()}\n")

    features = [c for c in data.columns if c != "label"]
    y = data["label"].to_numpy()

    print("per-feature AUROC (0.5 = no signal, 1.0 = perfectly separating):")
    singles = []
    for f in features:
        auc = roc_auc(y, data[f].to_numpy())
        auc = max(auc, 1 - auc)  # direction-agnostic
        singles.append((f, auc))
    for f, auc in sorted(singles, key=lambda x: -x[1]):
        flag = "  <-- SHORTCUT" if auc > 0.9 else ("  <-- strong" if auc > 0.8 else "")
        print(f"  {f:12s} {auc:.3f}{flag}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(model, data[features], y, cv=5, scoring="roc_auc")
    print(f"\nlinear model on all {len(features)} descriptors: "
          f"AUROC {scores.mean():.3f} +/- {scores.std():.3f}")

    verdict = scores.mean()
    print()
    if verdict > 0.95:
        print("VERDICT: the classes are separable from recording properties alone.")
        print("  The detector's score reflects that shortcut, not generation artefacts.")
    elif verdict > 0.80:
        print("VERDICT: a substantial shortcut exists; results are partly confounded.")
    else:
        print("VERDICT: no trivial shortcut -- high detector scores are plausibly real.")

    if args.out:
        data.to_csv(args.out, index=False)
        print(f"\nfeatures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
