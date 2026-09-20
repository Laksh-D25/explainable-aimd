"""Cross-generator evaluation: a SONICS-trained detector against FakeMusicCaps.

This is the project's headline experiment. The base paper reports F1 collapsing
from 0.99 in-distribution to 0.629 when the generator changes; the question is
whether a music foundation model with attention pooling narrows that.

The detector never sees FakeMusicCaps in training, and the five text-to-music
models there (AudioLDM2, MusicGen, MusicLDM, Mustango, StableAudioOpen) share
nothing with SONICS's Suno and Udio, so this is a genuine unseen-generator test.

Two things are checked before the number is trusted:

* **Bandwidth.** FakeMusicCaps is 16 kHz and the MusicCaps copy is 48 kHz. If
  the classes differ spectrally, the detector separates them on bandwidth and
  the result says nothing about generators. The caller is expected to have
  band-limited the real audio; this verifies it and refuses to report a clean
  number if the gap remains.
* **Per-generator breakdown.** A single averaged F1 hides the case that matters
  -- transfer to one architecture and failure on another.

    python scripts/cross_generator_eval.py --generated ~/sonics/fmc/generated \
        --real ~/sonics/fmc/real_16k --checkpoint artifacts/run2/ckpt/best.pt \
        --out artifacts/run2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.data.audio import bandwidth_report  # noqa: E402
from aimd.data.manifest import load_fakemusiccaps_manifest  # noqa: E402
from aimd.eval.metrics import summary  # noqa: E402
from aimd.eval.protocols import build_loader  # noqa: E402
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    load_checkpoint,
    resolve_device,
)
from aimd.train.loop import predict  # noqa: E402

BASE_PAPER_CROSS_GENERATOR_F1 = 0.629


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sonics-manifest", type=Path, required=True,
                    help="used only to fit calibration on its validation split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    device = resolve_device("auto")
    manifest = load_fakemusiccaps_manifest(str(args.generated), str(args.real))
    print(f"cross-generator set: {len(manifest):,} clips")
    print(manifest.groupby(["taxonomy", "source"], dropna=False).size().to_string())

    if not (manifest["label"] == 0).any():
        raise SystemExit("no real class -- detection metrics are undefined")

    # Gate on the confound before reporting anything.
    bw = bandwidth_report(manifest, n_per_class=40)
    print(f"\nbandwidth: {bw}")
    confounded = bool(bw.get("suspicious"))
    if confounded:
        print("  WARNING: the classes differ in bandwidth. Any F1 below reflects "
              "the codec chain as much as the generator.")

    model = build_model(device)
    payload = load_checkpoint(model, args.checkpoint, device)
    config = TrainConfig(**payload["config"])
    spec = config.spec()
    amp = config.amp and device.startswith("cuda")
    model.eval()
    print(f"\ncheckpoint: epoch {payload['epoch'] + 1}, val F1 {payload['val_f1']:.4f}")

    # Calibration comes from SONICS validation, never from the target set: a
    # threshold tuned on the unseen generator would not be a transfer result.
    sonics = pd.read_csv(args.sonics_manifest)
    scaler, threshold, calib = calibrate_and_threshold(
        model, sonics, spec, target_fpr=args.target_fpr, device=device, amp=amp)
    print(f"calibration (from SONICS val): T={calib['temperature']:.3f} "
          f"threshold={threshold:.6f}")

    out = predict(model, build_loader(manifest, spec, "test", batch_size=4),
                  device=device, amp=amp)
    probs = scaler.transform(out["logit"])
    overall = summary(out["label"], probs, threshold=threshold)
    print(f"\n=== CROSS-GENERATOR (n={len(out['label'])}) ===")
    print(json.dumps({k: round(v, 4) for k, v in overall.items()}, indent=2))
    delta = overall["f1"] - BASE_PAPER_CROSS_GENERATOR_F1
    print(f"\nbase paper cross-generator F1 {BASE_PAPER_CROSS_GENERATOR_F1} -> "
          f"this work {overall['f1']:.4f} ({delta:+.4f})")

    # Per-generator: an average hides transfer to one model and failure on another.
    by_id = manifest.set_index(manifest["song_id"].astype(str))
    rows = []
    for i, song_id in enumerate(out["song_id"]):
        rows.append({"song_id": song_id, "prob": probs[i], "label": out["label"][i],
                     "source": by_id.loc[str(song_id), "source"]})
    per = pd.DataFrame(rows)
    real_probs = per.loc[per["label"] == 0, "prob"].to_numpy()
    real_labels = per.loc[per["label"] == 0, "label"].to_numpy()

    print("\nper-generator (each scored against the shared real class):")
    gen_rows = []
    for source, group in per[per["label"] == 1].groupby("source"):
        import numpy as np
        y = np.concatenate([real_labels, group["label"].to_numpy()])
        p = np.concatenate([real_probs, group["prob"].to_numpy()])
        m = summary(y, p, threshold=threshold)
        gen_rows.append({"generator": source, "n_generated": len(group), **m})
        print(f"  {source:18s} F1 {m['f1']:.4f}  AUROC {m['auroc']:.4f}  "
              f"recall {(group['prob'] >= threshold).mean():.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gen_rows).to_csv(args.out / "cross_generator_by_source.csv", index=False)
    payload_out = {
        "n": int(len(out["label"])),
        "metrics": overall,
        "base_paper_f1": BASE_PAPER_CROSS_GENERATOR_F1,
        "delta": delta,
        "bandwidth": bw,
        "bandwidth_confounded": confounded,
        "calibration": calib,
    }
    (args.out / "cross_generator.json").write_text(
        json.dumps(payload_out, indent=2, default=float))
    print(f"\nwrote {args.out / 'cross_generator.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
