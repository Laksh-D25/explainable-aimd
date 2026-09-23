r"""Train the pooling ablations, one model at a time.

Split out of run_experiment.py because running them inside it is what broke:
the trained detector is still resident on the GPU when the first variant is
built, two backbones do not fit under the display-safe VRAM cap, and the whole
stage died with an OutOfMemoryError after the useful results had already been
written. Here only one model exists at a time.

Rows are appended to ablations.csv as each variant finishes, and a variant
already present is skipped, so an interrupted run resumes instead of restarting.

    python scripts/run_ablations.py --manifest ~/sonics/manifest_big_cached.csv \
        --out artifacts/abl --epochs 10 --batch-size 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.eval.protocols import evaluate_in_distribution  # noqa: E402
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    resolve_device,
    train_model,
)

# The name is the claim being tested; the overrides are how the head is built.
# "baseline" repeats the reported configuration so the comparison is against a
# model trained in this same process, not against a number from another run.
VARIANTS = {
    "baseline": {},
    "pool_mean": {"frame_pool": "mean"},
    "layers_last": {"layer_mode": "last"},
    "song_mean": {"song_pool": "mean"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--clips-per-song", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = resolve_device("auto")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    csv_path = args.out / "ablations.csv"

    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    already = set(done["ablation"]) if len(done) else set()
    if already:
        print(f"resuming; already done: {sorted(already)}", flush=True)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clip_seconds=args.clip_seconds,
        clips_per_song=args.clips_per_song,
        augment=True,
        num_workers=2,
        amp=device.startswith("cuda"),
    )
    spec = config.spec()

    for name, overrides in VARIANTS.items():
        if name in already:
            continue
        # Same seed per variant: the comparison is between architectures, so the
        # initialisation and the batch order must not also differ.
        torch.manual_seed(config.seed)
        model = build_model(device, **overrides)
        result = train_model(model, manifest, config, device=device, log=lambda _m: None)
        scaler, threshold, _ = calibrate_and_threshold(
            model, manifest, spec, target_fpr=0.05, device=device, amp=config.amp)
        r = evaluate_in_distribution(model, manifest, spec, scaler, threshold,
                                     device=device, amp=config.amp)
        rows.append({"ablation": name, "best_val_f1": result["best"]["f1"], **r.metrics})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  {name:12s} F1 {r.metrics['f1']:.4f}  AUROC {r.metrics['auroc']:.4f}  "
              f"FPR {r.metrics['fpr']:.3f}  (val F1 {result['best']['f1']:.4f})", flush=True)

        del model, result, scaler
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
