"""Evaluate the Suno/Udio transfer checkpoints without retraining.

Both directions trained successfully, but the run was piping through `grep` when
the machine rebooted, so the buffered results were lost while the checkpoints
survived. Retraining would take longer and risk the same loss, so this rebuilds
the identical split (same seed and helper as the training run) and evaluates.

Results are written per direction, as each finishes, rather than once at the
end -- that is precisely what was lost last time.

    python scripts/gensplit_eval_only.py --manifest ~/sonics/manifest_big_cached.csv \
        --out artifacts/gensplit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from generator_split_eval import BASE_PAPER_CROSS, BASE_PAPER_IN_DIST, build_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = resolve_device("auto")
    manifest = pd.read_csv(args.manifest)
    results = []

    for train_gen, test_gen in [("suno", "udio"), ("udio", "suno")]:
        ckpt = args.out / f"ckpt_{train_gen}" / "best.pt"
        if not ckpt.exists():
            print(f"no checkpoint for {train_gen}; skipping", flush=True)
            continue

        train_part, test_part = build_split(manifest, train_gen, test_gen)
        combined = pd.concat([train_part, test_part], ignore_index=True)

        model = build_model(device)
        payload = load_checkpoint(model, ckpt, device)
        config = TrainConfig(**payload["config"])
        spec = config.spec()
        amp = config.amp and device.startswith("cuda")
        model.eval()

        print(f"\n=== train {train_gen} -> test {test_gen} ===", flush=True)
        print(f"  checkpoint epoch {payload['epoch'] + 1}, val F1 {payload['val_f1']:.4f}",
              flush=True)
        print(f"  test: {test_part['label'].value_counts().to_dict()}", flush=True)

        scaler, threshold, calib = calibrate_and_threshold(
            model, combined, spec, target_fpr=0.05, device=device, amp=amp)

        val = predict(model, build_loader(combined, spec, "val", batch_size=1), device=device, amp=amp)
        in_dist = summary(val["label"], scaler.transform(val["logit"]), threshold=threshold)

        test = predict(model, build_loader(combined, spec, "test", batch_size=1), device=device, amp=amp)
        cross = summary(test["label"], scaler.transform(test["logit"]), threshold=threshold)

        # A detector that labels everything one way scores deceptively well when
        # the classes are unbalanced; state the baseline beside the result.
        n_pos = int((test["label"] == 1).sum())
        n_tot = len(test["label"])
        prec = n_pos / n_tot if n_tot else 0.0
        all_pos_f1 = 2 * prec / (prec + 1) if prec else 0.0

        print(f"  in-distribution ({train_gen}): F1 {in_dist['f1']:.4f}  AUROC {in_dist['auroc']:.4f}",
              flush=True)
        print(f"  cross-generator ({test_gen}) : F1 {cross['f1']:.4f}  AUROC {cross['auroc']:.4f}  "
              f"FPR {cross['fpr']:.3f}  ECE {cross['ece']:.4f}", flush=True)
        print(f"  all-positive baseline F1 would be {all_pos_f1:.4f}"
              f"{'  <-- result is degenerate' if abs(cross['f1'] - all_pos_f1) < 0.01 else ''}",
              flush=True)
        print(f"  collapse: F1 {in_dist['f1']:.4f} -> {cross['f1']:.4f} "
              f"({cross['f1'] - in_dist['f1']:+.4f}) | AUROC {in_dist['auroc']:.4f} -> "
              f"{cross['auroc']:.4f} ({cross['auroc'] - in_dist['auroc']:+.4f})", flush=True)

        results.append({
            "train_generator": train_gen, "test_generator": test_gen,
            "n_test": len(test_part), "in_distribution": in_dist,
            "cross_generator": cross, "all_positive_f1": all_pos_f1,
            "f1_collapse": cross["f1"] - in_dist["f1"],
            "auroc_collapse": cross["auroc"] - in_dist["auroc"],
            "calibration": calib,
        })

        # Write after every direction: the previous run lost everything by
        # holding results until the end.
        (args.out / "generator_split.json").write_text(
            json.dumps(results, indent=2, default=float))
        pd.DataFrame([{
            "direction": f"{r['train_generator']}->{r['test_generator']}",
            "n_test": r["n_test"],
            "in_dist_f1": r["in_distribution"]["f1"],
            "in_dist_auroc": r["in_distribution"]["auroc"],
            "cross_f1": r["cross_generator"]["f1"],
            "cross_auroc": r["cross_generator"]["auroc"],
            "cross_fpr": r["cross_generator"]["fpr"],
            "cross_ece": r["cross_generator"]["ece"],
            "all_positive_f1": r["all_positive_f1"],
            "f1_collapse": r["f1_collapse"],
            "auroc_collapse": r["auroc_collapse"],
        } for r in results]).to_csv(args.out / "generator_split.csv", index=False)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if results:
        print("\n=== SUMMARY ===", flush=True)
        print(pd.read_csv(args.out / "generator_split.csv").to_string(index=False), flush=True)
        print(f"\nbase paper: F1 {BASE_PAPER_IN_DIST} -> {BASE_PAPER_CROSS} "
              f"(collapse {BASE_PAPER_CROSS - BASE_PAPER_IN_DIST:+.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
