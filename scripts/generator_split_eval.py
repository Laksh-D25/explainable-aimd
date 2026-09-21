"""Cross-generator transfer within SONICS: train on one generator, test on the other.

The FakeMusicCaps comparison carries a domain gap -- a different corpus, a
different sample rate, a different clip length -- so a weak transfer number
there is hard to attribute. SONICS contains both Suno and Udio, which share a
corpus, a pipeline and a real class, so holding one out isolates the generator
and nothing else. That is the base paper's protocol, and its F1 0.99 -> 0.629
collapse is the figure to compare against.

Real songs are split disjointly between the two halves, so nothing the model saw
as "real" during training reappears at test.

Both directions are run. Transfer is not symmetric in general, and reporting one
direction would let the easier one stand for both.

    python scripts/generator_split_eval.py --manifest ~/sonics/manifest_big_cached.csv \
        --out artifacts/gensplit --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.data.manifest import assert_no_leakage  # noqa: E402
from aimd.eval.metrics import summary  # noqa: E402
from aimd.eval.protocols import build_loader  # noqa: E402
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    resolve_device,
    train_model,
)
from aimd.train.loop import predict  # noqa: E402

BASE_PAPER_IN_DIST = 0.99
BASE_PAPER_CROSS = 0.629


def build_split(manifest: pd.DataFrame, train_gen: str, test_gen: str,
                seed: int = 1337) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training manifest (train_gen + half the real songs) and a held-out test set."""
    real = manifest[manifest["label"] == 0].sample(frac=1.0, random_state=seed)
    half = len(real) // 2
    real_train, real_test = real.iloc[:half], real.iloc[half:]

    fake_train = manifest[(manifest["label"] == 1) & (manifest["source"] == train_gen)]
    fake_test = manifest[(manifest["label"] == 1) & (manifest["source"] == test_gen)]

    # Balance within each half so neither F1 is carried by the class prior.
    n_train = min(len(real_train), len(fake_train))
    n_test = min(len(real_test), len(fake_test))
    train_part = pd.concat([real_train.head(n_train), fake_train.head(n_train)])
    test_part = pd.concat([real_test.head(n_test), fake_test.head(n_test)])

    # Training half gets its own train/val; the test half is never trained on.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_part))
    n_val = max(int(0.2 * len(train_part)), 8)
    train_part = train_part.copy()
    train_part["split"] = "train"
    train_part.iloc[idx[:n_val], train_part.columns.get_loc("split")] = "val"

    test_part = test_part.copy()
    test_part["split"] = "test"

    overlap = set(train_part["song_id"]) & set(test_part["song_id"])
    assert not overlap, f"{len(overlap)} songs in both halves"
    return train_part, test_part


def run_direction(manifest: pd.DataFrame, train_gen: str, test_gen: str,
                  config: TrainConfig, device: str, out: Path) -> dict:
    train_part, test_part = build_split(manifest, train_gen, test_gen)
    combined = pd.concat([train_part, test_part], ignore_index=True)
    assert_no_leakage(combined)

    print(f"\n=== train on {train_gen}, test on {test_gen} ===", flush=True)
    print(f"  train/val: {train_part.groupby(['split','label']).size().to_dict()}", flush=True)
    print(f"  test     : {test_part['label'].value_counts().to_dict()}", flush=True)

    torch.manual_seed(1337)
    model = build_model(device)
    result = train_model(model, combined, config, device=device,
                         checkpoint_dir=out / f"ckpt_{train_gen}", log=lambda m: print("   ", m, flush=True))

    spec = config.spec()
    amp = config.amp and device.startswith("cuda")
    scaler, threshold, calib = calibrate_and_threshold(
        model, combined, spec, target_fpr=0.05, device=device, amp=amp)

    # In-distribution reference: the validation half, same generator.
    val = predict(model, build_loader(combined, spec, "val"), device=device, amp=amp)
    in_dist = summary(val["label"], scaler.transform(val["logit"]), threshold=threshold)

    # The transfer number: unseen generator, unseen real songs.
    test = predict(model, build_loader(combined, spec, "test"), device=device, amp=amp)
    cross = summary(test["label"], scaler.transform(test["logit"]), threshold=threshold)

    print(f"  in-distribution ({train_gen}): F1 {in_dist['f1']:.4f}  AUROC {in_dist['auroc']:.4f}", flush=True)
    print(f"  cross-generator ({test_gen}) : F1 {cross['f1']:.4f}  AUROC {cross['auroc']:.4f}  "
          f"FPR {cross['fpr']:.3f}", flush=True)
    print(f"  collapse: F1 {in_dist['f1']:.4f} -> {cross['f1']:.4f} "
          f"({cross['f1'] - in_dist['f1']:+.4f})", flush=True)

    return {
        "train_generator": train_gen,
        "test_generator": test_gen,
        "n_train": int((train_part["split"] == "train").sum()),
        "n_val": int((train_part["split"] == "val").sum()),
        "n_test": len(test_part),
        "in_distribution": in_dist,
        "cross_generator": cross,
        "f1_collapse": cross["f1"] - in_dist["f1"],
        "calibration": calib,
        "best_val_f1": result["best"]["f1"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    device = resolve_device("auto")
    manifest = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                         clip_seconds=5.0, clips_per_song=4, augment=True,
                         num_workers=2, amp=device.startswith("cuda"))

    results = []
    for train_gen, test_gen in [("suno", "udio"), ("udio", "suno")]:
        try:
            results.append(run_direction(manifest, train_gen, test_gen, config, device, args.out))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {train_gen}->{test_gen}; skipped", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if results:
        rows = [{
            "direction": f"{r['train_generator']}->{r['test_generator']}",
            "n_test": r["n_test"],
            "in_dist_f1": r["in_distribution"]["f1"],
            "cross_f1": r["cross_generator"]["f1"],
            "cross_auroc": r["cross_generator"]["auroc"],
            "cross_fpr": r["cross_generator"]["fpr"],
            "cross_ece": r["cross_generator"]["ece"],
            "f1_collapse": r["f1_collapse"],
        } for r in results]
        table = pd.DataFrame(rows)
        table.to_csv(args.out / "generator_split.csv", index=False)
        (args.out / "generator_split.json").write_text(
            json.dumps(results, indent=2, default=float))
        print("\n=== SUMMARY ===")
        print(table.to_string(index=False))
        print(f"\nbase paper: F1 {BASE_PAPER_IN_DIST} in-distribution -> "
              f"{BASE_PAPER_CROSS} cross-generator "
              f"(collapse {BASE_PAPER_CROSS - BASE_PAPER_IN_DIST:+.3f})")
        mean_collapse = table["f1_collapse"].mean()
        print(f"this work: mean collapse {mean_collapse:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
