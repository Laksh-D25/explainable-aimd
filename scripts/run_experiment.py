"""Run the full experiment end to end and write every number the paper needs.

Sequence: manifest -> train -> calibrate -> evaluate -> robustness -> ablations
-> explanations -> figures. Everything lands in one output directory as CSV
plus PDF/PNG, so the write-up reads from files rather than from scrollback.

Scale is set by what is actually on disk. With a few hundred songs per class the
numbers are real but wide-error; the summary states the sample size next to every
metric so nothing can be quoted without it.

    python scripts/run_experiment.py --manifest ~/sonics/manifest.csv --out artifacts/run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.data.audio import bandwidth_report  # noqa: E402
from aimd.data.datasets import SongClipsDataset  # noqa: E402
from aimd.eval.protocols import (  # noqa: E402
    build_loader,
    evaluate_in_distribution,
    evaluate_robustness,
    results_table,
    rows_for_split,
)
from aimd.eval.report import (  # noqa: E402
    faithfulness_curves,
    layer_profile,
    reliability_diagram,
    results_markdown,
    robustness_chart,
)
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    resolve_device,
    train_model,
)
from aimd.train.loop import predict  # noqa: E402
from aimd.xai.faithfulness import deletion_curve  # noqa: E402
from aimd.xai.relevance import temporal_relevance  # noqa: E402
from aimd.xai.shap_layers import agreement_with_attention, layer_shapley  # noqa: E402


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--clips-per-song", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--explain-songs", type=int, default=12)
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    device = resolve_device("auto")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)

    manifest = pd.read_csv(args.manifest)
    log(f"device: {device} | {len(manifest):,} songs")
    log(f"  {manifest.groupby(['split', 'label']).size().unstack(fill_value=0).to_string()}")

    # A codec/bandwidth gap between classes produces an excellent score that
    # measures the encoder rather than the generator, so it is checked first.
    log("\n=== bandwidth check ===")
    bw = bandwidth_report(manifest, n_per_class=30)
    log(f"  {bw}")
    if bw.get("suspicious"):
        log("  WARNING: classes differ in bandwidth; any score below is suspect")

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
    amp = config.amp

    log("\n=== training ===")
    model = build_model(device)
    log(f"  trainable {sum(p.numel() for p in model.trainable_parameters()):,} | "
        f"frozen {sum(p.numel() for p in model.backbone.parameters()):,}")
    t0 = time.time()
    result = train_model(model, manifest, config, device=device,
                         checkpoint_dir=args.out / "ckpt", log=log)
    result["history"].to_csv(args.out / "history.csv", index=False)
    log(f"  {time.time() - t0:.0f}s | best val F1 {result['best']['f1']:.4f} "
        f"(epoch {result['best']['epoch'] + 1})")

    log("\n=== calibration ===")
    scaler, threshold, calib = calibrate_and_threshold(
        model, manifest, spec, target_fpr=0.05, device=device, amp=amp)
    log(f"  {json.dumps({k: round(v, 5) for k, v in calib.items()})}")

    log("\n=== in-distribution ===")
    indist = evaluate_in_distribution(model, manifest, spec, scaler, threshold,
                                      device=device, amp=amp)
    log(f"  n={indist.n_songs} " +
        json.dumps({k: round(v, 4) for k, v in indist.metrics.items()}))
    table = results_table([indist])
    table.to_csv(args.out / "protocols.csv")

    log("\n=== robustness (held-out conditions) ===")
    rob = evaluate_robustness(model, manifest, spec, scaler, threshold,
                              device=device, amp=amp)
    rob.to_csv(args.out / "robustness.csv")
    log(rob[["f1", "f1_drop", "family"]].to_string())

    ablations = pd.DataFrame()
    if not args.skip_ablations:
        log("\n=== ablations ===")
        rows = []
        variants = {
            "baseline": {},
            "pool_mean": {"frame_pool": "mean"},
            "layers_last": {"layer_mode": "last"},
            "song_mean": {"song_pool": "mean"},
        }
        # Two backbones do not fit under the display-safe VRAM cap, and the
        # trained model is not needed again until the explanations below.
        model.to("cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        for name, overrides in variants.items():
            torch.manual_seed(1337)
            m = build_model(device, **overrides)
            train_model(m, manifest, config, device=device, log=lambda _m: None)
            s, t, _ = calibrate_and_threshold(m, manifest, spec, target_fpr=0.05,
                                              device=device, amp=amp)
            r = evaluate_in_distribution(m, manifest, spec, s, t, device=device, amp=amp)
            rows.append({"ablation": name, **r.metrics})
            log(f"  {name:14s} F1 {r.metrics['f1']:.4f}  AUROC {r.metrics['auroc']:.4f}")
            del m
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        ablations = pd.DataFrame(rows).set_index("ablation")
        ablations.to_csv(args.out / "ablations.csv")
        model.to(device)

    log("\n=== explanations ===")
    model.eval()
    test_rows = rows_for_split(manifest, "test")
    ds = SongClipsDataset(test_rows.head(args.explain_songs), spec, split="test")
    audit = []
    for i in range(len(ds)):
        item = ds[i]
        wav = item["clips"].unsqueeze(0).to(device)
        mask = item["clip_mask"].unsqueeze(0).to(device)
        rel = temporal_relevance(model, wav, mask)
        hidden = model.backbone(wav.flatten(0, 1), no_grad=True)
        phi = layer_shapley(model, hidden.view(*wav.shape[:2], *hidden.shape[1:]), mask)
        curve = deletion_curve(model, wav, rel, mask, steps=8)
        audit.append({
            "song_id": item["song_id"],
            "label": int(item["label"].item()),
            "faithfulness_gap": curve.gap,
            "beats_chance": bool(curve.gap < 0),
            "l1_l3_agreement": agreement_with_attention(
                phi, model.layer_attn.logits.softmax(0)),
        })
        if i == 0:
            faithfulness_curves(curve, args.out / "fig_faithfulness")
            layer_profile(model.layer_attn.logits.softmax(0).detach().cpu().numpy(),
                          phi, args.out / "fig_layers")
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(args.out / "faithfulness.csv", index=False)
    log(f"  explanations beating chance: {audit_df['beats_chance'].mean():.0%} "
        f"of {len(audit_df)}")
    log(f"  mean L1-vs-L3 agreement: {audit_df['l1_l3_agreement'].mean():+.3f}")

    log("\n=== figures ===")
    val = predict(model, build_loader(manifest, spec, "val"), device=device, amp=amp)
    reliability_diagram(val["label"], val["prob"], scaler.transform(val["logit"]),
                        args.out / "fig_calibration")
    robustness_chart(rob, args.out / "fig_robustness")
    for f in sorted(args.out.glob("fig_*.pdf")):
        log(f"  {f.name}")

    summary = {
        "n_songs": len(manifest),
        "n_test": int(indist.n_songs),
        "clip_seconds": args.clip_seconds,
        "clips_per_song": args.clips_per_song,
        "epochs": args.epochs,
        "best_val_f1": result["best"]["f1"],
        "in_distribution": indist.metrics,
        "calibration": calib,
        "robustness": rob["f1_drop"].drop(index="clean", errors="ignore").to_dict(),
        "faithfulness_beats_chance": float(audit_df["beats_chance"].mean()),
        "l1_l3_agreement": float(audit_df["l1_l3_agreement"].mean()),
        "bandwidth": bw,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (args.out / "results.md").write_text(results_markdown(table))

    log("\n=== summary ===")
    log(f"  test set is {indist.n_songs} songs -- quote every metric with that n")
    log(json.dumps(summary["in_distribution"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
