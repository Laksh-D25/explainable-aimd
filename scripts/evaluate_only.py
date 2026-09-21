"""Evaluate a trained checkpoint without retraining, at low memory.

The overnight run trained successfully and then hit CUDA OOM in evaluation: the
VRAM cap that stops the desktop freezing (2.38 GiB) is tighter than the
explanation pass needs at batch 2. The cap is doing its job -- it converted a
hard desktop freeze into a recoverable error -- so the fix is to evaluate within
it rather than to raise it.

Batch 1, and expandable segments to avoid the fragmentation the error itself
pointed at ("211 MiB reserved but unallocated").

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python scripts/evaluate_only.py --manifest ~/sonics/manifest_big_cached.csv \
      --out artifacts/run2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    robustness_chart,
)
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    load_checkpoint,
    resolve_device,
)
from aimd.train.loop import predict  # noqa: E402
from aimd.xai.faithfulness import deletion_curve  # noqa: E402
from aimd.xai.relevance import temporal_relevance  # noqa: E402
from aimd.xai.shap_layers import agreement_with_attention, layer_shapley  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--explain-songs", type=int, default=12)
    ap.add_argument("--skip-explanations", action="store_true")
    args = ap.parse_args()

    device = resolve_device("auto")
    manifest = pd.read_csv(args.manifest)
    model = build_model(device)
    payload = load_checkpoint(model, args.out / "ckpt" / "best.pt", device)
    config = TrainConfig(**payload["config"])
    spec = config.spec()
    amp = config.amp and device.startswith("cuda")
    model.eval()
    print(f"checkpoint: epoch {payload['epoch'] + 1}, val F1 {payload['val_f1']:.4f}",
          flush=True)
    print(f"test split: {manifest[manifest['split'] == 'test']['label'].value_counts().to_dict()}",
          flush=True)

    scaler, threshold, calib = calibrate_and_threshold(
        model, manifest, spec, target_fpr=0.05, device=device, amp=amp)
    print(f"calibration: T={calib['temperature']:.4f} "
          f"ECE {calib['ece_before']:.6f} -> {calib['ece_after']:.6f}", flush=True)

    print("\n=== in-distribution ===", flush=True)
    indist = evaluate_in_distribution(model, manifest, spec, scaler, threshold,
                                      device=device, amp=amp)
    print(f"n={indist.n_songs} " +
          json.dumps({k: round(v, 4) for k, v in indist.metrics.items()}), flush=True)
    results_table([indist]).to_csv(args.out / "protocols.csv")

    print("\n=== robustness ===", flush=True)
    rob = evaluate_robustness(model, manifest, spec, scaler, threshold,
                              device=device, amp=amp)
    rob.to_csv(args.out / "robustness.csv")
    print(rob[["f1", "f1_drop", "family"]].to_string(), flush=True)

    val = predict(model, build_loader(manifest, spec, "val"), device=device, amp=amp)
    reliability_diagram(val["label"], val["prob"], scaler.transform(val["logit"]),
                        args.out / "fig_calibration")
    robustness_chart(rob, args.out / "fig_robustness")

    if not args.skip_explanations:
        print("\n=== explanations ===", flush=True)
        audit_path = args.out / "faithfulness.csv"
        done = set(pd.read_csv(audit_path)["song_id"].astype(str)) if audit_path.exists() else set()
        test_rows = rows_for_split(manifest, "test")
        per_class = max(args.explain_songs // 2, 1)
        chosen = pd.concat(
            [g.sample(n=min(per_class, len(g)), random_state=1337)
             for _, g in test_rows.groupby("label")], ignore_index=True)
        ds = SongClipsDataset(chosen, spec, split="test")
        for i in range(len(ds)):
            item = ds[i]
            if str(item["song_id"]) in done:
                continue
            wav = item["clips"].unsqueeze(0).to(device)
            mask = item["clip_mask"].unsqueeze(0).to(device)
            try:
                rel = temporal_relevance(model, wav, mask)
                hidden = model.backbone(wav.flatten(0, 1), no_grad=True)
                phi = layer_shapley(model, hidden.view(*wav.shape[:2], *hidden.shape[1:]), mask)
                curve = deletion_curve(model, wav, rel, mask, steps=8)
            except torch.OutOfMemoryError:
                # One song failing must not cost the whole audit.
                torch.cuda.empty_cache()
                print(f"  {item['song_id']}: OOM, skipped", flush=True)
                continue
            pd.DataFrame([{
                "song_id": item["song_id"],
                "label": int(item["label"].item()),
                "faithfulness_gap": curve.gap,
                "beats_chance": bool(curve.gap < 0),
                "l1_l3_agreement": agreement_with_attention(
                    phi, model.layer_attn.logits.softmax(0)),
            }]).to_csv(audit_path, mode="a", header=not audit_path.exists(), index=False)
            print(f"  {item['song_id']} gap={curve.gap:+.4f}", flush=True)
            if not (args.out / "fig_faithfulness.pdf").exists():
                faithfulness_curves(curve, args.out / "fig_faithfulness")
                layer_profile(model.layer_attn.logits.softmax(0).detach().cpu().numpy(),
                              phi, args.out / "fig_layers")
            torch.cuda.empty_cache()

    summary = {"in_distribution": indist.metrics, "n_test": indist.n_songs,
               "calibration": calib,
               "robustness": rob["f1_drop"].drop(index="clean", errors="ignore").to_dict()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {args.out}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
