"""Finish an interrupted run: explanations and figures, from a saved checkpoint.

The machine hard-powered-off twice mid-run, so the expensive stages are made
resumable rather than repeated. Training, evaluation and robustness already
wrote their CSVs; this reloads the checkpoint and produces what is missing.

Explanations are appended to `faithfulness.csv` after each song, so another
crash costs one song rather than the whole pass.

    python scripts/finish_experiment.py --manifest ~/sonics/manifest_cached.csv \
        --out artifacts/run1
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
from aimd.eval.protocols import build_loader, rows_for_split  # noqa: E402
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
    ap.add_argument("--explain-songs", type=int, default=12)
    args = ap.parse_args()

    device = resolve_device("auto")
    manifest = pd.read_csv(args.manifest)
    model = build_model(device)
    payload = load_checkpoint(model, args.out / "ckpt" / "best.pt", device)
    config = TrainConfig(**payload["config"])
    spec, amp = config.spec(), config.amp and device.startswith("cuda")
    model.eval()
    print(f"resumed checkpoint from epoch {payload['epoch'] + 1} "
          f"(val F1 {payload['val_f1']:.4f})", flush=True)

    scaler, threshold, calib = calibrate_and_threshold(
        model, manifest, spec, target_fpr=0.05, device=device, amp=amp)
    print(f"calibration: {calib}", flush=True)

    # --- explanations, appended per song so a crash costs one song ------------
    audit_path = args.out / "faithfulness.csv"
    done = set()
    if audit_path.exists():
        done = set(pd.read_csv(audit_path)["song_id"].astype(str))
        print(f"  {len(done)} songs already explained; skipping them", flush=True)

    test_rows = rows_for_split(manifest, "test")
    # Stratify: the manifest is class-sorted, so head() draws a single class and
    # the explanation audit would cover only real songs.
    per_class = max(args.explain_songs // 2, 1)
    chosen = pd.concat(
        [g.sample(n=min(per_class, len(g)), random_state=1337)
         for _, g in test_rows.groupby("label")],
        ignore_index=True,
    )
    print(f"  explaining {len(chosen)} songs: "
          f"{chosen['label'].value_counts().to_dict()}", flush=True)
    ds = SongClipsDataset(chosen, spec, split="test")
    for i in range(len(ds)):
        item = ds[i]
        if str(item["song_id"]) in done:
            continue
        wav = item["clips"].unsqueeze(0).to(device)
        mask = item["clip_mask"].unsqueeze(0).to(device)

        rel = temporal_relevance(model, wav, mask)
        hidden = model.backbone(wav.flatten(0, 1), no_grad=True)
        phi = layer_shapley(model, hidden.view(*wav.shape[:2], *hidden.shape[1:]), mask)
        curve = deletion_curve(model, wav, rel, mask, steps=8)

        row = pd.DataFrame([{
            "song_id": item["song_id"],
            "label": int(item["label"].item()),
            "faithfulness_gap": curve.gap,
            "beats_chance": bool(curve.gap < 0),
            "l1_l3_agreement": agreement_with_attention(
                phi, model.layer_attn.logits.softmax(0)),
        }])
        row.to_csv(audit_path, mode="a", header=not audit_path.exists(), index=False)
        print(f"  explained {item['song_id']} gap={curve.gap:+.4f}", flush=True)

        if not (args.out / "fig_faithfulness.pdf").exists():
            faithfulness_curves(curve, args.out / "fig_faithfulness")
            layer_profile(model.layer_attn.logits.softmax(0).detach().cpu().numpy(),
                          phi, args.out / "fig_layers")

    # --- figures -------------------------------------------------------------
    val = predict(model, build_loader(manifest, spec, "val"), device=device, amp=amp)
    reliability_diagram(val["label"], val["prob"], scaler.transform(val["logit"]),
                        args.out / "fig_calibration")
    rob = pd.read_csv(args.out / "robustness.csv", index_col="condition")
    robustness_chart(rob, args.out / "fig_robustness")

    audit = pd.read_csv(audit_path)
    summary = {
        "calibration": calib,
        "faithfulness_beats_chance": float(audit["beats_chance"].mean()),
        "l1_l3_agreement": float(audit["l1_l3_agreement"].mean()),
        "n_explained": len(audit),
    }
    (args.out / "xai_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nexplanations beating chance: {audit['beats_chance'].mean():.0%} of {len(audit)}")
    print(f"mean L1-vs-L3 agreement: {audit['l1_l3_agreement'].mean():+.3f}")
    for f in sorted(args.out.glob("fig_*.pdf")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
