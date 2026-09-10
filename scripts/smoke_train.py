"""End-to-end pipeline check on synthetic audio.

This is the M2 sanity gate in miniature, and it exists to fail early. It builds
songs whose "fake" class carries a planted artefact, then runs the entire real
pipeline over them: manifest -> song-level split -> dataset -> frozen MERT ->
attention pooling -> heads -> calibration -> explanation -> faithfulness.

If the head cannot separate a deliberately obvious artefact, the data path is
broken and no amount of Kaggle GPU time will fix it. Catching that here costs
two minutes of CPU instead of hours of quota.

    python scripts/smoke_train.py            # real MERT (slower, the real gate)
    python scripts/smoke_train.py --stub     # random backbone, mechanics only
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.calibrate.temperature import TemperatureScaler  # noqa: E402
from aimd.data.datasets import ClipSpec, SongClipsDataset, collate  # noqa: E402
from aimd.data.manifest import assert_no_leakage, split_by_song, summarize  # noqa: E402
from aimd.eval.metrics import summary  # noqa: E402
from aimd.models.detector import Detector  # noqa: E402
from aimd.train.losses import DetectorLoss  # noqa: E402
from aimd.train.loop import predict, run_epoch  # noqa: E402
from aimd.xai.faithfulness import deletion_curve  # noqa: E402
from aimd.xai.relevance import temporal_relevance  # noqa: E402
from aimd.xai.shap_layers import agreement_with_attention, layer_shapley  # noqa: E402

SR = 24_000


def synth_corpus(root: Path, n_songs: int, seconds: float, seed: int = 0) -> pd.DataFrame:
    """Songs where the fake class carries a planted tonal artefact.

    A pure 6 kHz tone stands in for the kind of narrowband decoder artefact
    that Afchar et al. show real detectors latch onto. It is deliberately easy:
    the point is to prove the pipeline can learn *anything*, not to model the
    real task.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    rows = []
    for i in range(n_songs):
        is_fake = i % 2
        # Pink-ish noise as a crude musical background.
        audio = np.cumsum(rng.normal(0, 1, size=len(t)))
        audio = audio / (np.abs(audio).max() + 1e-9) * 0.3
        if is_fake:
            audio = audio + 0.15 * np.sin(2 * np.pi * 6000 * t)
        path = root / f"song{i:03d}.wav"
        sf.write(path, audio.astype(np.float32), SR)
        rows.append(
            {
                "song_id": f"s{i:03d}",
                "path": str(path),
                "label": is_fake,
                "taxonomy": "full_fake" if is_fake else "real",
                "source": "synthetic" if is_fake else None,
                "duration": seconds,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="random backbone instead of MERT")
    ap.add_argument("--songs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--clip-seconds", type=float, default=1.0)
    ap.add_argument("--clips-per-song", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device: {device}\n")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = split_by_song(synth_corpus(root, args.songs, args.clip_seconds * 3))
        assert_no_leakage(manifest)
        print("split:\n", summarize(manifest), "\n", sep="")

        spec = ClipSpec(
            clip_seconds=args.clip_seconds,
            clips_per_song=args.clips_per_song,
            max_clips=args.clips_per_song,
            sample_rate=SR,
        )

        def loader(split: str, shuffle: bool) -> DataLoader:
            rows = manifest[manifest["split"] == split]
            ds = SongClipsDataset(rows, spec, split=split)
            return DataLoader(ds, batch_size=4, shuffle=shuffle, collate_fn=collate)

        if args.stub:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
            from conftest import StubBackbone

            backbone = StubBackbone(dim=64)
        else:
            from aimd.models.backbone import MertBackbone

            print("loading MERT...")
            backbone = MertBackbone()

        model = Detector(backbone=backbone).to(device)
        loss_fn = DetectorLoss(aux_weight=0.3).to(device)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)

        train_dl, val_dl = loader("train", True), loader("val", False)
        print(f"training {sum(p.numel() for p in model.trainable_parameters()):,} params")
        first = last = None
        for epoch in range(args.epochs):
            stats = run_epoch(model, train_dl, loss_fn, optimizer, device=device)
            first = stats.loss if first is None else first
            last = stats.loss
            print(f"  epoch {epoch + 1}/{args.epochs}  loss {stats.loss:.4f}")

        print(f"\nloss {first:.4f} -> {last:.4f}")

        # --- evaluate, then calibrate on the same held-out split -------------
        val = predict(model, val_dl, device=device)
        n_val = len(val["label"])
        print(f"\nvalidation on {n_val} songs -- these metrics check plumbing, "
              "not performance; the task is synthetic and the split is tiny")
        print("uncalibrated:", {k: round(v, 3) for k, v in summary(val["label"], val["prob"]).items()})
        scaler = TemperatureScaler().fit(val["logit"], val["label"])
        calibrated = scaler.transform(val["logit"])
        print(f"temperature: {scaler.temperature:.3f}")
        print("calibrated  :", {k: round(v, 3) for k, v in summary(val["label"], calibrated).items()})

        # --- explanation layer on one song ----------------------------------
        item = SongClipsDataset(manifest[manifest["split"] == "val"], spec, split="val")[0]
        wav = item["clips"].unsqueeze(0).to(device)
        mask = item["clip_mask"].unsqueeze(0).to(device)

        rel = temporal_relevance(model, wav, mask)
        print(f"\nL2a temporal relevance: {rel.shape}, peak {rel.max():.3f}")

        hidden = model.backbone(wav.flatten(0, 1), no_grad=True)
        hidden = hidden.view(*wav.shape[:2], *hidden.shape[1:])
        phi = layer_shapley(model, hidden, mask)
        top = np.argsort(np.abs(phi))[::-1][:3]
        print(f"L3 SHAP: top layers {top.tolist()}, sum {phi.sum():+.4f}")
        print(f"L1 vs L3 agreement (Spearman): {agreement_with_attention(phi, model.layer_attn.logits.softmax(0)):+.3f}")

        curve = deletion_curve(model, wav, rel, mask, steps=4)
        verdict = "beats chance" if curve.gap < 0 else "no better than chance"
        print(f"faithfulness gap: {curve.gap:+.4f} ({verdict})")
        if last < 1e-3:
            # A saturated model has vanishing gradients, so grad-based relevance
            # degenerates to noise and the faithfulness gap becomes meaningless.
            # On this synthetic task the head separates the classes perfectly in
            # one epoch, so treat the XAI lines above as plumbing checks only.
            print("  NOTE: training loss saturated to ~0, so gradient-based relevance")
            print("        is uninformative here. Faithfulness and L1-vs-L3 agreement")
            print("        are only meaningful on a model trained on the real task.")

        learned = last < first
        print(f"\n{'PASS' if learned else 'FAIL'}: pipeline {'learned' if learned else 'did NOT learn'} the planted artefact")
        return 0 if learned else 1


if __name__ == "__main__":
    raise SystemExit(main())
