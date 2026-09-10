"""Measure what actually fits in GPU memory, per configuration.

Batch sizes in the training configs should come from measurement, not from
guesswork: the two paths through the model have very different memory profiles.

* The **training** path runs the frozen backbone under `no_grad`, so only the
  13 x T x 768 hidden states and the small trainable tail are resident.
* The **XAI** path needs gradients reaching the audio, so every activation
  inside MERT's 12 transformer layers is retained. This is the path that
  decides whether explanations can run locally at all.

Run on any target (4 GB laptop, Kaggle T4) to size `configs/train/*.yaml`:

    python scripts/bench_vram.py
    python scripts/bench_vram.py --clip-seconds 5 --max-batch 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.models.backbone import MertBackbone  # noqa: E402
from aimd.models.detector import Detector  # noqa: E402


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def probe(model: Detector, batch: int, clips: int, seconds: float, xai: bool) -> float | None:
    """Peak GB for one step, or None if it does not fit."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n = int(seconds * model.backbone.sample_rate)
    try:
        wav = torch.randn(batch, clips, n, device="cuda", requires_grad=xai)
        out = model.forward(wav, backbone_no_grad=not xai)
        # A backward pass is part of both paths: training updates the tail,
        # XAI attributes back to the audio. Measuring forward only would
        # understate peak memory by a wide margin.
        out.binary_logit.sum().backward()
        peak = _peak_gb()
        del wav, out
        return peak
    except torch.OutOfMemoryError:
        return None
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-seconds", type=float, default=10.0)
    ap.add_argument("--clips-per-song", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; nothing to measure")
        return 1

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{name} — {total:.1f} GB total\n")

    model = Detector(backbone=MertBackbone()).cuda()
    frames = int(args.clip_seconds * 75)
    print(f"clip={args.clip_seconds}s (~{frames} frames), clips/song={args.clips_per_song}\n")

    for label, xai in [("train  (backbone no_grad)", False), ("xai    (grads to audio)", True)]:
        largest = None
        for batch in [1, 2, 4, 8, 12, 16]:
            if batch > args.max_batch:
                break
            peak = probe(model, batch, args.clips_per_song, args.clip_seconds, xai)
            if peak is None:
                print(f"  {label}  batch={batch:<3} OOM")
                break
            largest = (batch, peak)
            print(f"  {label}  batch={batch:<3} peak={peak:5.2f} GB")
        if largest:
            print(f"  -> largest fitting batch: {largest[0]} at {largest[1]:.2f} GB\n")
        else:
            print("  -> nothing fits; reduce clip_seconds or clips_per_song\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
