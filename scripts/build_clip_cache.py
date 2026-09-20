"""Decode each song once into a short WAV so training is not I/O bound.

The dataset needs a handful of short clips per song, but `load_audio` decodes
the whole file and resamples it -- for 200 s mp3s that dominates every epoch.
Observed: the GPU sat at 0% utilisation with memory allocated, waiting on CPU.

This decodes each song once to a `window`-second, 24 kHz mono WAV. Training then
reads a small uncompressed file with no resampling, and segmentation still has a
real window to sample offsets from, so the random-offset behaviour on the train
split is preserved.

Cost: ~2.9 MB per song at 60 s, against roughly a 50x saving in per-epoch decode.

    python scripts/build_clip_cache.py --manifest ~/sonics/manifest.csv \
        --out ~/sonics/cache --window 60
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SAMPLE_RATE = 24_000


def cache_one(row: dict, out_dir: Path, window: float, timeout: int = 120) -> tuple[str, bool]:
    """Decode one song to a trimmed, resampled mono WAV."""
    name = str(row["song_id"])
    target = out_dir / f"{name}.wav"
    if target.exists() and target.stat().st_size > 4096:
        return name, True

    tmp = target.with_suffix(".part.wav")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(row["path"]),
         "-t", str(window), "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, timeout=timeout,
    )
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 4096:
        tmp.unlink(missing_ok=True)
        return name, False
    tmp.replace(target)  # atomic, so an interrupt leaves no partial wav
    return name, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=float, default=60.0,
                    help="seconds kept per song; must exceed clip_seconds")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--write-manifest", type=Path, default=None,
                    help="write a copy of the manifest pointing at the cache")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"caching {len(manifest):,} songs at {args.window:.0f}s / {SAMPLE_RATE} Hz mono")

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(cache_one, r, args.out, args.window)
                   for r in manifest.to_dict("records")]
        for i, fut in enumerate(futures, 1):
            _, success = fut.result()
            ok += success
            if i % 100 == 0:
                print(f"  {i:,}/{len(manifest):,}  ok={ok:,}", flush=True)

    size_gb = sum(f.stat().st_size for f in args.out.glob("*.wav")) / 1024**3
    print(f"done: {ok:,}/{len(manifest):,} cached, {size_gb:.2f} GB")

    out_manifest = args.write_manifest or args.manifest.with_name(
        args.manifest.stem + "_cached.csv")
    cached = manifest.copy()
    cached["path"] = cached["song_id"].map(lambda s: str(args.out / f"{s}.wav"))
    # Drop anything that failed to decode rather than fail later at load time.
    present = {p.stem for p in args.out.glob("*.wav")}
    before = len(cached)
    cached = cached[cached["song_id"].astype(str).isin(present)]
    if len(cached) < before:
        print(f"  dropped {before - len(cached)} songs that failed to decode")
    cached.to_csv(out_manifest, index=False)
    print(f"wrote {out_manifest} ({len(cached):,} songs)")
    print(cached.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
