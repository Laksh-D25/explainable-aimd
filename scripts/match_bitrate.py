"""Re-encode one class so both traverse the same codec chain.

`bandwidth_report` flagged a 1.20x spectral-rolloff gap between the real songs
fetched from YouTube (48 kbps) and SONICS's generated set (22-57 kbps, mean 37).
A detector can separate those on bandwidth alone, which scores well and measures
the encoder rather than the generator.

Rather than pick one bitrate, each file is re-encoded at a bitrate drawn from
the *other* class's actual distribution, so the two classes end up with matching
codec statistics instead of merely matching averages.

    python scripts/match_bitrate.py --audio-dir ~/sonics/real_songs \
        --match-csv ~/sonics/metadata/fake_songs.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def reencode(path: Path, bitrate_kbps: int, timeout: int = 120) -> tuple[Path, bool]:
    """Re-encode in place via a temporary file, so an interrupt cannot corrupt."""
    tmp = path.with_suffix(".reenc.mp3")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", "-ac", "1", str(tmp)],
        capture_output=True, timeout=timeout,
    )
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        return path, False
    tmp.replace(path)
    return path, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--match-csv", type=Path, required=True,
                    help="CSV with a bit_rate column describing the target class")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--backup", type=Path, default=None,
                    help="copy originals here before re-encoding")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")

    target = pd.read_csv(args.match_csv, low_memory=False)
    if "bit_rate" not in target.columns:
        raise SystemExit(f"{args.match_csv} has no bit_rate column")
    rates = (target["bit_rate"].dropna().to_numpy() / 1000).astype(int)
    rates = rates[(rates >= 16) & (rates <= 128)]
    print(f"target bitrate distribution: median {np.median(rates):.0f} kbps, "
          f"range {rates.min()}-{rates.max()} ({len(rates):,} samples)")

    files = sorted(args.audio_dir.glob("*.mp3"))
    if not files:
        raise SystemExit(f"no mp3 files in {args.audio_dir}")
    print(f"re-encoding {len(files):,} files in {args.audio_dir}")

    if args.backup:
        args.backup.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = args.backup / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
        print(f"  originals copied to {args.backup}")

    rng = np.random.default_rng(1337)
    assigned = rng.choice(rates, size=len(files))

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(reencode, f, int(b)) for f, b in zip(files, assigned)]
        for i, fut in enumerate(futures, 1):
            _, success = fut.result()
            ok += success
            if i % 50 == 0:
                print(f"  {i:,}/{len(files):,}  ok={ok:,}", flush=True)

    print(f"done: {ok:,}/{len(files):,} re-encoded")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
