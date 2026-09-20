"""Stream a SONICS fake-songs part and stop once enough tracks are extracted.

Parts are ~3.5 GB and this connection runs at ~200 KB/s, so a full part is a
five-hour download. A balanced experiment against a few hundred real songs needs
only a few hundred generated ones -- roughly 200 MB.

A zip's central directory sits at the end of the file, so the archive cannot be
opened until it is fully downloaded. Streaming reads the local file headers
instead, decompressing entries as they arrive, which makes it possible to take
the first N and abandon the rest of the transfer.

The connection resets often. A zip stream cannot resume from the middle -- the
reader must see local headers in order -- so a reset restarts the stream and
skips entries already on disk. Bytes are re-read, but no file is fetched twice.

    python scripts/stream_fetch_fake.py --out ~/sonics/fake_songs --limit 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
from stream_unzip import stream_unzip

URL = ("https://huggingface.co/datasets/awsaf49/sonics/resolve/main/"
       "fake_songs/part_{part:02d}.zip")


def stream_once(url: str, out: Path, limit: int, start: float) -> int:
    """One pass over the archive. Returns how many new files were written."""
    existing = {p.stem for p in out.glob("*.mp3")}
    written = 0
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for name, _size, unzipped in stream_unzip(resp.iter_content(chunk_size=262144)):
            filename = Path(name.decode(errors="replace")).name
            keep = filename.endswith(".mp3") and Path(filename).stem not in existing
            if not keep:
                for _ in unzipped:
                    pass  # an entry's chunks must be drained even when skipped
                continue

            tmp = (out / filename).with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in unzipped:
                    fh.write(chunk)
            tmp.rename(out / filename)  # atomic: no half-written mp3 on interrupt
            written += 1

            if written % 25 == 0:
                total = len(existing) + written
                print(f"  {total:,} tracks  {time.time() - start:.0f}s elapsed", flush=True)
            if len(existing) + written >= limit:
                return written
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--retries", type=int, default=15)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    have = len(list(args.out.glob("*.mp3")))
    if have >= args.limit:
        print(f"already have {have:,} tracks")
        return 0

    url = URL.format(part=args.part)
    start = time.time()
    print(f"streaming part_{args.part:02d} for {args.limit - have:,} more tracks", flush=True)

    for attempt in range(1, args.retries + 1):
        try:
            stream_once(url, args.out, args.limit, start)
        except Exception as exc:  # connection resets are the norm on this link
            have = len(list(args.out.glob("*.mp3")))
            if have >= args.limit:
                break
            print(f"  connection lost ({type(exc).__name__}) at {have:,} tracks; "
                  f"retry {attempt}/{args.retries}", flush=True)
            time.sleep(min(5 * attempt, 30))
            continue
        break

    total = len(list(args.out.glob("*.mp3")))
    print(f"done: {total:,} tracks in {args.out} after {time.time() - start:.0f}s")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
