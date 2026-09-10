"""Fetch SONICS real songs from YouTube, resumably.

SONICS ships only generated audio; the real class must be sourced from the
`youtube_id` column, as its README instructs. At ~26 hours for the full 48,090
this *will* be interrupted, so resume is a first-class concern rather than an
afterthought.

Three details matter for the result, not just for convenience:

* **Bitrate is matched to the generated set.** SONICS's fake songs sit at
  22-57 kbps (mean 37). Downloading real audio at 192 kbps would let a detector
  separate the classes on codec artefacts alone -- an excellent score that
  measures the encoder, not the generator. Default is 48 kbps.
* **The clip window is honoured.** `real_songs.csv` carries `skip_time` and
  `duration`; the YouTube video is usually longer, and taking all of it would
  include material the dataset excludes.
* **Coverage is recorded.** Roughly one video in eight is gone, private or
  region-locked. Missing songs are logged, never silently dropped, so the
  write-up can state real coverage.

How resume works:

* Every attempt appends one JSON line to `fetch_state.jsonl` **as it finishes**,
  so an interrupt at any moment loses at most the in-flight downloads.
* On restart, songs already `ok` are skipped, and so are *permanent* failures
  (removed, private, blocked) -- retrying those every run would waste hours.
  *Transient* failures (timeout, network) are retried automatically.
* Completion is verified against the audio, not the file's existence: an
  interrupted download can leave a plausible-looking partial file behind, and
  trusting size alone silently yields truncated clips.

    python scripts/fetch_real_songs.py --csv real_songs.csv --out real_songs/
    # ...interrupt with Ctrl-C at any point, then re-run the same command.

    python scripts/fetch_real_songs.py ... --limit 4000       # balanced subset
    python scripts/fetch_real_songs.py ... --retry-failed     # also retry permanent
    python scripts/fetch_real_songs.py ... --shard-size 1500  # tar shards for Drive
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import pandas as pd

BITRATE_MATCHING_SONICS = "48K"

#: Retrying these on every resume would waste hours for no chance of success.
PERMANENT = {"private", "removed", "unavailable", "blocked", "age_restricted", "too_short"}
#: These are worth another attempt on the next run.
TRANSIENT = {"timeout", "failed", "network"}

_stop = threading.Event()


def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def is_complete(path: Path, expected: float, tolerance: float = 0.25) -> bool:
    """Whether an existing file is a usable clip rather than a partial download.

    Size alone is not enough: an interrupted fetch leaves a large-but-truncated
    file that would otherwise be accepted on resume and quietly train on short
    audio. Short clips are legitimate when the source video is shorter than the
    dataset's window, so the check is a fraction of the expected length.
    """
    if not path.exists() or path.stat().st_size < 4096:
        return False
    if expected <= 0:
        return True
    duration = _probe_duration(path)
    return duration is not None and duration >= expected * (1 - tolerance)


class State:
    """Append-only progress log, flushed per completed song."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.entries: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self.entries[rec["filename"]] = rec["status"]
                except (json.JSONDecodeError, KeyError):
                    continue  # a torn final line from a hard kill

    def record(self, filename: str, status: str) -> None:
        with self.lock:
            self.entries[filename] = status
            with self.path.open("a") as fh:
                fh.write(json.dumps({"filename": filename, "status": status,
                                     "ts": time.time()}) + "\n")
                fh.flush()

    def should_skip(self, filename: str, retry_failed: bool) -> bool:
        status = self.entries.get(filename)
        if status is None:
            return False
        if status == "ok":
            return True
        return status in PERMANENT and not retry_failed


def classify_error(stderr: bytes) -> str:
    err = (stderr or b"").decode(errors="replace").lower()
    for marker, status in [
        ("private", "private"), ("removed", "removed"), ("unavailable", "unavailable"),
        ("blocked", "blocked"), ("not available in your country", "blocked"),
        ("age", "age_restricted"), ("sign in to confirm", "blocked"),
    ]:
        if marker in err:
            return status
    return "failed"


def fetch_one(row: dict, out_dir: Path, bitrate: str, timeout: int) -> tuple[str, str]:
    name = str(row["filename"])
    target = out_dir / f"{name}.mp3"
    expected = float(row.get("duration") or 0.0)

    if is_complete(target, expected):
        return name, "ok"
    target.unlink(missing_ok=True)  # drop any partial before retrying

    if _stop.is_set():
        return name, "interrupted"

    start = float(row.get("skip_time") or 0.0)
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "--no-progress",
        "-f", "bestaudio/best", "-x", "--audio-format", "mp3",
        "--audio-quality", bitrate,
        "-o", str(out_dir / f"{name}.%(ext)s"),
    ]
    if expected > 0:
        cmd += ["--download-sections", f"*{start:.2f}-{start + expected:.2f}",
                "--force-keyframes-at-cuts"]
    cmd.append(f"https://www.youtube.com/watch?v={row['youtube_id']}")

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        return name, "timeout"

    if is_complete(target, expected):
        return name, "ok"
    if target.exists():
        # Downloaded but shorter than the window -- the source video is shorter
        # than the dataset assumed. Keep it and mark it, rather than retrying
        # forever on something that will never get longer.
        duration = _probe_duration(target) or 0.0
        if duration > 5:
            return name, "too_short"
        target.unlink(missing_ok=True)
    return name, classify_error(proc.stderr)


def shard(out_dir: Path, size: int) -> None:
    """Pack into tar shards.

    Cloud storage and notebook mounts handle a few large files far better than
    tens of thousands of small ones -- reading 48k individual files over a FUSE
    mount makes training I/O-bound. Copy one shard to local disk and extract.
    Existing shards are left alone so this is resumable too.
    """
    files = sorted(out_dir.glob("*.mp3"))
    shard_dir = out_dir.parent / f"{out_dir.name}_shards"
    shard_dir.mkdir(exist_ok=True)
    for i in range(0, len(files), size):
        path = shard_dir / f"real_songs_{i // size:03d}.tar"
        if path.exists():
            print(f"  {path.name}  (exists, skipped)")
            continue
        tmp = path.with_suffix(".tar.part")
        with tarfile.open(tmp, "w") as tar:
            for f in files[i : i + size]:
                tar.add(f, arcname=f.name)
        tmp.rename(tmp.with_suffix(""))  # atomic: never leave a half-written shard
        print(f"  {path.name}  {path.stat().st_size / 1024**3:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="SONICS real_songs.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bitrate", default=BITRATE_MATCHING_SONICS)
    ap.add_argument("--limit", type=int, default=None, help="fetch only N (subset)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry permanent failures (removed, private, blocked)")
    ap.add_argument("--shard-size", type=int, default=None)
    args = ap.parse_args()

    rows = pd.read_csv(args.csv, low_memory=False)
    if args.split:
        rows = rows[rows["split"] == args.split]
    if args.limit:
        # Sample rather than head: the CSV is ordered and its head is not
        # representative. Fixed seed so a resumed subset is the SAME subset.
        rows = rows.sample(n=min(args.limit, len(rows)), random_state=1337)

    args.out.mkdir(parents=True, exist_ok=True)
    state = State(args.out.parent / "fetch_state.jsonl")

    pending = [r for r in rows.to_dict("records")
               if not state.should_skip(str(r["filename"]), args.retry_failed)]
    done = len(rows) - len(pending)
    print(f"{len(rows):,} songs | {done:,} already resolved | {len(pending):,} to fetch")
    if not pending:
        print("nothing to do")
    else:
        print(f"  {args.workers} workers, {args.bitrate}, state -> {state.path}")

    def on_signal(_sig, _frame):
        if not _stop.is_set():
            print("\ninterrupt: finishing in-flight downloads, progress is saved...",
                  flush=True)
            _stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    start_time = time.time()
    completed = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, r, args.out, args.bitrate, args.timeout): r
            for r in pending
        }
        for fut in cf.as_completed(futures):
            name, status = fut.result()
            if status != "interrupted":
                state.record(name, status)
            completed += 1
            if completed % 50 == 0 or completed == len(futures):
                rate = completed / max(time.time() - start_time, 1e-9)
                eta = (len(futures) - completed) / max(rate, 1e-9) / 3600
                ok = sum(1 for s in state.entries.values() if s == "ok")
                print(f"  {completed:,}/{len(futures):,}  ok={ok:,}  "
                      f"{rate:.1f}/s  eta {eta:.1f}h", flush=True)

    report = pd.DataFrame(sorted(state.entries.items()), columns=["filename", "status"])
    report.to_csv(args.out.parent / "fetch_report.csv", index=False)
    print("\nstatus:", report["status"].value_counts().to_dict())

    ok = int((report["status"] == "ok").sum())
    print(f"coverage: {ok:,}/{len(rows):,} ({ok / max(len(rows), 1):.1%}) "
          "-- state this in the write-up; the missing songs are not random")
    if _stop.is_set():
        print("stopped early -- re-run the same command to continue")
        return 130

    if args.shard_size:
        print("\nsharding:")
        shard(args.out, args.shard_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
