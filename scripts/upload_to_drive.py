"""Upload dataset shards to Google Drive with rclone, resumably.

Written for the disk squeeze: the real-song fetch produces ~69 GB while this
machine has ~17 GB free. So the loop is fetch -> shard -> upload -> delete
local -> repeat, and each stage has to survive being interrupted.

Why shards rather than loose files: reading tens of thousands of small files
over a mounted drive makes training I/O-bound. A few 1.5 GB tars copy to a
notebook VM's local disk in one go and extract there.

`--delete-after-upload` only ever removes a shard whose size on Drive matches
the local file. A partial upload is re-uploaded rather than deleted, because
losing a shard means re-downloading hours of audio.

Setup (once):

    rclone config
      n) New remote  ->  name: gdrive  ->  storage: drive
      client_id/secret: blank is fine    ->  scope: 1 (full access)
      Edit advanced config? n            ->  Use web browser to auto authenticate? y
      Configure this as a Shared Drive?  n

Then:

    python scripts/upload_to_drive.py --local ~/sonics/real_songs_shards \
        --remote gdrive:sonics/real_songs_shards --delete-after-upload
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RCLONE = "rclone"


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_remote(remote: str) -> None:
    """Fail early with a usable message rather than mid-upload."""
    name = remote.split(":")[0]
    listed = run([RCLONE, "listremotes"], timeout=60)
    if listed.returncode != 0:
        raise SystemExit(f"rclone not usable: {listed.stderr.strip()}")
    if f"{name}:" not in listed.stdout:
        raise SystemExit(
            f"remote {name!r} is not configured. Run `rclone config`, choose "
            f"'n' for new remote, name it {name!r}, pick 'drive', and authorise "
            "in the browser. See this script's docstring."
        )


def remote_sizes(remote: str) -> dict[str, int]:
    """Sizes of files already on the remote, so finished shards are skipped."""
    proc = run([RCLONE, "lsjson", remote], timeout=300)
    if proc.returncode != 0:
        return {}  # directory does not exist yet
    try:
        return {e["Name"]: e["Size"] for e in json.loads(proc.stdout or "[]")}
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, required=True, help="directory of shards")
    ap.add_argument("--remote", required=True, help="e.g. gdrive:sonics/real_songs_shards")
    ap.add_argument("--pattern", default="*.tar")
    ap.add_argument("--delete-after-upload", action="store_true",
                    help="free local disk once a shard is verified on Drive")
    ap.add_argument("--transfers", type=int, default=4)
    args = ap.parse_args()

    check_remote(args.remote)
    shards = sorted(args.local.glob(args.pattern))
    if not shards:
        raise SystemExit(f"no {args.pattern} files in {args.local}")

    already = remote_sizes(args.remote)
    todo = [s for s in shards if already.get(s.name) != s.stat().st_size]
    print(f"{len(shards)} shards | {len(shards) - len(todo)} already on Drive | "
          f"{len(todo)} to upload")

    uploaded = failed = freed = 0
    for i, shard in enumerate(todo, 1):
        size_gb = shard.stat().st_size / 1024**3
        print(f"[{i}/{len(todo)}] {shard.name} ({size_gb:.2f} GB)...", end=" ", flush=True)
        start = time.time()
        proc = run([RCLONE, "copy", str(shard), args.remote,
                    "--transfers", str(args.transfers), "--retries", "3"])
        if proc.returncode != 0:
            print(f"FAILED: {proc.stderr.strip()[:110]}")
            failed += 1
            continue

        # Verify by size before trusting it; a partial upload must not cause a
        # local delete, since losing a shard costs hours of re-downloading.
        remote_size = remote_sizes(args.remote).get(shard.name)
        if remote_size != shard.stat().st_size:
            print(f"SIZE MISMATCH (remote {remote_size}), kept locally")
            failed += 1
            continue

        elapsed = time.time() - start
        print(f"ok ({size_gb / max(elapsed, 1e-9) * 60:.1f} GB/min)", end="")
        uploaded += 1
        if args.delete_after_upload:
            shard.unlink()
            freed += size_gb
            print(f"  freed {size_gb:.2f} GB", end="")
        print()

    print(f"\nuploaded {uploaded}, failed {failed}"
          + (f", freed {freed:.1f} GB locally" if args.delete_after_upload else ""))
    if failed:
        print("re-run the same command to retry the failures")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
