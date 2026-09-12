"""Reclassify stale `blocked` entries in a fetch state file.

An earlier version of the fetcher lumped YouTube's rate-limiting in with
genuine region blocks under one `blocked` status, and treated it as permanent.
Resuming such a run would skip those songs forever -- in the observed case 1,799
of 4,000, or 45% of the fetch, discarded for a block that expires in hours.

This rewrites those entries so they are retried. Anything that really is
region-blocked will be re-detected as `geo_blocked` on the next attempt and
then correctly skipped, so being generous here costs one retry and nothing else.

    python scripts/repair_fetch_state.py --state ~/sonics/fetch_state.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--status", default="blocked", help="status to reclassify")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.state.exists():
        raise SystemExit(f"no state file at {args.state}")

    records, malformed = [], 0
    for line in args.state.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1

    before = Counter(r.get("status") for r in records)
    affected = [r for r in records if r.get("status") == args.status]
    print(f"{len(records):,} entries" + (f" ({malformed} malformed, dropped)" if malformed else ""))
    print("current:", dict(before.most_common()))
    print(f"\n{len(affected):,} entries with status {args.status!r} would be retried")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0
    if not affected:
        print("nothing to repair")
        return 0

    backup = args.state.with_suffix(".jsonl.bak")
    shutil.copy2(args.state, backup)

    # Dropping the entries entirely is what makes them retryable: the fetcher
    # skips only what it has a recorded verdict for.
    kept = [r for r in records if r.get("status") != args.status]
    with args.state.open("w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nbacked up to {backup.name}")
    print(f"removed {len(affected):,} entries; {len(kept):,} remain")
    print("re-run the fetch command to retry them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
