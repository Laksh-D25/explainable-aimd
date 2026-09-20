"""Build a manifest from the audio that is actually on disk.

Neither SONICS nor FakeMusicCaps ships a complete corpus locally: real songs
are fetched from YouTube with ~85% coverage, and the generated set arrives in
3 GB parts. So a manifest has to be built from what is present, not from what
the CSVs describe, and it has to say clearly what it is working with.

It also balances the classes. With a few hundred real songs against thousands of
generated ones, an unbalanced split would let a detector score well by leaning
on the prior, and the F1 would say more about the class ratio than the model.

    python scripts/build_available_manifest.py --audio-root ~/sonics \
        --out ~/sonics/manifest.csv --balance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=Path.home() / "sonics/metadata")
    ap.add_argument("--audio-root", type=Path, required=True,
                    help="directory holding real_songs/ and fake_songs/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--balance", action="store_true",
                    help="subsample the larger class so the split is 50/50")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    sys_path = Path(__file__).resolve().parents[1] / "src"
    import sys
    sys.path.insert(0, str(sys_path))
    from aimd.data.manifest import SPLITS, assert_no_leakage, summarize

    frames = []
    for split, name in [("train", "train.csv"), ("val", "valid.csv"), ("test", "test.csv")]:
        raw = pd.read_csv(args.metadata / name, low_memory=False)
        frames.append(
            pd.DataFrame({
                "song_id": raw["filename"].astype(str),
                "path": raw["filepath"].astype(str),
                "taxonomy": raw["label"].astype(str).str.strip().str.replace(" ", "_"),
                "source": raw.get("source", pd.NA),
                "duration": raw.get("duration", pd.NA),
                "label": raw["target"].astype(int),
                "group": raw.get("id", raw["filename"]).astype(str),
                "split": split,
            })
        )
    catalogue = pd.concat(frames, ignore_index=True)
    print(f"catalogue: {len(catalogue):,} songs described by the CSVs")

    # Index what is actually on disk, by basename, so layout differences between
    # the fetch output and the zip's internal structure do not matter.
    present = {p.stem: str(p) for p in args.audio_root.rglob("*.mp3")}
    print(f"on disk  : {len(present):,} audio files under {args.audio_root}")

    catalogue["path"] = catalogue["song_id"].map(present)
    have = catalogue.dropna(subset=["path"]).copy()
    print(f"usable   : {len(have):,} songs with both metadata and audio")
    if have.empty:
        raise SystemExit("no overlap between metadata and audio -- check --audio-root")

    counts = have["label"].value_counts().to_dict()
    print(f"  real={counts.get(0, 0):,}  fake={counts.get(1, 0):,}")

    if args.balance:
        n = min(counts.get(0, 0), counts.get(1, 0))
        if n == 0:
            raise SystemExit("one class has no audio; cannot balance or train")
        have = pd.concat(
            [g.sample(n=n, random_state=args.seed) for _, g in have.groupby("label")],
            ignore_index=True,
        )
        print(f"balanced : {n:,} per class ({len(have):,} total)")

    # Splits are inherited from SONICS, but a subsample can empty one. Re-split
    # only when that happens, and say so, because it breaks comparability.
    # A subsample of the official splits can leave a split too small, or with one
    # class barely present -- calibration and model selection then run on noise.
    # Re-splitting costs comparability with SONICS's published F1, but that was
    # already lost by subsampling, whereas a 14-song validation set is simply
    # unusable.
    per_split = have["split"].value_counts()
    per_class = have.groupby(["split", "label"]).size().unstack(fill_value=0)
    too_small = len(per_split) < 3 or per_split.min() < 30
    too_skewed = per_class.min().min() < 10 if not per_class.empty else True
    if too_small or too_skewed:
        print("  NOTE: official splits left a split too small after subsampling; "
              "re-splitting by song. In-distribution numbers are no longer "
              "directly comparable to the SONICS paper.")
        from aimd.data.manifest import SplitRatios, split_by_song
        have = split_by_song(have.drop(columns=["split"]), SplitRatios(0.7, 0.15, 0.15),
                             seed=args.seed)

    assert_no_leakage(have)
    have.to_csv(args.out, index=False)
    print()
    print(summarize(have).to_string())
    print(f"\nwrote {len(have):,} songs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
