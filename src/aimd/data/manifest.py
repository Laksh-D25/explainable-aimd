"""Song-level manifests and splits.

The split is the part of the data pipeline most able to invent a result. Two
rules are enforced rather than documented:

1. **Split by song, never by clip.** A song yields 8-12 clips; if clips from one
   song land in both train and test, the model can recognise the song instead of
   the generator and the test score becomes meaningless. This is the same
   confound SingFake and CtrSVDD warn about, arriving through the split rather
   than through pre-processing.
2. **Stratify by taxonomy.** SONICS's fake classes are unbalanced, so an
   unstratified split can leave a rare class nearly absent from validation and
   make the auxiliary head's metrics unreadable.

`assert_no_leakage` is called by the split function itself, so producing a
leaky manifest requires deliberately bypassing the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")

#: Canonical manifest columns. `taxonomy` is the SONICS 4-way label, `label` its
#: binary collapse (the primary target, and the only one that transfers to
#: FakeMusicCaps). `source` is the generator, populated for attribution.
COLUMNS = ("song_id", "path", "label", "taxonomy", "source", "duration")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {total}")


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any song appears in more than one split."""
    if "split" not in df.columns:
        raise ValueError("manifest has no 'split' column")
    counts = df.groupby("song_id")["split"].nunique()
    offenders = counts[counts > 1]
    if len(offenders):
        raise ValueError(
            f"{len(offenders)} song(s) appear in multiple splits, e.g. "
            f"{list(offenders.index[:5])}. Clips of one song must never straddle "
            "splits -- the model would recognise the song rather than the generator."
        )


def assert_no_duplicate_songs(df: pd.DataFrame) -> None:
    """Raise if a song_id appears twice (duplicate rows inflate a split silently)."""
    dupes = df["song_id"].duplicated()
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate song_id rows in manifest")


def split_by_song(
    df: pd.DataFrame,
    ratios: SplitRatios | None = None,
    seed: int = 1337,
    stratify_on: str = "taxonomy",
) -> pd.DataFrame:
    """Assign a `split` column, stratified by `stratify_on`, grouped by song.

    Returns a copy; the input is left untouched.
    """
    ratios = ratios or SplitRatios()
    assert_no_duplicate_songs(df)
    out = df.copy()
    out["split"] = pd.NA
    rng = np.random.default_rng(seed)

    # Split within each stratum so class proportions hold in every split, even
    # for the rare fake subtypes.
    for _, group in out.groupby(stratify_on, dropna=False):
        # permutation, not in-place shuffle: Index.to_numpy() is read-only.
        idx = rng.permutation(group.index.to_numpy())
        n = len(idx)
        n_train = int(round(n * ratios.train))
        n_val = int(round(n * ratios.val))
        # Give the remainder to test rather than rounding it away.
        out.loc[idx[:n_train], "split"] = "train"
        out.loc[idx[n_train : n_train + n_val], "split"] = "val"
        out.loc[idx[n_train + n_val :], "split"] = "test"

    out["split"] = out["split"].astype("string")
    assert_no_leakage(out)
    return out


def binary_label(taxonomy: pd.Series) -> pd.Series:
    """Collapse the 4-way taxonomy to real(0) / fake(1).

    This is the primary target: it is what compares to the base paper's
    F1 0.99 -> 0.629 and the only label that transfers to FakeMusicCaps.
    """
    return (taxonomy != "real").astype(int)


def load_sonics_manifest(real_csv: str, fake_csv: str) -> pd.DataFrame:
    """Build a canonical manifest from SONICS's two CSVs."""
    real = pd.read_csv(real_csv)
    fake = pd.read_csv(fake_csv)

    real_out = pd.DataFrame(
        {
            "song_id": real.get("id", real.index).astype(str),
            "path": real.get("filepath", pd.NA),
            "taxonomy": "real",
            "source": pd.NA,
            "duration": real.get("duration", pd.NA),
        }
    )
    # SONICS spells the subtypes with spaces ("full fake"); normalise to the
    # underscored form used by TAXONOMY_CLASSES.
    fake_out = pd.DataFrame(
        {
            "song_id": fake.get("id", fake.index).astype(str),
            "path": fake.get("filepath", pd.NA),
            "taxonomy": fake["target"].astype(str).str.strip().str.replace(" ", "_"),
            "source": fake.get("source", pd.NA),
            "duration": fake.get("duration", pd.NA),
        }
    )

    df = pd.concat([real_out, fake_out], ignore_index=True)
    df["label"] = binary_label(df["taxonomy"])
    return df[list(COLUMNS)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts by taxonomy -- the table to eyeball before training."""
    return (
        df.groupby(["split", "taxonomy"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(list(SPLITS))
    )
