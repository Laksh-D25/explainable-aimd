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

#: SONICS 4-way taxonomy, underscored. Index 0 is the only non-fake class.
TAXONOMY_ORDER = ("real", "full_fake", "half_fake", "mostly_fake")


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
    warn_on_empty_strata(out, stratify_on)
    return out


def apply_official_splits(
    manifest: pd.DataFrame,
    split_files: dict[str, str],
    id_column: str = "id",
) -> pd.DataFrame:
    """Adopt a dataset's own published splits instead of generating new ones.

    SONICS ships train.csv / valid.csv / test.csv. Using those rather than a
    fresh random split is what makes the in-distribution number comparable to
    SONICS's published ~0.97 F1 -- a different partition measures a different
    problem, however carefully it is stratified.

    Args:
        split_files: mapping of split name -> CSV path, e.g.
            {"train": ".../train.csv", "val": ".../valid.csv", "test": ".../test.csv"}
        id_column: the column in those CSVs holding the song id.
    """
    assert_no_duplicate_songs(manifest)
    out = manifest.copy()
    out["split"] = pd.NA

    for split, path in split_files.items():
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        ids = set(pd.read_csv(path)[id_column].astype(str))
        out.loc[out["song_id"].astype(str).isin(ids), "split"] = split

    unassigned = int(out["split"].isna().sum())
    if unassigned:
        raise ValueError(
            f"{unassigned} songs are in no official split. Either the id column "
            f"({id_column!r}) is wrong or the manifest and split files disagree; "
            "silently dropping them would change the evaluation set."
        )

    out["split"] = out["split"].astype("string")
    assert_no_leakage(out)
    warn_on_empty_strata(out)
    return out


def warn_on_empty_strata(df: pd.DataFrame, stratify_on: str = "taxonomy") -> list[str]:
    """Warn when a class is absent from a split.

    Rounding can empty a rare class out of val or test on a small manifest --
    at which point metrics for that class are computed over nothing and read as
    though they were measured. Loud at build time beats a silent zero in a
    results table. Returns the problems found, for tests to assert on.
    """
    import warnings

    problems = []
    for split in SPLITS:
        rows = df[df["split"] == split]
        if rows.empty:
            problems.append(f"split {split!r} is empty")
            continue
        for value in df[stratify_on].dropna().unique():
            if not (rows[stratify_on] == value).any():
                problems.append(f"{value!r} absent from split {split!r}")
    if problems:
        warnings.warn(
            "split leaves some classes unrepresented: "
            + "; ".join(problems)
            + ". Metrics for those classes would be computed over no data.",
            stacklevel=3,
        )
    return problems


def binary_label(taxonomy: pd.Series) -> pd.Series:
    """Collapse the 4-way taxonomy to real(0) / fake(1).

    This is the primary target: it is what compares to the base paper's
    F1 0.99 -> 0.629 and the only label that transfers to FakeMusicCaps.
    """
    return (taxonomy != "real").astype(int)


def load_sonics_manifest(
    split_csvs: dict[str, str] | None = None,
    audio_root: str | None = None,
    *,
    train_csv: str | None = None,
    valid_csv: str | None = None,
    test_csv: str | None = None,
) -> pd.DataFrame:
    """Build a manifest from SONICS's split CSVs.

    Read the **split** CSVs (train/valid/test.csv), not real_songs.csv and
    fake_songs.csv. The split files are the only complete manifests: they carry
    `filepath` and `split`, which the two per-class files do not, and they
    already merge real and generated rows.

    Column semantics, which are easy to get backwards:

    * ``label``  -- the 4-way taxonomy (``real``/``full fake``/``half fake``/
      ``mostly fake``), normalised here to underscores.
    * ``target`` -- the binary 0/1 flag. Reading the taxonomy off ``target``
      yields "1" for every generated track and silently destroys the auxiliary
      head's supervision.
    * ``filename`` -- unique per track, so it is the song id.
    * ``id`` -- the *source* track id, and **not unique**: several generated
      variants share one id (``fake_53858_suno_0`` and ``..._1``). It is kept
      as ``group`` so a custom split can avoid putting variants of the same
      source on both sides.

    Args:
        split_csvs: mapping of split name -> CSV path, e.g.
            ``{"train": ".../train.csv", "val": ".../valid.csv",
               "test": ".../test.csv"}``. SONICS names its file ``valid.csv``
            while this project uses ``val`` internally.
        audio_root: prefix joined to each relative ``filepath``.
    """
    from pathlib import Path

    if split_csvs is None:
        pairs = {"train": train_csv, "val": valid_csv, "test": test_csv}
        split_csvs = {k: v for k, v in pairs.items() if v}
    if not split_csvs:
        raise ValueError("pass split_csvs, e.g. {'train': 'train.csv', ...}")

    frames = []
    for split, path in split_csvs.items():
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        raw = pd.read_csv(path, low_memory=False)
        missing = {"filename", "filepath", "label", "target"} - set(raw.columns)
        if missing:
            raise ValueError(
                f"{path} is missing {sorted(missing)}. Use SONICS's train/valid/"
                "test.csv -- real_songs.csv and fake_songs.csv carry no filepath."
            )
        frames.append(
            pd.DataFrame(
                {
                    "song_id": raw["filename"].astype(str),
                    "path": raw["filepath"].astype(str),
                    "taxonomy": raw["label"].astype(str).str.strip().str.replace(" ", "_"),
                    "source": raw.get("source", pd.NA),
                    "duration": raw.get("duration", pd.NA),
                    "label": raw["target"].astype(int),
                    "group": raw.get("id", raw["filename"]).astype(str),
                    "split": split,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    if audio_root:
        df["path"] = df["path"].apply(lambda p: str(Path(audio_root) / p))

    unknown = set(df["taxonomy"]) - set(TAXONOMY_ORDER)
    if unknown:
        raise ValueError(f"unexpected taxonomy values {sorted(unknown)}")

    df["split"] = df["split"].astype("string")
    assert_no_duplicate_songs(df)
    assert_no_leakage(df)
    return df[[*COLUMNS, "group", "split"]]


#: The five text-to-music generators in FakeMusicCaps. Directory names are the
#: attribution class labels -- the dataset's own loader uses `path.split("/")[-2]`.
FAKEMUSICCAPS_GENERATORS = (
    "AudioLDM2",
    "MusicGen",
    "MusicLDM",
    "Mustango",
    "StableAudioOpen",
)


def load_fakemusiccaps_manifest(
    root: str,
    real_root: str | None = None,
    extension: str = "wav",
) -> pd.DataFrame:
    """Build an eval-only manifest from a FakeMusicCaps directory tree.

    Layout (one directory per generator, which is also the attribution label)::

        root/AudioLDM2/<ytid>.wav
        root/MusicGen/<ytid>.wav
        ...

    Generators are discovered from the subdirectories rather than hard-coded,
    so a partial download or an added model still works.

    **`real_root` is not optional in practice.** The FakeMusicCaps release
    contains only generated audio -- the real recordings are a separate
    MusicCaps download. A manifest with no real class cannot produce a
    detection F1 at all, and pairing these clips with real audio from a
    *different* corpus (SONICS, say) measures the domain gap rather than the
    generator: FakeMusicCaps is 10 s of 16 kHz mono regenerated from MusicCaps
    captions, while SONICS is full-length YouTube audio. A detector separating
    those is reading clip length and bandwidth, not synthesis artefacts, and
    the resulting cross-generator number would be meaningless.

    Args:
        root: directory holding one subdirectory per generator.
        real_root: directory of real reference clips, processed the same way
            (MusicCaps originals for the captions these were generated from).
        extension: audio file extension to collect.

    Returns:
        A manifest with the canonical columns and no `split` column -- this
        dataset is never trained on, so every row is the evaluation set.
    """
    import warnings
    from pathlib import Path

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"FakeMusicCaps root not found: {root}")

    rows = []
    generators = sorted(d.name for d in root_path.iterdir() if d.is_dir())
    if not generators:
        raise FileNotFoundError(f"no generator subdirectories under {root}")

    for generator in generators:
        for path in sorted((root_path / generator).glob(f"*.{extension}")):
            rows.append(
                {
                    # The same MusicCaps id is regenerated by every model, so the
                    # generator has to be part of the id or they collide.
                    "song_id": f"{generator}__{path.stem}",
                    "path": str(path),
                    "taxonomy": "full_fake",
                    "source": generator,
                    "duration": pd.NA,
                }
            )

    if real_root:
        real_path = Path(real_root)
        if not real_path.is_dir():
            raise FileNotFoundError(f"real_root not found: {real_root}")
        for path in sorted(real_path.rglob(f"*.{extension}")):
            rows.append(
                {
                    "song_id": f"real__{path.stem}",
                    "path": str(path),
                    "taxonomy": "real",
                    "source": pd.NA,
                    "duration": pd.NA,
                }
            )
    else:
        warnings.warn(
            "no real_root given, so this manifest contains only generated audio. "
            "Detection metrics (F1, AUROC, EER) are undefined without a real "
            "class, and substituting real songs from another corpus would "
            "measure the domain gap rather than the generator. Only closed-set "
            "attribution is meaningful from this manifest.",
            stacklevel=2,
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no .{extension} files found under {root}")
    df["label"] = binary_label(df["taxonomy"])
    assert_no_duplicate_songs(df)
    return df[list(COLUMNS)]


def group_overlap_report(df: pd.DataFrame) -> dict:
    """Quantify source tracks whose rows land in more than one split.

    SONICS's ``id`` links a real song to the AI regenerations derived from it,
    so one id can cover ``real_10003`` and ``fake_10003_suno_0`` alike. In the
    official splits roughly 13% of rows belong to an id that spans splits --
    a real song can sit in test while a track generated from its lyrics and
    style sits in validation.

    This is **not** silently repaired. Re-splitting would break comparability
    with SONICS's published F1, which is the whole reason to adopt their
    partition. Report the figure alongside the in-distribution result and let
    the reader weigh it; use ``split_by_song(..., group_column="group")`` if a
    stricter secondary analysis is wanted.
    """
    if "group" not in df.columns:
        raise ValueError("manifest has no 'group' column; load it with load_sonics_manifest")
    spans = df.groupby("group")["split"].nunique()
    offending = set(spans[spans > 1].index)
    rows = df[df["group"].isin(offending)]
    return {
        "groups_spanning_splits": len(offending),
        "groups_total": int(spans.size),
        "rows_affected": len(rows),
        "rows_total": len(df),
        "fraction_affected": len(rows) / max(len(df), 1),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts by taxonomy -- the table to eyeball before training."""
    return (
        df.groupby(["split", "taxonomy"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(list(SPLITS))
    )
