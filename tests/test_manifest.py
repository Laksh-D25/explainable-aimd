"""Split integrity. These are the tests that stop a fabricated result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aimd.data.manifest import (
    SplitRatios,
    assert_no_duplicate_songs,
    assert_no_leakage,
    binary_label,
    split_by_song,
    summarize,
)

TAXONOMY = ["real", "full_fake", "half_fake", "mostly_fake"]


def fake_manifest(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "song_id": [f"s{i}" for i in range(n)],
            "path": [f"/data/{i}.mp3" for i in range(n)],
            "taxonomy": rng.choice(TAXONOMY, size=n, p=[0.5, 0.25, 0.15, 0.10]),
            "source": rng.choice(["suno", "udio"], size=n),
            "duration": rng.uniform(32, 240, size=n),
        }
    ).assign(label=lambda d: binary_label(d["taxonomy"]))


def test_split_is_leak_free():
    assert_no_leakage(split_by_song(fake_manifest()))


def test_leakage_is_detected_when_it_exists():
    """The guard must actually fire -- a green test suite on a leaky split is
    worse than no test."""
    df = split_by_song(fake_manifest())
    leaked = pd.concat([df, df.iloc[[0]].assign(split="test")], ignore_index=True)
    leaked.loc[leaked.index[-1], "split"] = "test"
    leaked.loc[leaked.index[0], "split"] = "train"
    with pytest.raises(ValueError, match="appear in multiple splits"):
        assert_no_leakage(leaked)


def test_every_song_lands_in_exactly_one_split():
    df = split_by_song(fake_manifest())
    assert df["split"].notna().all()
    assert set(df["split"].unique()) == {"train", "val", "test"}
    assert len(df) == len(fake_manifest())


def test_split_is_reproducible_and_seed_sensitive():
    a = split_by_song(fake_manifest(), seed=1337)["split"].tolist()
    b = split_by_song(fake_manifest(), seed=1337)["split"].tolist()
    c = split_by_song(fake_manifest(), seed=7)["split"].tolist()
    assert a == b and a != c


def test_ratios_are_approximately_honoured():
    df = split_by_song(fake_manifest(1000), SplitRatios(0.8, 0.1, 0.1))
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03
    assert abs(frac["val"] - 0.1) < 0.03
    assert abs(frac["test"] - 0.1) < 0.03


def test_stratification_keeps_rare_classes_in_every_split():
    """mostly_fake is the smallest class; if it vanishes from val, the auxiliary
    head's metrics become unreadable."""
    df = split_by_song(fake_manifest(1000))
    table = summarize(df)
    for taxon in TAXONOMY:
        assert (table[taxon] > 0).all(), f"{taxon} missing from a split:\n{table}"


def test_binary_label_collapses_all_fake_subtypes():
    tax = pd.Series(TAXONOMY)
    assert binary_label(tax).tolist() == [0, 1, 1, 1]


def test_duplicate_song_ids_are_rejected():
    df = fake_manifest(10)
    with pytest.raises(ValueError, match="duplicate song_id"):
        assert_no_duplicate_songs(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        SplitRatios(0.8, 0.2, 0.2)


def test_input_manifest_is_not_mutated():
    df = fake_manifest(50)
    split_by_song(df)
    assert "split" not in df.columns
