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




def test_empty_stratum_in_a_split_is_warned_about():
    """Rounding can empty a rare class out of test on a small manifest; a
    silent zero in a results table reads as a measurement."""
    from aimd.data.manifest import warn_on_empty_strata

    with pytest.warns(UserWarning, match="unrepresented"):
        df = split_by_song(fake_manifest(40))

    problems = warn_on_empty_strata(df)
    assert problems, "expected at least one absent class at this size"


def test_large_manifest_warns_about_nothing():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_by_song(fake_manifest(2000))


# --- FakeMusicCaps -----------------------------------------------------------


def _fmc_tree(tmp_path, generators=("AudioLDM2", "MusicGen"), n=3, with_real=False):
    root = tmp_path / "FakeMusicCaps"
    for gen in generators:
        (root / gen).mkdir(parents=True)
        for i in range(n):
            (root / gen / f"yt{i:03d}.wav").write_bytes(b"")
    real = None
    if with_real:
        real = tmp_path / "MusicCaps"
        real.mkdir()
        for i in range(n):
            (real / f"yt{i:03d}.wav").write_bytes(b"")
    return str(root), (str(real) if real else None)


def test_fakemusiccaps_uses_directory_names_as_attribution_labels(tmp_path):
    """The dataset's own loader takes the label from the parent directory."""
    from aimd.data.manifest import load_fakemusiccaps_manifest

    root, real = _fmc_tree(tmp_path, with_real=True)
    df = load_fakemusiccaps_manifest(root, real)
    assert set(df.loc[df["label"] == 1, "source"]) == {"AudioLDM2", "MusicGen"}
    assert (df.loc[df["label"] == 1, "taxonomy"] == "full_fake").all()
    assert (df.loc[df["label"] == 0, "taxonomy"] == "real").all()


def test_fakemusiccaps_ids_do_not_collide_across_generators(tmp_path):
    """The same MusicCaps id is regenerated by every model, so a bare stem
    would collide and silently halve the evaluation set."""
    from aimd.data.manifest import load_fakemusiccaps_manifest

    root, real = _fmc_tree(tmp_path, with_real=True)
    df = load_fakemusiccaps_manifest(root, real)
    assert not df["song_id"].duplicated().any()
    assert len(df) == 2 * 3 + 3  # two generators x 3 clips, plus 3 real


def test_fakemusiccaps_without_real_audio_warns(tmp_path):
    """The release contains only generated audio; detection metrics are
    undefined without a real class."""
    from aimd.data.manifest import load_fakemusiccaps_manifest

    root, _ = _fmc_tree(tmp_path)
    with pytest.warns(UserWarning, match="only generated audio"):
        df = load_fakemusiccaps_manifest(root)
    assert (df["label"] == 1).all()


def test_fakemusiccaps_has_no_split_column(tmp_path):
    """It is never trained on, so every row is the evaluation set."""
    from aimd.data.manifest import load_fakemusiccaps_manifest
    from aimd.eval.protocols import rows_for_split

    root, real = _fmc_tree(tmp_path, with_real=True)
    df = load_fakemusiccaps_manifest(root, real)
    assert "split" not in df.columns
    assert len(rows_for_split(df, "test")) == len(df)


def test_fakemusiccaps_generators_discovered_not_hardcoded(tmp_path):
    """A partial download or an added model must still work."""
    from aimd.data.manifest import load_fakemusiccaps_manifest

    root, real = _fmc_tree(tmp_path, generators=("MusicGen", "SomeNewTTM"), with_real=True)
    df = load_fakemusiccaps_manifest(root, real)
    assert set(df.loc[df["label"] == 1, "source"]) == {"MusicGen", "SomeNewTTM"}


def test_fakemusiccaps_missing_root_fails_loudly(tmp_path):
    from aimd.data.manifest import load_fakemusiccaps_manifest

    with pytest.raises(FileNotFoundError, match="root not found"):
        load_fakemusiccaps_manifest(str(tmp_path / "nope"))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no generator subdirectories"):
        load_fakemusiccaps_manifest(str(empty))


# --- SONICS split CSVs (real schema) -----------------------------------------


def _sonics_csv(tmp_path, name, rows):
    """Write a CSV with SONICS's real column names and semantics."""
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _sonics_rows():
    return {
        "train": [
            # `id` is the SOURCE track and is shared by a real song and the
            # tracks generated from it -- it is not a unique key.
            {"id": 1, "filename": "real_00001", "filepath": "real_songs/real_00001.mp3",
             "label": "real", "target": 0, "source": "youtube", "duration": 200.0},
            {"id": 2, "filename": "fake_2_suno_0", "filepath": "fake_songs/fake_2_suno_0.mp3",
             "label": "mostly fake", "target": 1, "source": "suno", "duration": 180.0},
            {"id": 2, "filename": "fake_2_suno_1", "filepath": "fake_songs/fake_2_suno_1.mp3",
             "label": "full fake", "target": 1, "source": "suno", "duration": 170.0},
        ],
        "val": [
            {"id": 3, "filename": "real_00003", "filepath": "real_songs/real_00003.mp3",
             "label": "real", "target": 0, "source": "youtube", "duration": 210.0},
            {"id": 4, "filename": "fake_4_udio_0", "filepath": "fake_songs/fake_4_udio_0.mp3",
             "label": "half fake", "target": 1, "source": "udio", "duration": 160.0},
        ],
        "test": [
            {"id": 5, "filename": "real_00005", "filepath": "real_songs/real_00005.mp3",
             "label": "real", "target": 0, "source": "youtube", "duration": 190.0},
        ],
    }


@pytest.fixture
def sonics_csvs(tmp_path):
    rows = _sonics_rows()
    return {k: _sonics_csv(tmp_path, k, v) for k, v in rows.items()}


def test_taxonomy_comes_from_label_not_target(sonics_csvs):
    """`target` is the binary 0/1 flag. Reading the taxonomy off it yields "1"
    for every generated track and destroys the auxiliary head's supervision."""
    from aimd.data.manifest import load_sonics_manifest

    m = load_sonics_manifest(sonics_csvs)
    assert set(m["taxonomy"]) == {"real", "mostly_fake", "full_fake", "half_fake"}
    assert set(m["label"]) == {0, 1}
    assert (m.loc[m["taxonomy"] == "real", "label"] == 0).all()
    assert (m.loc[m["taxonomy"] != "real", "label"] == 1).all()


def test_song_id_is_filename_because_id_is_not_unique(sonics_csvs):
    """Two generated variants share one source `id`; only `filename` is unique."""
    from aimd.data.manifest import load_sonics_manifest

    m = load_sonics_manifest(sonics_csvs)
    assert m["song_id"].is_unique
    assert set(m.loc[m["group"] == "2", "song_id"]) == {"fake_2_suno_0", "fake_2_suno_1"}


def test_splits_are_taken_from_the_files_not_regenerated(sonics_csvs):
    from aimd.data.manifest import load_sonics_manifest

    m = load_sonics_manifest(sonics_csvs)
    assert m.groupby("split").size().to_dict() == {"train": 3, "val": 2, "test": 1}
    assert_no_leakage(m)


def test_audio_root_is_prefixed_to_relative_paths(sonics_csvs):
    from aimd.data.manifest import load_sonics_manifest

    m = load_sonics_manifest(sonics_csvs, audio_root="/data/sonics")
    assert m["path"].iloc[0] == "/data/sonics/real_songs/real_00001.mp3"


def test_per_class_csvs_are_rejected_with_a_useful_message(tmp_path):
    """real_songs.csv and fake_songs.csv carry no filepath -- using them would
    yield a manifest of NaN paths that only fails at load time."""
    from aimd.data.manifest import load_sonics_manifest

    path = _sonics_csv(tmp_path, "real_songs",
                       [{"filename": "real_00001", "label": "real", "target": 0}])
    with pytest.raises(ValueError, match="missing.*filepath|carry no filepath"):
        load_sonics_manifest({"train": path})


def test_unexpected_taxonomy_value_fails_loudly(tmp_path):
    from aimd.data.manifest import load_sonics_manifest

    path = _sonics_csv(tmp_path, "train", [
        {"id": 1, "filename": "x", "filepath": "a.mp3", "label": "sort of fake", "target": 1}])
    with pytest.raises(ValueError, match="unexpected taxonomy"):
        load_sonics_manifest({"train": path})


def test_group_overlap_report_counts_sources_spanning_splits(tmp_path):
    """SONICS's own splits let a real song and its regenerations straddle
    splits; the figure is reported rather than silently repaired."""
    from aimd.data.manifest import group_overlap_report, load_sonics_manifest

    csvs = {
        "train": _sonics_csv(tmp_path, "train", [
            {"id": 7, "filename": "real_7", "filepath": "r.mp3", "label": "real", "target": 0}]),
        "test": _sonics_csv(tmp_path, "test", [
            {"id": 7, "filename": "fake_7_suno_0", "filepath": "f.mp3",
             "label": "full fake", "target": 1}]),
    }
    report = group_overlap_report(load_sonics_manifest(csvs))
    assert report["groups_spanning_splits"] == 1
    assert report["rows_affected"] == 2
    assert report["fraction_affected"] == pytest.approx(1.0)
