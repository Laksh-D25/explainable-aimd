"""The robustness protocol depends on train/eval perturbations staying disjoint."""

from __future__ import annotations

import numpy as np
import pytest

from aimd.data.perturb import (
    EVAL_PERTURBATIONS,
    TRAIN_PERTURBATIONS,
    Perturbation,
    TrainAugment,
    assert_disjoint,
)


def test_shipped_sets_are_disjoint():
    assert_disjoint()


def test_overlapping_range_in_a_shared_family_is_rejected():
    train = [Perturbation("mp3_train", "codec", lambda sr: None, {"bitrate": (64, 192)})]
    evaluation = [Perturbation("mp3_eval", "codec", lambda sr: None, {"bitrate": (128, 128)})]
    with pytest.raises(ValueError, match="overlaps training augmentation"):
        assert_disjoint(train, evaluation)


def test_unseen_family_is_disjoint_by_construction():
    train = [Perturbation("mp3_train", "codec", lambda sr: None, {"bitrate": (64, 192)})]
    evaluation = [Perturbation("reverb", "room", lambda sr: None, {})]
    assert_disjoint(train, evaluation)  # must not raise


def test_every_eval_family_or_range_is_actually_unseen():
    """Guards the intent, not just the mechanism: each eval condition must be
    either a family absent from training or a non-overlapping range."""
    train_families = {p.family for p in TRAIN_PERTURBATIONS}
    for ev in EVAL_PERTURBATIONS:
        if ev.family in train_families:
            assert ev.ranges, f"{ev.name} shares a training family but declares no ranges"


def test_train_augment_is_reproducible_and_bounded():
    wav = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    a = TrainAugment(probability=1.0, seed=7)(wav, 24_000)
    b = TrainAugment(probability=1.0, seed=7)(wav, 24_000)
    np.testing.assert_allclose(a, b)
    assert a.shape == wav.shape, "clip length must survive augmentation"


def test_train_augment_off_is_identity():
    wav = np.random.default_rng(0).standard_normal(4096).astype(np.float32)
    np.testing.assert_allclose(TrainAugment(probability=0.0, seed=0)(wav, 24_000), wav)


def test_all_shipped_conditions_are_runnable():
    """Every declared condition must actually execute in this environment.

    The MP3 backend needs `fast_mp3_augment` and reverb needs
    `pyroomacoustics`; both are required, not optional, because MP3 is the
    base paper's headline robustness condition. A silently-skipped condition
    would show up as a suspiciously flat robustness table.
    """
    from aimd.data.perturb import available

    status = {**available(TRAIN_PERTURBATIONS), **available(EVAL_PERTURBATIONS)}
    missing = [name for name, ok in status.items() if not ok]
    assert not missing, f"perturbation backends unavailable: {missing}"


def test_clips_must_exceed_the_codec_minimum_length():
    """The MP3 backend rejects inputs under 100 ms. Real clips are 5-10 s, but
    this pins the constraint so a future short-clip config fails loudly here
    rather than silently disabling MP3 augmentation at training time."""
    from aimd.data.perturb import available

    assert available(TRAIN_PERTURBATIONS, seconds=0.05)["mp3_128"] is False
    assert available(TRAIN_PERTURBATIONS, seconds=1.0)["mp3_128"] is True
