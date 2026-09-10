"""Dataset discipline: augmentation must never reach val/test, and a song's
clips must arrive shaped for the model's song-level pooling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from aimd.data.datasets import ClipSpec, SongClipsDataset, collate
from aimd.data.perturb import EVAL_PERTURBATIONS, TrainAugment

SR = 24_000
SPEC = ClipSpec(clip_seconds=1.0, clips_per_song=3, max_clips=4, sample_rate=SR)


@pytest.fixture
def manifest(tmp_path):
    """Three short songs of differing length written to disk as real audio."""
    rng = np.random.default_rng(0)
    rows = []
    for i, seconds in enumerate([5.0, 3.0, 0.5]):
        path = tmp_path / f"song{i}.wav"
        sf.write(path, rng.standard_normal(int(seconds * SR)) * 0.1, SR)
        rows.append(
            {
                "song_id": f"s{i}",
                "path": str(path),
                "label": i % 2,
                "taxonomy": "real" if i % 2 == 0 else "full_fake",
                "source": "suno",
                "duration": seconds,
            }
        )
    return pd.DataFrame(rows)


def test_item_shape_matches_the_model_contract(manifest):
    item = SongClipsDataset(manifest, SPEC, split="val")[0]
    assert item["clips"].shape == (SPEC.max_clips, int(SPEC.clip_seconds * SR))
    assert item["clip_mask"].shape == (SPEC.max_clips,)
    assert item["clip_mask"].sum() == SPEC.clips_per_song


def test_short_song_is_masked_not_dropped(manifest):
    """The 0.5 s song yields one real clip; the rest must be masked padding."""
    item = SongClipsDataset(manifest, SPEC, split="val")[2]
    assert item["clip_mask"].sum() == 1
    assert item["clips"][1:].abs().sum() == 0


def test_collate_produces_the_batched_shapes(manifest):
    ds = SongClipsDataset(manifest, SPEC, split="val")
    batch = collate([ds[i] for i in range(3)])
    assert batch["clips"].shape == (3, SPEC.max_clips, int(SPEC.clip_seconds * SR))
    assert batch["clip_mask"].shape == (3, SPEC.max_clips)
    assert batch["label"].shape == (3,)
    assert batch["song_id"] == ["s0", "s1", "s2"]


def test_augmenting_val_or_test_is_refused(manifest):
    for split in ["val", "test"]:
        with pytest.raises(ValueError, match="must not touch val/test"):
            SongClipsDataset(manifest, SPEC, split=split, augment=TrainAugment())


def test_augment_and_perturbation_cannot_be_stacked(manifest):
    with pytest.raises(ValueError, match="mutually exclusive"):
        SongClipsDataset(
            manifest,
            SPEC,
            split="train",
            augment=TrainAugment(),
            perturbation=EVAL_PERTURBATIONS[0],
        )


def test_eval_items_are_deterministic(manifest):
    """A checkpoint evaluated twice must produce identical inputs."""
    a = SongClipsDataset(manifest, SPEC, split="test")[0]["clips"]
    b = SongClipsDataset(manifest, SPEC, split="test")[0]["clips"]
    assert (a == b).all()


def test_train_items_vary_across_epochs_but_are_seedable(manifest):
    """Random offsets act as augmentation in their own right."""
    a = SongClipsDataset(manifest, SPEC, split="train", seed=1)[0]["clips"]
    b = SongClipsDataset(manifest, SPEC, split="train", seed=1)[0]["clips"]
    c = SongClipsDataset(manifest, SPEC, split="train", seed=2)[0]["clips"]
    assert (a == b).all()
    assert not (a == c).all()


def test_robustness_perturbation_changes_audio_on_eval_split(manifest):
    clean = SongClipsDataset(manifest, SPEC, split="test")[0]["clips"]
    degraded = SongClipsDataset(
        manifest, SPEC, split="test", perturbation=EVAL_PERTURBATIONS[0]
    )[0]["clips"]
    assert clean.shape == degraded.shape
    assert not (clean == degraded).all(), "perturbation had no effect"


def test_source_index_only_present_when_requested(manifest):
    assert "source" not in SongClipsDataset(manifest, SPEC, split="val")[0]
    item = SongClipsDataset(manifest, SPEC, split="val", sources=("suno", "udio"))[0]
    assert item["source"].item() == 0
