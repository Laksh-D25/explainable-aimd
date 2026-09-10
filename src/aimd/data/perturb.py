"""Audio perturbations, split into a train-seen set and a held-out eval set.

The architecture diagram merged augmentation straight into the data stream with
no train/eval separation. That is a measurement problem rather than a coding
one: if the model trains on MP3 and pitch shift and the "robustness evaluation"
then applies MP3 and pitch shift, the result reports how well augmentation was
memorised, not whether the detector is robust. The base paper's robustness gap
would be invisible.

So perturbations are partitioned here, and the partition is enforced:

* TRAIN_PERTURBATIONS  -- sampled during training, train split only.
* EVAL_PERTURBATIONS   -- never seen in training. Applied one at a time to build
  the per-condition robustness table.

Disjointness is by *family* (a transformation the model never saw at all, e.g.
time-stretch or reverb) or, within a shared family, by non-overlapping parameter
ranges (e.g. trained on 64-192 kbps MP3, evaluated at 32 kbps). `assert_disjoint`
holds that line so a later "just add one more augmentation" cannot quietly
invalidate the robustness numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    """One named degradation condition."""

    name: str
    family: str
    build: Callable[[int], Callable]
    #: Numeric ranges keyed by parameter, used for the disjointness check and
    #: reproduced in the robustness table.
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __call__(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.build(sample_rate)(samples=wav, sample_rate=sample_rate)


def _mp3(min_rate: int, max_rate: int) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import Mp3Compression

        return Mp3Compression(min_bitrate=min_rate, max_bitrate=max_rate, p=1.0)

    return build


def _pitch(min_st: float, max_st: float) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import PitchShift

        return PitchShift(min_semitones=min_st, max_semitones=max_st, p=1.0)

    return build


def _noise(min_snr: float, max_snr: float) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import AddGaussianSNR

        return AddGaussianSNR(min_snr_db=min_snr, max_snr_db=max_snr, p=1.0)

    return build


def _gain(min_db: float, max_db: float) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import Gain

        return Gain(min_gain_db=min_db, max_gain_db=max_db, p=1.0)

    return build


def _time_stretch(min_rate: float, max_rate: float) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import TimeStretch

        return TimeStretch(min_rate=min_rate, max_rate=max_rate, leave_length_unchanged=True, p=1.0)

    return build


def _lowpass(min_hz: float, max_hz: float) -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import LowPassFilter

        return LowPassFilter(min_cutoff_freq=min_hz, max_cutoff_freq=max_hz, p=1.0)

    return build


def _reverb() -> Callable[[int], Callable]:
    def build(sample_rate: int):
        from audiomentations import RoomSimulator

        return RoomSimulator(p=1.0)

    return build


# --- Seen during training -----------------------------------------------------
# Deliberately the transformations a distribution pipeline actually applies, at
# the parameter ranges the literature reports as damaging (Afchar et al.; Shi et al.).
TRAIN_PERTURBATIONS: list[Perturbation] = [
    Perturbation("mp3_128", "codec", _mp3(64, 192), {"bitrate": (64, 192)}),
    Perturbation("pitch_small", "pitch", _pitch(-2, 2), {"semitones": (-2, 2)}),
    Perturbation("noise_high_snr", "noise", _noise(20, 40), {"snr_db": (20, 40)}),
    Perturbation("gain_mild", "gain", _gain(-6, 6), {"gain_db": (-6, 6)}),
]

# --- Held out for evaluation only ---------------------------------------------
# Either an unseen family (time/room/filter) or an unseen range within a seen
# family (32 kbps MP3, +-4 semitones, 5-15 dB SNR).
EVAL_PERTURBATIONS: list[Perturbation] = [
    Perturbation("mp3_32", "codec", _mp3(32, 32), {"bitrate": (32, 32)}),
    Perturbation("pitch_large", "pitch", _pitch(-4, -3), {"semitones": (-4, -3)}),
    Perturbation("noise_low_snr", "noise", _noise(5, 15), {"snr_db": (5, 15)}),
    Perturbation("time_stretch", "time", _time_stretch(0.9, 1.1), {"rate": (0.9, 1.1)}),
    Perturbation("lowpass_4k", "filter", _lowpass(3000, 5000), {"cutoff_hz": (3000, 5000)}),
    Perturbation("reverb", "room", _reverb(), {}),
]


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def assert_disjoint(
    train: list[Perturbation] | None = None, evaluation: list[Perturbation] | None = None
) -> None:
    """Raise if any eval condition is reachable from training.

    Within a shared family the parameter ranges must not overlap; a family that
    never appears in training is disjoint by construction.
    """
    train = TRAIN_PERTURBATIONS if train is None else train
    evaluation = EVAL_PERTURBATIONS if evaluation is None else evaluation

    by_family: dict[str, list[Perturbation]] = {}
    for p in train:
        by_family.setdefault(p.family, []).append(p)

    for ev in evaluation:
        for tr in by_family.get(ev.family, []):
            for key, ev_range in ev.ranges.items():
                tr_range = tr.ranges.get(key)
                if tr_range is not None and _overlaps(ev_range, tr_range):
                    raise ValueError(
                        f"eval condition {ev.name!r} overlaps training augmentation "
                        f"{tr.name!r} on {key}: {ev_range} vs {tr_range}. Robustness "
                        "numbers from an overlapping condition measure memorisation, "
                        "not generalisation."
                    )


class TrainAugment:
    """Samples at most one training perturbation per clip.

    One at a time rather than a stacked chain: chaining makes it impossible to
    attribute a robustness result to any single condition, and stacked
    degradations quickly leave the distribution the detector must actually serve.
    """

    def __init__(self, probability: float = 0.5, seed: int | None = None):
        self.probability = probability
        self.perturbations = TRAIN_PERTURBATIONS
        self.rng = np.random.default_rng(seed)

    def __call__(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.rng.random() >= self.probability:
            return wav
        choice = self.perturbations[self.rng.integers(len(self.perturbations))]
        # audiomentations samples its own parameters from the `random` and
        # numpy global generators, so seeding only self.rng would leave runs
        # irreproducible. Drive the globals from self.rng and restore them, so
        # augmentation is deterministic without perturbing caller RNG state.
        seed = int(self.rng.integers(2**31 - 1))
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            return np.asarray(choice(wav, sample_rate), dtype=np.float32)
        except Exception:
            # A missing optional codec backend must not take down training;
            # aimd.data.perturb.available() reports what is actually usable.
            return wav
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def available(perturbations: list[Perturbation], seconds: float = 1.0) -> dict[str, bool]:
    """Which conditions can actually run here (codecs need extra backends).

    The probe must be a realistic clip length: the MP3 backend rejects inputs
    under 100 ms, so a short buffer reports a working codec as missing.
    """
    rng = np.random.default_rng(0)
    probe = rng.standard_normal(int(seconds * 24_000)).astype(np.float32) * 0.1
    status = {}
    for p in perturbations:
        try:
            p(probe, 24_000)
            status[p.name] = True
        except Exception:
            status[p.name] = False
    return status
