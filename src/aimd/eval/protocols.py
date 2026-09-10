"""The three evaluation protocols.

This module is where the project's central claim lives. The architecture is an
integration of known components; what does not yet exist in the literature is
the combination of cross-generator, robustness, calibration and
explanation-faithfulness evaluation applied to full-length music. So these
protocols are the contribution, and they are kept deliberately strict:

* **In-distribution** -- SONICS test split. A sanity number, comparable to
  SONICS's own ~0.97 F1. A large shortfall means a pipeline bug, not a finding.
* **Cross-generator** -- FakeMusicCaps, never trained on. This is the headline
  number, set against the base paper's collapse from F1 0.99 to 0.629. It uses
  the *binary* head, because the 4-way SONICS taxonomy does not exist in
  FakeMusicCaps and could not transfer.
* **Robustness** -- the held-out perturbation set, applied one condition at a
  time so each result is attributable to a single degradation.

Every protocol reports on the *calibrated* scale at a chosen operating
threshold. Reporting F1 at an arbitrary 0.5 cutoff hides the failure the
project exists to prevent: a confident, wrong accusation against a human artist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..calibrate.temperature import TemperatureScaler
from ..data.datasets import ClipSpec, SongClipsDataset, collate
from ..data.perturb import EVAL_PERTURBATIONS, Perturbation
from ..models.detector import Detector
from ..train.loop import predict
from .metrics import summary


def rows_for_split(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select a split's rows.

    A manifest with no `split` column is an eval-only dataset (FakeMusicCaps),
    where every row is the evaluation set. A manifest that *has* the column
    must be filtered -- passing the whole thing through would evaluate on
    training songs and inflate every number reported.
    """
    if "split" not in manifest.columns:
        return manifest
    rows = manifest[manifest["split"] == split]
    if rows.empty:
        raise ValueError(
            f"split {split!r} is empty; manifest has "
            f"{manifest['split'].value_counts().to_dict()}"
        )
    return rows


def build_loader(
    manifest: pd.DataFrame,
    spec: ClipSpec,
    split: str = "test",
    perturbation: Perturbation | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Deterministic evaluation loader (uniform clip offsets, no shuffling)."""
    dataset = SongClipsDataset(
        rows_for_split(manifest, split), spec, split=split, perturbation=perturbation
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )


@dataclass
class ProtocolResult:
    name: str
    metrics: dict[str, float]
    n_songs: int
    threshold: float
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"protocol": self.name, "n_songs": self.n_songs, **self.metrics}


def _evaluate(
    model: Detector,
    loader: DataLoader,
    name: str,
    scaler: TemperatureScaler | None,
    threshold: float,
    device: str,
    amp: bool,
) -> ProtocolResult:
    out = predict(model, loader, device=device, amp=amp)
    probs = scaler.transform(out["logit"]) if scaler is not None else out["prob"]
    return ProtocolResult(
        name=name,
        metrics=summary(out["label"], probs, threshold=threshold),
        n_songs=len(out["label"]),
        threshold=threshold,
    )


def evaluate_in_distribution(
    model: Detector,
    manifest: pd.DataFrame,
    spec: ClipSpec,
    scaler: TemperatureScaler | None = None,
    threshold: float = 0.5,
    split: str = "test",
    device: str = "cpu",
    amp: bool = False,
) -> ProtocolResult:
    """SONICS held-out test split."""
    loader = build_loader(manifest, spec, split=split)
    return _evaluate(model, loader, "in_distribution", scaler, threshold, device, amp)


def evaluate_cross_generator(
    model: Detector,
    manifest: pd.DataFrame,
    spec: ClipSpec,
    scaler: TemperatureScaler | None = None,
    threshold: float = 0.5,
    baseline_f1: float = 0.629,
    device: str = "cpu",
    amp: bool = False,
) -> ProtocolResult:
    """Unseen-generator evaluation -- the headline number.

    `baseline_f1` is the base paper's cross-generator F1, carried in the result
    so the comparison is explicit in the report rather than left to the reader.
    """
    loader = build_loader(manifest, spec, split="test")
    result = _evaluate(model, loader, "cross_generator", scaler, threshold, device, amp)
    result.extra["baseline_f1"] = baseline_f1
    result.extra["delta_vs_baseline"] = result.metrics["f1"] - baseline_f1
    return result


def evaluate_robustness(
    model: Detector,
    manifest: pd.DataFrame,
    spec: ClipSpec,
    scaler: TemperatureScaler | None = None,
    threshold: float = 0.5,
    conditions: list[Perturbation] | None = None,
    split: str = "test",
    device: str = "cpu",
    amp: bool = False,
) -> pd.DataFrame:
    """Per-condition degradation table.

    Conditions default to the held-out set, which by construction contains
    nothing seen during training -- see `aimd.data.perturb.assert_disjoint`.
    Applying them one at a time is what makes a row attributable to a single
    degradation; stacking them would report a number no reader could act on.
    """
    conditions = conditions if conditions is not None else EVAL_PERTURBATIONS

    clean = _evaluate(
        model, build_loader(manifest, spec, split=split), "clean", scaler, threshold, device, amp
    )
    rows = [{**clean.row(), "condition": "clean", "family": "-", "f1_drop": 0.0}]

    for condition in conditions:
        loader = build_loader(manifest, spec, split=split, perturbation=condition)
        result = _evaluate(model, loader, condition.name, scaler, threshold, device, amp)
        rows.append(
            {
                **result.row(),
                "condition": condition.name,
                "family": condition.family,
                # Positive = degradation, which is the direction a reader expects.
                "f1_drop": clean.metrics["f1"] - result.metrics["f1"],
            }
        )

    return pd.DataFrame(rows).set_index("condition")


def fit_calibration(
    model: Detector,
    manifest: pd.DataFrame,
    spec: ClipSpec,
    target_fpr: float = 0.01,
    device: str = "cpu",
    amp: bool = False,
) -> tuple[TemperatureScaler, float]:
    """Fit temperature and pick an operating threshold on the validation split.

    Both must come from validation, never from test: a threshold chosen on the
    test set reports the best case rather than the expected one.
    """
    from ..calibrate.temperature import threshold_for_target_fpr

    out = predict(model, build_loader(manifest, spec, split="val"), device=device, amp=amp)
    scaler = TemperatureScaler().fit(out["logit"], out["label"])
    threshold = threshold_for_target_fpr(
        out["logit"], out["label"], target_fpr=target_fpr, scaler=scaler
    )
    return scaler, threshold


def results_table(results: list[ProtocolResult]) -> pd.DataFrame:
    """Assemble protocol results into the report's headline table."""
    return pd.DataFrame([r.row() for r in results]).set_index("protocol")
