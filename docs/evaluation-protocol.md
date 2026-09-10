# Evaluation protocol

The architecture is an integration of known components. What does not yet exist
in the literature — and what this project's contribution rests on — is the
combination of cross-generator, robustness, calibration and explanation-faithfulness
evaluation applied to full-length music. So the protocol is treated as the
result, not the plumbing.

## Rules the code enforces

| Rule | Why | Enforced by |
|---|---|---|
| Split by song, never by clip | A song yields 8–12 clips; clips of one song across splits let the model recognise the *song*, not the generator | `assert_no_leakage`, called inside both split functions |
| Adopt SONICS's published splits | A different partition measures a different problem and breaks comparison with their ~0.97 F1 | `apply_official_splits` |
| Augmentation on train only | Augmenting val/test leaks the training distribution into the generalisation measurement | `SongClipsDataset` raises |
| Robustness conditions never seen in training | Training on MP3 then "testing robustness" on MP3 measures memorisation | `assert_disjoint`, by family or non-overlapping range |
| One condition at a time | A stacked degradation produces a number attributable to nothing | `evaluate_robustness` |
| Threshold and temperature from validation | A threshold chosen on test reports the best case, not the expected one | `fit_calibration` |
| Evaluate only the requested split | Passing a whole manifest to an eval loader silently includes training songs | `rows_for_split` |

## The three protocols

**In-distribution** — SONICS test split. A sanity number against SONICS's ~0.97 F1.
A large shortfall is a pipeline bug, not a finding.

**Cross-generator** — FakeMusicCaps, never trained on. The headline result,
reported against the base paper's F1 **0.629**. Uses the *binary* head: the 4-way
SONICS taxonomy has no counterpart in FakeMusicCaps and cannot transfer.
`ProtocolResult.extra` carries the baseline so the report cannot omit what the
number is compared against.

**Robustness** — the held-out perturbation set, one condition per row, reported
as F1 drop from clean.

## Calibration

Temperature scaling on validation, then an operating threshold chosen against a
**1% false-positive budget** rather than left at 0.5. The asymmetry is the point:
wrongly flagging a human artist's track is not the same cost as missing an
AI-generated one. Temperature scaling is monotonic, so F1/AUROC/EER are unchanged
— it only makes the probability mean what it says.

## Explanation faithfulness

Every explanation is scored before it is trusted: mask the regions it calls
important and compare the logit movement against masking random regions. The
number reported is the **gap against chance**, not the raw deletion curve —
masking any audio degrades a detector somewhat.

Two caveats the code states rather than hides:

- **A saturated model cannot be explained by gradients.** If training loss
  collapses to ~0, gradients vanish and relevance becomes noise.
- **Levels 1 and 3 measure different things.** Attention weights are what the
  model *allocates*; Shapley values are what *moved this decision*. Disagreement
  is a faithfulness finding worth reporting, not a bug to tune away.
