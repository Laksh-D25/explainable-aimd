# Architecture review — `Research Architechture - 2343031.pdf`

Verdict: **the core is sound and well-grounded in the cited literature, but the
diagram is not implementable as drawn.** Three blocking errors and seven design
or methodology issues. Each finding below is marked with how it was confirmed.

Measurements come from `tests/test_mert_integration.py` against the real
`m-a-p/MERT-v1-95M` checkpoint, so they are reproducible rather than asserted:

| Property | Measured |
|---|---|
| Hidden states | 13 (12 layers + embedding output) |
| Hidden dim | 768 |
| Sample rate | 24 kHz, `do_normalize=True` |
| Frame rate | **74.8 Hz** (5 s → 374 frames) |
| Conv stack output | `HubertFeatureEncoder`, **512 channels × time** |
| Parameters | 94.4 M frozen / 0.62 M trainable |
| Gradient reaches audio | yes, with `backbone_no_grad=False` |

## Sound — kept as drawn

Frozen MERT backbone with a light trainable head; aggregation across all 13
hidden states (Zhang et al. SLS; El Kheir et al.); attention pooling over frames
rather than mean pooling (Guo et al.); the SONICS → FakeMusicCaps cross-generator
protocol; robustness evaluation under MP3 and pitch shift; attention weights as a
free Level-1 explanation; SHAP over 13 layer contributions (cheap — 13 "features").

## Blocking

**B1 — RVQ-VAE and CQT teachers are drawn on the inference path; they do not
exist at inference.** They are MERT's *pre-training* masked-prediction targets.
*Confirmed:* the loaded checkpoint is a `HubertFeatureEncoder` plus a 12-layer
transformer, with no teacher modules. Nothing in that sub-box is executable.
→ Relabel as pre-training provenance. Documented in `models/backbone.py`.

**B2 — No clip→song aggregation, but the task is song-level.** SONICS songs run
32–240 s; MERT was pre-trained on 5 s and its attention is O(T²).
*Confirmed:* at the measured 74.8 Hz, a 120 s song is ~8,976 frames, whose
attention alone is ~1.9 GB per layer in fp16 — over budget on a 16 GB T4. The
diagram segments into clips (correct) then classifies (incorrect), leaving
per-clip outputs while SONICS's ~0.97 F1 and the base paper's numbers are
per-song, so neither comparison is valid.
→ Added `song_pool` in `models/detector.py`, with masking for variable clip counts.

**B3 — Grad-CAM cannot give spectrogram localisation here.** MERT consumes raw
waveform; no time–frequency map exists in the forward graph.
*Confirmed:* `conv_module` emits **512 channels × time**, not frequency bins.
Grad-CAM there localises *when*, not *at which frequency*.
→ Level 2 splits into **2a** temporal relevance (fast, every example) and **2b**
STFT-occlusion for true time–frequency evidence (slow, audit subset only).

## Design and methodology

**D1 — Four aggregation stages, three redundant.** "Temporal" and "Frame"
attention are the same axis (MERT frames *are* time frames); "Feature Fusion"
had a single input. → Collapsed to layer (13→1) then time (T→1), plus B2's song
stage. The dropped blocks survive as ablations, not deletions.

**D2 — "Layer-wise Feature Alignment" was undefined.** All 13 states are already
`[T, 768]`, so nothing needs geometric alignment — but layer norms differ enough
that the layer-attention softmax would track magnitude instead of
informativeness. → Defined as per-layer LayerNorm (+ optional projection);
`test_layernorm_stack_normalises_each_layer_separately` pins it.

**D3 — Calibration was an adjective, not a module.** The literature review makes
it one of four contributions, yet only the output box said "Calibrated".
→ Explicit temperature scaling + ECE + FPR at the operating threshold.

**D4 — Augmentation and robustness evaluation were conflated.** Training on MP3
and pitch shift then "testing robustness" on MP3 and pitch shift measures
memorisation. → `data/perturb.py` partitions the conditions and
`assert_disjoint()` enforces it, by unseen family or non-overlapping range.

**D5 — XAI levels 1 and 3 measure nearly the same thing.** → Framed as a
faithfulness check: L1 is what the model *claims* to weight, L3 is what actually
moved a given logit. Disagreement is a finding.

**D6 — "Source & Label Prediction" had no defined label source.** SONICS labels
Suno vs Udio (2 classes); FakeMusicCaps has 5 generators. → Scoped to
FakeMusicCaps closed-set; `SourceHead` is off unless `n_sources` is set.

**D7 — Frozen ≠ no gradients.** Wrapping the backbone in a permanent `no_grad`
is the obvious way to freeze it and silently breaks every gradient-based
explanation. → Freezing is `requires_grad_(False)`; `no_grad` only on the
training fast path. `tests/test_freezing_and_gradients.py` holds the line.

## On the novelty claim

"Attention pooling over a music foundation model" is true but narrow, and
integration-shaped rather than methodological. The more defensible contribution
is the **evaluation protocol** — cross-generator + robustness + calibration +
explanation faithfulness on full-length music — which genuinely does not exist.
Recommend leading with that and with the ablation table. Position explicitly
against Li et al. 2025 (*Explainable detection of machine-generated music*,
Sci. Rep.), the closest competing work.

## Addendum — verified in implementation

Building the corrected design surfaced two things the review predicted only in
principle:

**D7 has a second, sharper form.** Freezing the backbone with
`requires_grad_(False)` on *every* parameter means nothing inside the conv stack
creates an autograd graph at all. Gradient-based attribution therefore fails
with `can't retain_grad on Tensor that has requires_grad=False` unless the
**input waveform** is marked as requiring grad. Any gradient-based explanation
over a fully frozen backbone has to enter through the input.
Handled in `xai/relevance.py`; pinned by `test_gradients_reach_the_audio_when_requested`.

**Explanations must not contaminate training.** The backward pass used for
attribution leaves gradients on the trainable head. Running an explanation
mid-epoch would fold them into the next optimiser step. `temporal_relevance`
now clears gradients on exit.

**A saturated model cannot be explained by gradients.** On the synthetic smoke
task the head reaches zero loss in one epoch; gradients vanish, grad x
activation relevance becomes noise, and the faithfulness gap reports "no better
than chance". This is expected rather than a defect, but it means faithfulness
and L1-vs-L3 agreement are only interpretable on a model trained on the real
task — `scripts/smoke_train.py` now says so in its output rather than letting
the number be misread.
