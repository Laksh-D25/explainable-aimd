# Measured costs

All figures from the development laptop (12-core CPU, torch 2.8.0+cpu,
MERT-v1-95M). They exist to make Kaggle's ~30 h/week quota plannable, since
that quota — not correctness — is what limits the full run and the ablation
sweep.

## Backbone

| Input | Time | Per clip |
|---|---|---|
| 1 × 0.5 s | 0.21 s | 0.21 s |
| 16 × 0.5 s | 1.37 s | 0.086 s |
| 1 × 10 s | 2.00 s | 2.00 s |

Batching matters: per-clip cost drops ~2.5× from batch 1 to 16. Load time for
the checkpoint is ~12 s (cached) or ~35 s (first download).

## Perturbations, per 10 s clip

| Condition | Time | |
|---|---|---|
| `gain_mild` | 0.000 s | |
| `lowpass_4k` | 0.007 s | |
| `noise_low_snr` | 0.012 s | |
| `time_stretch` | 0.107 s | |
| `pitch_large` | 0.123 s | |
| `mp3_32` | 0.037 s | |
| **`reverb`** | **0.516 s** | dominates the robustness pass |
| **`mp3_128`** | **0.629 s** | dominates training augmentation |

Robustness applies 11 conditions (clean + 10) over the test split, so reverb and
MP3 set the floor on that pass. Both are load-bearing — MP3 is the base paper's
headline robustness condition — so the cost is accepted rather than trimmed.

## Padded clips

Songs differ in length, so a batch contains padding. Padded clips get zero
weight at the song pool, so `Detector.forward` skips them in the backbone:

| Batch (69% real clips) | Time |
|---|---|
| all clips through backbone | 2.65 s |
| padding skipped | 1.63 s |

A 38% saving, tracking the real-clip fraction exactly. Two tests pin this:
one asserts only real clips reach the backbone, the other that the optimisation
is *exactly* equivalent rather than approximately.

Relatedly, `max_clips` now defaults to `clips_per_song`. A larger value only
pads, and before this fix every padded clip cost a full MERT forward.

## Planning note

Running the pipeline while other CPU-heavy jobs compete (a test suite, a second
notebook) slows it by more than an order of magnitude — a training run that
takes ~2 min per epoch alone took hours under contention. On Kaggle this is not
a concern; locally, run one thing at a time before concluding something is stuck.
