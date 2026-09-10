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

## GPU memory — RTX 3050 Laptop, 3.7 GB usable

Measured with `scripts/bench_vram.py`. Note the usable figure is 3.7 GB, not the
4 GB the spec sheet implies.

**Training** (backbone under `no_grad`), 10 s clips, 8 clips/song:

| Batch | Peak |
|---|---|
| 1 | 2.20 GB |
| 2 | 3.33 GB |
| 4 | OOM |

So `configs/train/local.yaml` is capped at batch 2. Kaggle's 16 GB T4 has roughly
4x the headroom; re-run the script there rather than extrapolating.

**Explanation** (gradients retained to the audio), batch 1:

| Clips x seconds | Total frames | Peak |
|---|---|---|
| 8 x 10 s | 6000 | OOM |
| 4 x 10 s | 3000 | OOM |
| 8 x 5 s | 3000 | OOM |
| 2 x 10 s | 1500 | 2.13 GB |
| 4 x 5 s | 1500 | 2.13 GB |
| 8 x 2 s | 1200 | 1.78 GB |
| 1 x 10 s | 750 | 1.26 GB |

The ceiling is **total frames (~1500)**, not clips or duration separately —
gradients are retained through all 12 transformer layers, so cost scales with
clips x seconds x 75 however that product is reached.

**This has a methodological consequence.** A model trained at 8 x 10 s cannot be
explained at 8 x 10 s on this GPU, so a local explanation necessarily covers a
smaller slice of the song than the prediction did. Either say so when reporting
the explanation, or resolve it one of two ways:

* run explanations on Kaggle at full clip settings, or
* add gradient checkpointing to the backbone, trading compute for memory so the
  full song fits locally.

Not yet decided; the tests do not depend on either.

## Planning note

Running the pipeline while other CPU-heavy jobs compete (a test suite, a second
notebook) slows it by more than an order of magnitude — a training run that
takes ~2 min per epoch alone took hours under contention. On Kaggle this is not
a concern; locally, run one thing at a time before concluding something is stuck.
