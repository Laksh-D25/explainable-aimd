# Explainable AI-Generated Music Detection (MERT + Attention Pooling)

Frozen MERT backbone → layer attention → temporal attention pooling → clip→song
aggregation → calibrated real/fake decision, with a three-level explanation layer.

Targets the four gaps named in the literature review: cross-generator
generalisation (base paper: F1 0.99 → **0.629**), robustness to routine audio
transformations, explainability, and calibration.

**Read [`docs/architecture-review.md`](docs/architecture-review.md) first** — the
architecture diagram has three blocking errors, and the code implements the
corrected version with the as-drawn variants kept as ablations.

## Pipeline

```
audio → 24 kHz mono → fixed clips (10 s)
      → [train split only] augmentation, seen conditions
      → frozen MERT-v1-95M, all 13 hidden states     [B, 13, T, 768]
      → per-layer LayerNorm            (D2)
      → layer attention   (13 → 1)                   [B, T, 768]
      → temporal attention pool (T → 1)              [B, 768]  per clip
      → clip→song attention    (B2)                  [B, 768]  per song
      → binary head (primary) + auxiliary 4-way SONICS taxonomy head
      → temperature scaling    (D3)                  calibrated probability
      → XAI L1 attention · L2a temporal relevance / L2b T-F occlusion · L3 SHAP-over-layers
```

94.4 M frozen parameters, 0.62 M trainable.

## Where things run

Local (RTX 3050 4 GB, 14 GB RAM) is for **development and smoke tests only**.
SONICS is 97k songs / 4,751 hours and will not fit here, so reportable numbers
come from **Kaggle**, where `awsaf49/sonics-dataset` attaches without consuming
local disk (T4 16 GB, ~30 h/week — the binding constraint on the full run and
the ablation sweep).

Local torch: a CUDA build is fine now that disk has been freed (the GPU is an
RTX 3050 Laptop, 4 GB, compute 8.6). Kaggle images already ship their own CUDA
torch — do not reinstall it there. The 4 GB VRAM ceiling, not disk, is what
caps local batch size; run `scripts/bench_vram.py` to size it by measurement
rather than guesswork.

Feature caching is deliberately **not** used: 13 layers × 750 frames × 768 dims
in fp16 is ~15 MB per 10 s clip, so caching SONICS would need multiple TB. The
frozen backbone runs on the fly instead.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio  # local only
pip install -r requirements-dev.txt
pytest                 # 35 fast tests, no downloads
pytest -m real_mert    # 7 integration tests against the real checkpoint (~400 MB)
```

## Layout

| Path | What |
|---|---|
| `src/aimd/models/` | backbone, pooling stages, heads, assembled detector |
| `src/aimd/data/` | manifests, audio I/O, datasets, perturbations |
| `src/aimd/xai/` | the three explanation levels + faithfulness metrics |
| `src/aimd/calibrate/` | temperature scaling, ECE |
| `src/aimd/eval/` | in-distribution, cross-generator, robustness protocols |
| `configs/experiment/` | the ablation table ([README](configs/experiment/README.md)) |
| `src/aimd/eval/report.py` | the four write-up figures, each with its numbers as CSV |
| `scripts/fetch_real_songs.py` | resumable YouTube fetch for SONICS's real class |
| `scripts/upload_to_drive.py` | resumable rclone upload, frees local disk as it goes |

## Usage

```bash
pip install -e .

python -m aimd.cli info                 # environment + perturbation backends
python -m aimd.cli prepare-data --real-csv real_songs.csv --fake-csv fake_songs.csv
python -m aimd.cli train    --manifest manifest.csv --out artifacts/run
python -m aimd.cli evaluate --manifest manifest.csv --checkpoint artifacts/run/best.pt
python -m aimd.cli explain  --manifest manifest.csv --checkpoint artifacts/run/best.pt --song-id s0042
```

Reportable numbers come from `notebooks/kaggle_train.ipynb`, which drives the
same `aimd.pipeline` functions as the CLI — a notebook that diverges from the
repo is how irreproducible results happen.

## Quick check

```bash
python scripts/smoke_train.py --stub    # whole pipeline on synthetic audio, seconds
python scripts/smoke_train.py           # same, with the real MERT backbone
python scripts/bench_vram.py            # measure batch sizes that fit (run on Kaggle too)
pytest                                  # 141 tests, no downloads
```

`smoke_train.py` plants an obvious artefact in the fake class and runs the real
pipeline over it — manifest, split, dataset, MERT, pooling, heads, calibration,
explanation, faithfulness. If it cannot separate a deliberately easy signal, the
data path is broken and Kaggle quota would be wasted.

## Status

- [x] **M0** scaffold, pinned env, MERT verified end-to-end
- [x] **M1** manifests + song-ID splits + leakage guards + perturbation partition
- [x] **M2** pipeline learns end-to-end on synthetic data, real MERT included
- [x] **XAI** all three levels + faithfulness harness
- [x] **Protocols** in-distribution / cross-generator / robustness, CLI, Kaggle notebook (141 tests)
- [ ] **M2b** overfit 100 real songs — needs real audio first, see
      [docs/fetching-real-audio.md](docs/fetching-real-audio.md)
- [ ] **M3** full SONICS training, in-distribution F1 vs SONICS's ~0.97
- [ ] **M5** FakeMusicCaps cross-generator eval — loader ready; needs MusicCaps real audio
- [ ] **M6** robustness on held-out conditions
- [ ] **M8** ablation table
- [x] **M9a** report figures (`aimd.eval.report`) — calibration, robustness, layer profile, faithfulness
- [x] **M9b** corrected architecture diagram ([docs/architecture.md](docs/architecture.md))

### Splits

SONICS publishes its own `train/valid/test.csv`. Those are adopted verbatim via
`apply_official_splits` — a freshly generated partition measures a different
problem and would break comparability with SONICS's published F1. The fallback
`split_by_song` is leak-free and stratified, and warns when rounding leaves a
class unrepresented in a split.
