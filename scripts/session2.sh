#!/usr/bin/env bash
# Remaining work, in priority order, for the six hours available.
#
# 1. Suno <-> Udio transfer. The cleanest cross-generator test available: same
#    corpus, same pipeline, same real class, so the generator is the only thing
#    that changes. This is the base paper's protocol and the headline result.
# 2. Re-run the FakeMusicCaps evaluation with the fixed threshold. The previous
#    number was degenerate -- a saturated validation split pushed the operating
#    point against the real class, so everything was labelled fake.
# 3. Ablations, which justify each architectural choice rather than asserting it.
#
#   setsid bash scripts/session2.sh > ~/sonics/session2.log 2>&1 &

set -uo pipefail
PY=/home/laksh/Downloads/rmml/.venv/bin/python
S=/home/laksh/sonics
cd /home/laksh/Downloads/rmml/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NOISE='UserWarning|warnings.warn|torchaudio|StreamReader|torchcodec|^  s = |run_backward'
say() { echo; echo "######## $* ########"; date +%H:%M:%S; }

say "1/3 Suno <-> Udio cross-generator transfer"
$PY -u scripts/generator_split_eval.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/gensplit --epochs 12 --batch-size 2 2>&1 | grep -vE "$NOISE"

say "2/3 FakeMusicCaps re-evaluated with the corrected threshold"
$PY -u scripts/cross_generator_eval.py --generated $S/fmc/generated \
    --real $S/fmc/real_16k --checkpoint artifacts/run2/ckpt/best.pt \
    --sonics-manifest $S/manifest_big_cached.csv --out artifacts/run2_fixed 2>&1 | grep -vE "$NOISE"

say "3/3 ablations"
$PY -u scripts/run_experiment.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/abl --epochs 10 --batch-size 2 \
    --clip-seconds 5 --clips-per-song 4 --explain-songs 0 2>&1 | grep -vE "$NOISE"

say "ALL DONE"
