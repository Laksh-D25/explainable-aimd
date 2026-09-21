#!/usr/bin/env bash
# In-distribution evaluation, then cross-generator, from the trained checkpoint.
#
# Run as a file rather than an inline command: long inline invocations kept
# being killed before their redirect took effect, leaving no log and no process.
#
#   setsid bash scripts/finish_all.sh > ~/sonics/finish_all.log 2>&1 &

set -uo pipefail
PY=/home/laksh/Downloads/rmml/.venv/bin/python
S=/home/laksh/sonics
cd /home/laksh/Downloads/rmml/code

# The VRAM cap that stops the desktop freezing is tighter than the explanation
# pass wants; expandable segments avoids the fragmentation the OOM reported.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NOISE='UserWarning|warnings.warn|torchaudio|StreamReader|torchcodec|^  s = |run_backward'
say() { echo; echo "######## $* ########"; date +%H:%M:%S; }

say "in-distribution evaluation"
$PY -u scripts/evaluate_only.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/run2 --batch-size 1 --explain-songs 12 2>&1 | grep -vE "$NOISE"

say "cross-generator evaluation"
$PY -u scripts/cross_generator_eval.py --generated $S/fmc/generated \
    --real $S/fmc/real_16k --checkpoint artifacts/run2/ckpt/best.pt \
    --sonics-manifest $S/manifest_big_cached.csv --out artifacts/run2 2>&1 | grep -vE "$NOISE"

say "DONE"
ls artifacts/run2/ | tr '\n' ' '
