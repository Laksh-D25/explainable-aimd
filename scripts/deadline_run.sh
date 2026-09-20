#!/usr/bin/env bash
# Everything that must be finished by 08:00, in priority order, with no waiting.
#
# Real-song downloads are rate-limited to ~0.2/s with a 6-hour ETA, so the run
# uses what is already on disk rather than blocking on them. Downloads continue
# in the background and feed the cross-generator stage, which runs last.
#
# Ordered so that an overrun loses the least important thing: the main
# experiment and its figures complete first, ablations last.
#
#   setsid bash scripts/deadline_run.sh > ~/sonics/deadline.log 2>&1 &

set -uo pipefail
PY=/home/laksh/Downloads/rmml/.venv/bin/python
S=/home/laksh/sonics
cd /home/laksh/Downloads/rmml/code

say() { echo; echo "######## $* ########"; date +%H:%M:%S; }
NOISE='UserWarning|warnings.warn|torchaudio|StreamReader|torchcodec|^  s = |run_backward'

say "1/7 match bitrate on any newly fetched real songs"
$PY -u scripts/match_bitrate.py --audio-dir $S/real_songs \
    --match-csv $S/metadata/fake_songs.csv --workers 3 2>&1 | tail -2

say "2/7 balanced manifest from everything on disk"
$PY -u scripts/build_available_manifest.py --audio-root $S \
    --out $S/manifest_big.csv --balance 2>&1 | tail -10

say "3/7 cache decoded audio"
$PY -u scripts/build_clip_cache.py --manifest $S/manifest_big.csv \
    --out $S/cache_big --window 60 --workers 4 \
    --write-manifest $S/manifest_big_cached.csv 2>&1 | tail -6

say "4/7 shortcut diagnostic"
$PY -u scripts/diagnose_shortcut.py --manifest $S/manifest_big_cached.csv \
    --n-per-class 150 --normalize 2>&1 | grep -vE "$NOISE"

say "5/7 train + evaluate + robustness + explanations + figures"
$PY -u scripts/run_experiment.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/run2 --epochs 16 --batch-size 2 \
    --clip-seconds 5 --clips-per-song 4 --explain-songs 16 --skip-ablations \
    2>&1 | grep -vE "$NOISE"

say "6/7 cross-generator evaluation"
mkdir -p $S/fmc/real_16k
# FakeMusicCaps is 16 kHz mono; the MusicCaps copy is 48 kHz stereo. Without
# matching them the detector separates the classes on bandwidth alone.
for f in $S/fmc/real/*.wav; do
  b=$(basename "$f")
  [ -f "$S/fmc/real_16k/$b" ] && continue
  ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 -t 10 "$S/fmc/real_16k/$b" 2>/dev/null
done
echo "  real clips band-limited: $(ls $S/fmc/real_16k 2>/dev/null | grep -c '\.wav$')"
echo "  generators available: $(ls $S/fmc/generated 2>/dev/null | tr '\n' ' ')"
$PY -u scripts/cross_generator_eval.py --generated $S/fmc/generated \
    --real $S/fmc/real_16k --checkpoint artifacts/run2/ckpt/best.pt \
    --sonics-manifest $S/manifest_big_cached.csv --out artifacts/run2 \
    2>&1 | grep -vE "$NOISE"

say "7/7 ablations (lowest priority; safe to be cut short)"
$PY -u scripts/run_experiment.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/run2_abl --epochs 10 --batch-size 2 \
    --clip-seconds 5 --clips-per-song 4 --explain-songs 0 \
    2>&1 | grep -vE "$NOISE" | grep -E "ablation|F1|===" | tail -20

say "ALL DONE"
ls artifacts/run2/ 2>/dev/null | tr '\n' ' '
