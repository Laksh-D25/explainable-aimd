#!/usr/bin/env bash
# Unattended end-to-end run: scale the data, retrain without the shortcut,
# evaluate in-distribution and cross-generator, and write every figure.
#
# Designed to be interrupted. Each stage checks whether its output already
# exists, so re-running continues rather than repeating. The machine hard
# powers off under sustained load, so scripts/thermal_guard.py should already
# be running with --wait; this script does not start it.
#
#   setsid bash scripts/overnight.sh > ~/sonics/overnight.log 2>&1 &

set -uo pipefail
PY=/home/laksh/Downloads/rmml/.venv/bin/python
S=/home/laksh/sonics
cd /home/laksh/Downloads/rmml/code

say() { echo; echo "######## $* ########"; date +%H:%M:%S; }

# ---------------------------------------------------------------- 1. wait for data
say "waiting for downloads (cap the wait so a stalled fetch cannot block the run)"
DEADLINE=$(( $(date +%s) + 4800 ))   # 80 minutes
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  R=$(ls $S/real_songs 2>/dev/null | grep -c '\.mp3$')
  F=$(ls $S/fake_songs 2>/dev/null | grep -c '\.mp3$')
  MC=$(ls $S/fmc/real 2>/dev/null | grep -c '\.wav$')
  FMC=$(find $S/fmc/generated -name '*.wav' 2>/dev/null | wc -l)
  echo "  real=$R fake=$F musiccaps=$MC fakemusiccaps=$FMC  $(date +%H:%M:%S)"
  # Gate only on the SONICS data the main experiment needs. The cross-generator
  # fetch runs last and keeps downloading in the background: FakeMusicCaps has
  # no range support, so reaching each successive generator means streaming
  # past every byte of the previous one, and blocking on it here would stall
  # the whole run for hours.
  [ "$R" -ge 700 ] && [ "$F" -ge 700 ] && break
  sleep 120
done

# ------------------------------------------------- 2. match bitrate on new real songs
say "matching bitrate on real songs (idempotent for already-matched files)"
$PY -u scripts/match_bitrate.py --audio-dir $S/real_songs \
    --match-csv $S/metadata/fake_songs.csv --workers 3 2>&1 | tail -3

# ------------------------------------------------------------- 3. manifest + cache
say "building balanced manifest"
$PY -u scripts/build_available_manifest.py --audio-root $S \
    --out $S/manifest_big.csv --balance 2>&1 | tail -12

say "caching decoded audio"
$PY -u scripts/build_clip_cache.py --manifest $S/manifest_big.csv \
    --out $S/cache_big --window 60 --workers 4 \
    --write-manifest $S/manifest_big_cached.csv 2>&1 | tail -8

# --------------------------------------------------------- 4. confirm shortcut gone
say "shortcut diagnostic (must be well below the detector's AUROC)"
$PY -u scripts/diagnose_shortcut.py --manifest $S/manifest_big_cached.csv \
    --n-per-class 150 --normalize 2>&1 | grep -vE "warn|torch|^  s ="

# ------------------------------------------------------------------- 5. train + eval
say "training + in-distribution + robustness + explanations + figures"
# batch 2, not 4: at batch 4 training held 3.0 GB of 3.75 GB and starved the
# desktop compositor badly enough to need `systemctl restart lightdm`. The VRAM
# cap in aimd.pipeline turns that freeze into an ordinary OOM, and batch 2 stays
# under it comfortably.
$PY -u scripts/run_experiment.py --manifest $S/manifest_big_cached.csv \
    --out artifacts/run2 --epochs 20 --batch-size 2 \
    --clip-seconds 5 --clips-per-song 4 --explain-songs 16 \
    2>&1 | grep -vE "UserWarning|warnings.warn|torchaudio|StreamReader|torchcodec|^  s = |run_backward"

# ------------------------------------------------------------ 6. cross-generator set
say "preparing cross-generator audio (band-limit real to match generated)"
# FakeMusicCaps is 16 kHz; the MusicCaps copy is 48 kHz. Without matching them
# the detector separates the classes on bandwidth alone and the headline number
# is meaningless.
mkdir -p $S/fmc/real_16k
for f in $S/fmc/real/*.wav; do
  b=$(basename "$f")
  [ -f "$S/fmc/real_16k/$b" ] && continue
  ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 -t 10 "$S/fmc/real_16k/$b" 2>/dev/null
done
echo "  band-limited: $(ls $S/fmc/real_16k 2>/dev/null | grep -c '\.wav$')"

say "cross-generator evaluation"
$PY -u scripts/cross_generator_eval.py --generated $S/fmc/generated \
    --real $S/fmc/real_16k --checkpoint artifacts/run2/ckpt/best.pt \
    --sonics-manifest $S/manifest_big_cached.csv --out artifacts/run2 \
    2>&1 | grep -vE "UserWarning|warnings.warn|torchaudio|StreamReader|torchcodec|^  s = "

say "done"
ls -la artifacts/run2/ 2>/dev/null | tail -20
