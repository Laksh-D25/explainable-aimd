# Fetching the real audio (Linux)

SONICS ships only generated songs; the real class has to be pulled from YouTube
using `real_songs.csv`, as the dataset's own README instructs. This is the
step-by-step for doing that on Linux with 1 TB of Drive and a small local disk.

Measured on this machine: **1.92 s/song** at 8 workers, so ~26 h for all 48,090,
~69 GB at 48 kbps, ~85% success (the rest removed, private, region-locked, or
shorter than the dataset's window).

## 0. Prerequisites

`yt-dlp` must be **current** — a 2024 build fails on every video against
YouTube's present player API, which looks exactly like an IP ban but is not:

```bash
cd /home/laksh/Downloads/rmml/code
../.venv/bin/pip install -U yt-dlp
```

The venv is at `/home/laksh/Downloads/rmml/.venv` — one level **above** `code/`,
so every command below uses `../.venv/bin/python`. The scripts pick that
interpreter's `yt-dlp` themselves and refuse to start on a pre-2025 build, so a
stale system copy earlier on PATH cannot silently break the run.

`rclone` is already installed at `~/.local/bin/rclone` (v1.75.1).

## 1. Start small

Do not commit 26 hours before knowing the pipeline trains. A 4,000-song subset
takes about two hours and is enough for a defensible balanced result:

```bash
../.venv/bin/python scripts/fetch_real_songs.py \
  --csv /path/to/real_songs.csv \
  --out ~/sonics/real_songs \
  --limit 4000 --workers 8 --shard-size 1500
```

Interrupt with Ctrl-C whenever you like. **Re-run the identical command** to
continue — completed songs and permanently dead videos are skipped, timeouts are
retried. `--limit` samples with a fixed seed, so a resumed subset is the same
subset.

Progress lives in `~/sonics/fetch_state.jsonl`, written per song. Coverage is
printed at the end; put that number in the write-up, because the missing songs
are not a random sample.

## 2. Connect Drive (once)

```bash
rclone config
```

Answer as follows — everything not listed can take its default:

| Prompt | Answer |
|---|---|
| `e/n/d/r/c/s/q>` | `n` (new remote) |
| `name>` | `gdrive` |
| `Storage>` | `drive` |
| `client_id>` / `client_secret>` | blank (press Enter) |
| `scope>` | `1` (full access) |
| `Edit advanced config?` | `n` |
| `Use web browser to automatically authenticate?` | `y` |
| `Configure this as a Shared Drive?` | `n` |
| `Keep this "gdrive" remote?` | `y` |

A browser opens for Google sign-in. Confirm with `rclone listremotes` — it
should print `gdrive:`.

> The blank client_id uses rclone's shared Google credentials, which are
> rate-limited at busy times. For a 69 GB upload it is worth creating your own
> OAuth client ID in Google Cloud Console; rclone's docs walk through it.

## 3. Upload and reclaim disk

With ~17 GB free locally and ~69 GB of audio, the loop is fetch → shard →
upload → delete → repeat:

```bash
../.venv/bin/python scripts/upload_to_drive.py \
  --local ~/sonics/real_songs_shards \
  --remote gdrive:sonics/real_songs_shards \
  --delete-after-upload
```

Shards already on Drive at the right size are skipped, so this is resumable too.
A shard is deleted locally **only** after its size on Drive matches — a partial
upload is re-sent rather than removed, since losing a shard costs hours of
re-downloading.

Repeat 1 and 3, raising `--limit`, until coverage is what you want.

## 4. Train

Drive means **Colab**, not Kaggle — Kaggle notebooks cannot mount Drive.

In Colab: mount Drive, copy *one shard at a time* to the VM's local disk, and
extract there. Do not read the loose mp3s over the Drive mount; tens of
thousands of small files over FUSE makes training I/O-bound, which is the entire
reason for sharding.

```python
from google.colab import drive; drive.mount('/content/drive')
!cp /content/drive/MyDrive/sonics/real_songs_shards/real_songs_000.tar /content/
!mkdir -p /content/real_songs && tar -xf /content/real_songs_000.tar -C /content/real_songs
```

## Before training on it

Run the bandwidth check. If the real class and the generated class differ in
spectral bandwidth, the detector separates them on the codec chain and scores
beautifully while measuring nothing:

```python
from aimd.data.audio import bandwidth_report
print(bandwidth_report(manifest))     # `suspicious: True` means stop and fix it
```

The fetcher already matches the generated set's bitrate (48 kbps) and honours
`skip_time`/`duration`, so this should pass — but check rather than assume.
