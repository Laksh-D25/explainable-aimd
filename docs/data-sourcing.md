# Data sourcing — verified availability

Checked against the actual releases, not the papers. **Neither dataset ships the
real audio**, which is the binding constraint on the whole evaluation.

## SONICS — real audio is not distributed

The authors state it plainly in their README:

> **Note:** This dataset contains **only fake songs**. For real songs, use the
> `youtube_id` from `real_songs.csv` to manually download them and place them
> inside `/dataset/real_songs/` folder.

`youtube_id` is documented as "YouTube ID of real song (**not provided as mp3**)".
Both mirrors are the same fake-only content — HuggingFace `awsaf49/sonics` is
30.0 GiB of `fake_songs/part_*.zip` plus 0.43 GiB of CSVs; the Kaggle mirror
`awsaf49/sonics-dataset` is 30.55 GiB. The arithmetic matches; there is no real
audio in either.

What sourcing it costs:

| | |
|---|---|
| Real songs to fetch | **48,090** (train 32,686 / val 2,167 / test 13,237) |
| Total audio | 2,781 hours, mean 208 s |
| Storage at the shipped encoding (~37 kbps) | **~42 GB** |
| Download time | tens of hours at best; YouTube rate-limits aggressively |
| Attrition | some fraction now deleted, private or region-locked |

Kaggle is a poor place to do this — datacenter IPs are throttled or blocked, and
the 12 h session cap makes a long fetch fragile.

### Fetching it — measured, not estimated

`yt-dlp` works from a home connection; the failures seen initially were a stale
2024 build against YouTube's current player API, not IP blocking. With the
current version:

| | |
|---|---|
| Throughput, 8 workers | **1.92 s/song** |
| Full 48,090 | **~26 hours** wall clock |
| Size at 48 kbps | ~69 GB |
| Success rate | ~85% — the rest removed, private, region-locked or shorter than the dataset's window |

`scripts/fetch_real_songs.py` does this **resumably**, which matters over a
26-hour run:

* every attempt appends to `fetch_state.jsonl` *as it finishes*, so an interrupt
  loses at most the in-flight downloads;
* on restart, completed songs and *permanent* failures (removed, private,
  blocked) are skipped, while *transient* ones (timeout, network) are retried —
  re-attempting dead videos every run would waste hours;
* completion is verified against the audio, not file existence, because an
  interrupted download leaves a large-but-truncated file that would otherwise be
  accepted and silently train on short clips;
* `--limit` samples with a fixed seed, so a resumed subset is the *same* subset.

It also matches the generated set's bitrate (48 kbps) and honours `skip_time` +
`duration`, so the real class does not become separable on codec or clip length.

### Google Drive (1 TB) as the store

Drive solves storage, not the fetch — the download still has to happen
somewhere, and a home connection is more reliable than a datacenter IP for
YouTube.

A workflow that fits a 17 GB local disk:

1. Fetch in batches locally with `--limit`, resuming as needed.
2. `--shard-size 1500` packs finished songs into ~1.5 GB tar shards. **Do not
   sync 48,000 loose files to Drive** — reading many small files over a mounted
   drive makes training I/O-bound; a few large shards do not.
3. Upload shards to Drive, delete locally, repeat.
4. Train on **Colab**, which mounts Drive natively: copy one shard to the VM's
   local disk, extract, train. Kaggle cannot mount Drive, so choosing Drive
   means choosing Colab.

**Practical route: a balanced subset.** Fetch a few thousand real songs, take an
equal number of generated ones, and keep the official split proportions. State
the subset size in the write-up. This gives a defensible in-distribution number
without a multi-day fetch, at the cost of not reproducing SONICS's headline
figure exactly.

## FakeMusicCaps — solvable

The Zenodo release (12 GB) is generated audio only: 27,605 clips of 10 s,
16 kHz mono, five TTM models, one directory per model. The real class is
MusicCaps, and published audio copies exist:

| Source | Clips | Size | Access |
|---|---|---|---|
| `Sienna5/MusicCaps_with_wav` | 5,346 | 9.3 GB (wav only) | **gated** — request access |
| `kelvincai/MusicCaps_30s_wav` | 1,710 | 8.7 GB | open |
| `CLAPv2/MusicCaps` | — | 9.2 GB parquet | open (untested here) |

`Sienna5` gives near-full coverage of MusicCaps' 5,521 clips and is worth
requesting. `kelvincai` works today without access requests.

### The bandwidth trap

`kelvincai/MusicCaps_30s_wav` is **30 s at 48 kHz stereo**; FakeMusicCaps
generated audio is **10 s at 16 kHz mono**. Resample both to MERT's 24 kHz and
the generated clips carry nothing above 8 kHz while the real ones reach 12 kHz.

A detector separating those is reading the codec chain, not the generator — and
the failure is invisible in the metrics, because it produces an *excellent*
score. It would sit at the centre of the cross-generator claim.

**Band-limit the real audio to 16 kHz before resampling**, so both classes
traverse the same chain. `aimd.data.audio.bandwidth_report` measures median
spectral rolloff per class and flags a ratio above 1.15; run it on any manifest
before training on it.

## Summary

| Milestone | Status |
|---|---|
| M2b / M3 (SONICS) | **blocked on real audio** — 48,090 YouTube fetches, or a subset |
| M5 (cross-generator) | viable — FakeMusicCaps + a MusicCaps audio copy, band-limited to match |
| M6 (robustness) | follows whichever of the above is trained |
