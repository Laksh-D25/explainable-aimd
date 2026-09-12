"""Resume logic for the real-song fetcher.

The full fetch runs ~26 hours, so it will be interrupted. These cover the parts
that decide whether a resumed run continues or silently redoes/skips work --
none of them need the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_real_songs import PERMANENT, TRANSIENT, State, classify_error, is_complete


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "state.jsonl"
    State(path).record("real_1", "ok")
    State(path).record("real_2", "timeout")
    assert State(path).entries == {"real_1": "ok", "real_2": "timeout"}


def test_state_is_flushed_per_song_not_at_the_end(tmp_path):
    """A hard kill must lose at most the in-flight downloads, not the run."""
    path = tmp_path / "state.jsonl"
    state = State(path)
    state.record("real_1", "ok")
    assert json.loads(path.read_text().splitlines()[0])["filename"] == "real_1"


def test_torn_final_line_is_tolerated(tmp_path):
    """A kill mid-write can leave a partial line; it must not break resume."""
    path = tmp_path / "state.jsonl"
    path.write_text(json.dumps({"filename": "real_1", "status": "ok"}) + "\n{\"filena")
    assert State(path).entries == {"real_1": "ok"}


def test_completed_songs_are_skipped(tmp_path):
    state = State(tmp_path / "s.jsonl")
    state.record("real_1", "ok")
    assert state.should_skip("real_1", retry_failed=False)


def test_permanent_failures_are_skipped_but_transient_are_retried(tmp_path):
    """Retrying removed videos every run would waste hours; retrying timeouts
    is exactly what a resume is for."""
    state = State(tmp_path / "s.jsonl")
    for status in PERMANENT:
        state.record(f"perm_{status}", status)
    for status in TRANSIENT:
        state.record(f"tran_{status}", status)

    assert all(state.should_skip(f"perm_{s}", False) for s in PERMANENT)
    assert not any(state.should_skip(f"tran_{s}", False) for s in TRANSIENT)


def test_retry_failed_reattempts_permanent_ones(tmp_path):
    state = State(tmp_path / "s.jsonl")
    state.record("gone", "removed")
    assert state.should_skip("gone", retry_failed=False)
    assert not state.should_skip("gone", retry_failed=True)


def test_unseen_song_is_not_skipped(tmp_path):
    assert not State(tmp_path / "s.jsonl").should_skip("never_tried", False)


def test_permanent_and_transient_do_not_overlap():
    assert not (PERMANENT & TRANSIENT)


def test_partial_file_is_not_treated_as_complete(tmp_path):
    """The trap: an interrupted download leaves a large-but-truncated file.
    Trusting size alone would resume onto silently truncated audio."""
    tiny = tmp_path / "tiny.mp3"
    tiny.write_bytes(b"\x00" * 100)
    assert not is_complete(tiny, expected=180.0)

    missing = tmp_path / "absent.mp3"
    assert not is_complete(missing, expected=180.0)


def test_is_complete_accepts_a_real_clip_of_expected_length(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "full.wav"
    sf.write(path, np.zeros(16000 * 10, dtype="float32"), 16000)
    assert is_complete(path, expected=10.0)
    assert is_complete(path, expected=12.0)      # within 25% tolerance
    assert not is_complete(path, expected=60.0)  # far too short


@pytest.mark.parametrize(
    "stderr,expected",
    [
        (b"ERROR: Video unavailable", "unavailable"),
        (b"ERROR: This video is private", "private"),
        (b"ERROR: video has been removed by the uploader", "removed"),
        (b"ERROR: Sign in to confirm your age", "age_restricted"),
        (b"ERROR: not available in your country", "geo_blocked"),
        (b"ERROR: some transient network hiccup", "failed"),
        (b"", "failed"),
    ],
)
def test_errors_are_classified_for_resume(stderr, expected):
    assert classify_error(stderr) == expected


def test_limit_zero_means_zero_not_unlimited(tmp_path):
    """`--limit 0` once fell through a truthiness check and began fetching all
    48,090 songs. A bounded flag that silently unbounds itself is dangerous."""
    import subprocess
    import sys

    import pandas as pd

    csv = tmp_path / "real_songs.csv"
    pd.DataFrame([
        {"filename": f"real_{i}", "youtube_id": f"id{i}", "duration": 10.0, "skip_time": 0.0}
        for i in range(25)
    ]).to_csv(csv, index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_real_songs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--csv", str(csv),
         "--out", str(tmp_path / "out"), "--limit", "0"],
        capture_output=True, text=True, timeout=120,
    )
    assert "0 to fetch" in proc.stdout, proc.stdout + proc.stderr
    assert "25 to fetch" not in proc.stdout


def test_negative_limit_is_rejected(tmp_path):
    import subprocess
    import sys

    import pandas as pd

    csv = tmp_path / "real_songs.csv"
    pd.DataFrame([{"filename": "r", "youtube_id": "x", "duration": 10.0,
                   "skip_time": 0.0}]).to_csv(csv, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "fetch_real_songs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--csv", str(csv),
         "--out", str(tmp_path / "out"), "--limit", "-5"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "must be >= 0" in proc.stdout + proc.stderr


# --- throttling vs permanent refusal ------------------------------------------


def test_throttling_is_transient_not_permanent():
    """The bug this encodes: a 4,000-song run hit YouTube's rate limiter at
    ~1,600 downloads and the remaining 1,799 were refused. Classing those as
    permanent would discard 45% of the dataset for a block that expires."""
    assert "throttled" in TRANSIENT
    assert "throttled" not in PERMANENT
    assert "geo_blocked" in PERMANENT


@pytest.mark.parametrize("stderr", [
    b"ERROR: Sign in to confirm you're not a bot",
    b"ERROR: HTTP Error 429: Too Many Requests",
    b"ERROR: Sign in to confirm that you're not a bot. Use --cookies",
])
def test_rate_limit_messages_classify_as_throttled(stderr):
    assert classify_error(stderr) == "throttled"


def test_throttle_is_not_mistaken_for_age_restriction():
    """The throttle message contains "sign in", which an age rule would
    otherwise claim -- turning a retryable block into a permanent one."""
    assert classify_error(b"Sign in to confirm you're not a bot") == "throttled"
    assert classify_error(b"Sign in to confirm your age") == "age_restricted"


def test_region_block_stays_permanent():
    assert classify_error(b"ERROR: Video not available in your country") == "geo_blocked"


def test_throttled_songs_are_retried_on_resume(tmp_path):
    state = State(tmp_path / "s.jsonl")
    state.record("real_1", "throttled")
    assert not state.should_skip("real_1", retry_failed=False)


def test_detector_trips_only_on_a_consecutive_streak():
    from fetch_real_songs import ThrottleDetector

    det = ThrottleDetector(limit=3)
    assert not det.record("throttled")
    assert not det.record("throttled")
    det.record("ok")                      # a success clears the streak
    assert not det.record("throttled")
    assert not det.record("throttled")
    assert det.record("throttled")        # three in a row


def test_detector_streak_survives_unrelated_failures():
    """Once throttling starts, other failures interleave; only a real success
    means the limiter has let go."""
    from fetch_real_songs import ThrottleDetector

    det = ThrottleDetector(limit=3)
    det.record("throttled")
    det.record("unavailable")   # not a success, so the streak stands
    det.record("throttled")
    assert det.record("throttled")


def test_repair_tool_makes_blocked_entries_retryable(tmp_path):
    """The recovery path for state files written by the buggy version."""
    import json as _json
    import subprocess

    state = tmp_path / "fetch_state.jsonl"
    with state.open("w") as fh:
        for i in range(5):
            fh.write(_json.dumps({"filename": f"r{i}", "status": "blocked"}) + "\n")
        fh.write(_json.dumps({"filename": "keep", "status": "ok"}) + "\n")

    script = Path(__file__).resolve().parents[1] / "scripts" / "repair_fetch_state.py"
    proc = subprocess.run([sys.executable, str(script), "--state", str(state)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    restored = State(state)
    assert restored.entries == {"keep": "ok"}          # blocked entries gone
    assert not restored.should_skip("r0", False)       # so they get retried
    assert state.with_suffix(".jsonl.bak").exists()    # original preserved
