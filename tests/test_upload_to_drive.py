"""Drive upload logic.

Exercised against an rclone `local` remote, which uses the same rclone code
path as Drive without needing credentials. Skipped when rclone is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

RCLONE = shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")
pytestmark = pytest.mark.skipif(not Path(RCLONE).exists(), reason="rclone not installed")


@pytest.fixture(autouse=True)
def _rclone_on_path():
    os.environ["PATH"] = f"{Path(RCLONE).parent}:{os.environ['PATH']}"
    subprocess.run([RCLONE, "config", "create", "testlocal", "local"],
                   capture_output=True, timeout=60)


def _shard(path: Path, mb: int = 1) -> Path:
    path.write_bytes(os.urandom(mb * 1024 * 1024))
    return path


def _upload(local, remote, extra=()):
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "upload_to_drive.py"),
         "--local", str(local), "--remote", remote, *extra],
        capture_output=True, text=True, timeout=300,
    )


def test_unconfigured_remote_fails_before_uploading(tmp_path):
    """Fail early with an actionable message, not mid-transfer."""
    _shard(tmp_path / "a.tar")
    proc = _upload(tmp_path, "definitelynotconfigured:x")
    assert proc.returncode != 0
    assert "not configured" in proc.stdout + proc.stderr
    assert "rclone config" in proc.stdout + proc.stderr


def test_uploads_and_verifies(tmp_path):
    src, dest = tmp_path / "shards", tmp_path / "dest"
    src.mkdir(); dest.mkdir()
    _shard(src / "s0.tar")
    proc = _upload(src, f"testlocal:{dest}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "s0.tar").stat().st_size == (src / "s0.tar").stat().st_size


def test_already_uploaded_shards_are_skipped(tmp_path):
    """Resume must not re-send gigabytes that are already there."""
    src, dest = tmp_path / "shards", tmp_path / "dest"
    src.mkdir(); dest.mkdir()
    _shard(src / "s0.tar")
    _upload(src, f"testlocal:{dest}")

    proc = _upload(src, f"testlocal:{dest}")
    assert "1 already on Drive | 0 to upload" in proc.stdout


def test_truncated_remote_copy_is_reuploaded(tmp_path):
    """A partial upload must be detected by size and resent, not accepted."""
    src, dest = tmp_path / "shards", tmp_path / "dest"
    src.mkdir(); dest.mkdir()
    _shard(src / "s0.tar", mb=2)
    _upload(src, f"testlocal:{dest}")

    with open(dest / "s0.tar", "r+b") as fh:
        fh.truncate(1024)

    proc = _upload(src, f"testlocal:{dest}")
    assert "1 to upload" in proc.stdout
    assert (dest / "s0.tar").stat().st_size == (src / "s0.tar").stat().st_size


def test_delete_after_upload_frees_local_disk(tmp_path):
    """The whole point on a 17 GB disk: fetch -> upload -> reclaim."""
    src, dest = tmp_path / "shards", tmp_path / "dest"
    src.mkdir(); dest.mkdir()
    _shard(src / "s0.tar")
    proc = _upload(src, f"testlocal:{dest}", ["--delete-after-upload"])
    assert proc.returncode == 0
    assert not (src / "s0.tar").exists()
    assert (dest / "s0.tar").exists()


def test_nothing_is_deleted_without_the_flag(tmp_path):
    src, dest = tmp_path / "shards", tmp_path / "dest"
    src.mkdir(); dest.mkdir()
    _shard(src / "s0.tar")
    _upload(src, f"testlocal:{dest}")
    assert (src / "s0.tar").exists()


def test_empty_shard_directory_fails_loudly(tmp_path):
    src = tmp_path / "empty"; src.mkdir()
    proc = _upload(src, f"testlocal:{tmp_path}/dest")
    assert proc.returncode != 0
    assert "no *.tar files" in proc.stdout + proc.stderr
