"""Training orchestration and checkpoint round-trip.

Checkpoints deliberately exclude the frozen backbone, which makes the
save/load path easy to get wrong in a way that fails silently -- a model that
loads with randomly-initialised pooling would still produce plausible-looking
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from aimd.data.manifest import split_by_song
from aimd.models.detector import Detector
from aimd.pipeline import TrainConfig, calibrate_and_threshold, load_checkpoint, train_model

SR = 24_000


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * SR)) / SR
    rows = []
    for i in range(40):
        fake = i % 2
        audio = 0.2 * rng.standard_normal(len(t))
        if fake:
            audio = audio + 0.3 * np.sin(2 * np.pi * 6000 * t)
        path = tmp_path / f"s{i}.wav"
        sf.write(path, audio.astype(np.float32), SR)
        rows.append({
            "song_id": f"s{i}", "path": str(path), "label": fake,
            "taxonomy": "full_fake" if fake else "real",
            "source": "suno" if fake else None, "duration": 1.5,
        })
    return split_by_song(pd.DataFrame(rows))


@pytest.fixture
def config():
    return TrainConfig(
        epochs=2, batch_size=4, clip_seconds=0.5, clips_per_song=2, max_clips=2,
        augment=False, num_workers=0, amp=False, lr=3e-3,
    )


@pytest.fixture
def model(stub_backbone):
    torch.manual_seed(0)
    return Detector(backbone=stub_backbone)


def test_training_records_history_and_selects_a_best_epoch(model, corpus, config):
    result = train_model(model, corpus, config, log=lambda *_: None)
    assert len(result["history"]) == config.epochs
    assert {"epoch", "train_loss", "f1", "auroc"} <= set(result["history"].columns)
    assert 0 <= result["best"]["epoch"] < config.epochs


def test_checkpoint_excludes_the_frozen_backbone(model, corpus, config, tmp_path):
    """Saving 94 M frozen parameters per checkpoint would waste ~380 MB for no
    information -- the backbone is reloadable from its own name."""
    train_model(model, corpus, config, checkpoint_dir=tmp_path, log=lambda *_: None)
    payload = torch.load(tmp_path / "best.pt", weights_only=False)
    assert not any(k.startswith("backbone.") for k in payload["state_dict"])
    assert any("layer_attn" in k or "binary_head" in k for k in payload["state_dict"])


def test_checkpoint_round_trip_reproduces_predictions(stub_backbone, corpus, config, tmp_path):
    """A checkpoint that loads with re-randomised pooling would still produce
    plausible numbers, so this compares actual outputs."""
    torch.manual_seed(0)
    trained = Detector(backbone=stub_backbone)
    train_model(trained, corpus, config, checkpoint_dir=tmp_path, log=lambda *_: None)
    trained.eval()

    torch.manual_seed(999)  # different init
    restored = Detector(backbone=stub_backbone)
    load_checkpoint(restored, tmp_path / "best.pt")
    restored.eval()

    wav = torch.randn(2, 2, int(0.5 * SR))
    with torch.no_grad():
        torch.testing.assert_close(
            trained.forward(wav).binary_logit, restored.forward(wav).binary_logit
        )


def test_loading_a_checkpoint_missing_trainable_weights_fails_loudly(model, tmp_path):
    torch.save({"state_dict": {}, "config": {}}, tmp_path / "empty.pt")
    with pytest.raises(ValueError, match="missing trainable weights"):
        load_checkpoint(model, tmp_path / "empty.pt")


def test_calibration_reports_the_ece_change(model, corpus, config):
    _, threshold, report = calibrate_and_threshold(
        model, corpus, config.spec(), target_fpr=0.1
    )
    assert {"temperature", "ece_before", "ece_after", "threshold"} == set(report)
    assert report["temperature"] > 0
    assert 0.0 <= threshold <= 1.0


def test_limit_songs_caps_the_training_set(model, corpus, config, tmp_path):
    """The sanity gate runs on a deliberately tiny subset before Kaggle quota
    is spent."""
    capped = TrainConfig(**{**config.__dict__, "limit_songs": 4, "epochs": 1})
    result = train_model(model, corpus, capped, log=lambda *_: None)
    assert len(result["history"]) == 1


def test_kaggle_notebook_imports_resolve():
    """The notebook produces the reportable numbers. A stale import there fails
    on Kaggle only after setup time has been spent, so it is checked here."""
    import ast
    import importlib
    import json
    from pathlib import Path

    nb = json.loads(Path("notebooks/kaggle_train.ipynb").read_text())
    broken, checked = [], 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(l for l in cell["source"] if not l.lstrip().startswith(("!", "%")))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("aimd"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    checked += 1
                    if not hasattr(module, alias.name):
                        broken.append(f"{node.module}.{alias.name}")
    assert checked > 0, "no aimd imports found -- did the notebook move?"
    assert not broken, f"notebook imports missing from the package: {broken}"


def test_kaggle_notebook_code_cells_parse():
    import ast
    import json
    from pathlib import Path

    nb = json.loads(Path("notebooks/kaggle_train.ipynb").read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(l for l in cell["source"] if not l.lstrip().startswith(("!", "%")))
        ast.parse(src)  # raises SyntaxError with the offending cell in the traceback


def test_max_clips_defaults_to_clips_per_song():
    """A larger max_clips only pads, and every padded clip used to cost a full
    MERT forward."""
    assert TrainConfig(clips_per_song=2).spec().max_clips == 2
    assert TrainConfig(clips_per_song=8).spec().max_clips == 8
    assert TrainConfig(clips_per_song=8, max_clips=12).spec().max_clips == 12
