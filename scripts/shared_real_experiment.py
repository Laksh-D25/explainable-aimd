r"""Cross-generator transfer with the real class held genuinely fixed.

The paper's central claim is that the FakeMusicCaps collapse is caused by the
*real* class changing, not by the generators changing. The evidence for that is
circumstantial: FPR goes to 1.00, which says the detector stopped recognising
real music. It is circumstantial because in that evaluation the corpus and the
generator moved together, so either could be blamed.

This isolates them. One real corpus -- MusicCaps -- is used on both sides. The
detector trains against Suno/Udio and is tested against five text-to-music
models it has never seen, with the same 250 MusicCaps recordings supplying the
real class throughout. Nothing varies but the generator.

    train:  MusicCaps real  vs  SONICS (Suno/Udio)
    test :  MusicCaps real  vs  FakeMusicCaps (AudioLDM2, MusicGen, MusicLDM,
                                               Mustango, StableAudioOpen)

The preprocessing is the part that decides whether the result means anything.
The three sources arrive at different sample rates and lengths:

    MusicCaps real_16k   16 kHz   10.0 s   mono
    FakeMusicCaps TTM    16 kHz   10.0 s   mono
    SONICS Suno/Udio     24 kHz   60.0 s   mono

Training 24 kHz fakes against 16 kHz reals would teach the detector that energy
above 8 kHz means "real", and every test clip is also capped at 8 kHz, so it
would call the whole test set real and collapse for a reason that has nothing to
do with generators. Everything is therefore resampled to 16 kHz and truncated to
10 s before anything is trained, and the bandwidth check is run afterwards to
confirm it worked.

    python scripts/shared_real_experiment.py --out artifacts/sharedreal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimd.data.audio import bandwidth_report  # noqa: E402
from aimd.eval.metrics import summary  # noqa: E402
from aimd.eval.protocols import build_loader  # noqa: E402
from aimd.pipeline import (  # noqa: E402
    TrainConfig,
    build_model,
    calibrate_and_threshold,
    resolve_device,
    train_model,
)
from aimd.train.loop import predict  # noqa: E402

FMC = Path.home() / "sonics" / "fmc"
SONICS_CACHE = Path.home() / "sonics" / "cache_big"
COMMON_SR = 16_000
INTERMEDIATE_SR = 48_000
COMMON_SECONDS = 10.0

TTM = {
    "audioldm2": "audioldm2",
    "musicgen": "MusicGen_medium",
    "musicldm": "musicldm",
    "mustango": "mustango",
    "stable_audio_open": "stable_audio_open",
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def harmonise(src: Path, dst: Path) -> bool:
    """Write src to dst at a common sample rate and length.

    Returns False when the source is too short to fill the window: padding it
    would introduce silence that correlates with the source, which is the same
    class of shortcut this preprocessing exists to remove.
    """
    if dst.exists():
        return True
    import torchaudio

    wav, sr = torchaudio.load(str(src))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    # Route every source through the SAME resampling chain, including the ones
    # already at the target rate. Downsampling 24 kHz SONICS to 16 kHz while
    # leaving the 16 kHz MusicCaps and TTM clips untouched would stamp an
    # anti-alias signature on exactly one class, and a detector that learned it
    # would call every untouched clip real -- which is indistinguishable from
    # the generator effect this experiment is trying to measure.
    if sr != INTERMEDIATE_SR:
        wav = torchaudio.functional.resample(wav, sr, INTERMEDIATE_SR)
    wav = torchaudio.functional.resample(wav, INTERMEDIATE_SR, COMMON_SR)
    n = int(COMMON_SR * COMMON_SECONDS)
    if wav.shape[1] < n:
        return False
    # Take the middle: a generated clip's first moments are often near-silent,
    # and a 60 s song's opening is the least representative part of it.
    start = (wav.shape[1] - n) // 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), wav[:, start:start + n], COMMON_SR)
    return True


def build_corpus(cache: Path, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    reals = sorted((FMC / "real_16k").glob("*.wav"))
    log(f"  MusicCaps real: {len(reals)}")
    for p in reals:
        dst = cache / "real" / p.name
        if harmonise(p, dst):
            rows.append({"song_id": f"mc_{p.stem}", "path": str(dst), "label": 0,
                         "taxonomy": "real", "source": "musiccaps",
                         "group": f"mc_{p.stem}", "duration": COMMON_SECONDS})

    sonics = sorted(SONICS_CACHE.glob("fake_*.wav"))
    log(f"  SONICS fake:    {len(sonics)}")
    for p in sonics:
        dst = cache / "sonics" / p.name
        if harmonise(p, dst):
            gen = "suno" if "suno" in p.stem else "udio"
            rows.append({"song_id": p.stem, "path": str(dst), "label": 1,
                         "taxonomy": "mostly_fake", "source": f"sonics_{gen}",
                         "group": p.stem.rsplit("_", 1)[0], "duration": COMMON_SECONDS})

    for name, folder in TTM.items():
        files = sorted((FMC / "generated" / folder).glob("*.wav"))
        log(f"  TTM {name:18s} {len(files)}")
        for p in files:
            dst = cache / "ttm" / name / p.name
            if harmonise(p, dst):
                rows.append({"song_id": f"{name}_{p.stem}", "path": str(dst), "label": 1,
                             "taxonomy": "mostly_fake", "source": f"ttm_{name}",
                             "group": f"{name}_{p.stem}", "duration": COMMON_SECONDS})

    data = pd.DataFrame(rows)

    # Splits. The real class is divided once and its test portion is reused by
    # both evaluations, which is the entire point: the same recordings face the
    # seen generators and the unseen ones.
    real_idx = np.array(data.index[data["source"] == "musiccaps"])
    rng.shuffle(real_idx)
    n_train, n_val = 150, 40
    data.loc[real_idx[:n_train], "split"] = "train"
    data.loc[real_idx[n_train:n_train + n_val], "split"] = "val"
    data.loc[real_idx[n_train + n_val:], "split"] = "test"

    son_idx = np.array(data.index[data["source"].str.startswith("sonics")])
    rng.shuffle(son_idx)
    data.loc[son_idx[:n_train], "split"] = "train"
    data.loc[son_idx[n_train:n_train + n_val], "split"] = "val"
    # Balance the in-distribution test set against the real test portion.
    n_real_test = len(real_idx) - n_train - n_val
    data.loc[son_idx[n_train + n_val:n_train + n_val + n_real_test], "split"] = "test"
    data.loc[son_idx[n_train + n_val + n_real_test:], "split"] = "unused"

    data.loc[data["source"].str.startswith("ttm"), "split"] = "cross"
    return data.dropna(subset=["split"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/sharedreal"))
    ap.add_argument("--cache", type=Path, default=Path.home() / "sonics" / "shared_real")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    log("=== harmonising sources to 16 kHz / 10 s ===")
    data = build_corpus(args.cache)
    data.to_csv(args.out / "manifest.csv", index=False)
    log("\n" + data.groupby(["split", "source"]).size().to_string())

    # If this still separates the classes, nothing below is interpretable.
    log("\n=== bandwidth check (must be ~1.00 after harmonising) ===")
    bw = bandwidth_report(data[data["split"].isin(["train", "val"])], n_per_class=40)
    log(f"  {bw}")

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                         clip_seconds=5.0, clips_per_song=2, augment=True,
                         num_workers=2, amp=device.startswith("cuda"))
    spec = config.spec()
    amp = config.amp

    log("\n=== training: MusicCaps real vs Suno/Udio ===")
    torch.manual_seed(config.seed)
    model = build_model(device)
    result = train_model(model, data, config, device=device,
                         checkpoint_dir=args.out / "ckpt", log=log)
    result["history"].to_csv(args.out / "history.csv", index=False)
    log(f"  best val F1 {result['best']['f1']:.4f} (epoch {result['best']['epoch'] + 1})")

    scaler, threshold, calib = calibrate_and_threshold(
        model, data, spec, target_fpr=0.05, device=device, amp=amp)
    log(f"  calibration {json.dumps({k: round(v, 6) for k, v in calib.items()})}")

    def evaluate(rows: pd.DataFrame, name: str) -> dict:
        frame = rows.copy()
        frame["split"] = "test"
        out = predict(model, build_loader(frame, spec, "test", batch_size=1),
                      device=device, amp=amp)
        m = summary(out["label"], scaler.transform(out["logit"]), threshold=threshold)
        pos = float((out["label"] == 1).mean())
        m["all_positive_f1"] = 2 * pos / (pos + 1) if pos else 0.0
        m["n"] = len(out["label"])
        log(f"  {name:34s} F1 {m['f1']:.4f}  AUROC {m['auroc']:.4f}  "
            f"FPR {m['fpr']:.3f}  ECE {m['ece']:.4f}  (n={m['n']}, "
            f"all-positive F1 {m['all_positive_f1']:.4f})")
        return m

    real_test = data[(data["source"] == "musiccaps") & (data["split"] == "test")]
    son_test = data[data["source"].str.startswith("sonics") & (data["split"] == "test")]
    ttm = data[data["split"] == "cross"]

    log("\n=== results ===")
    results = {
        "in_distribution": evaluate(pd.concat([real_test, son_test]),
                                    "in-distribution (Suno/Udio)"),
        "cross_generator_shared_real": evaluate(pd.concat([real_test, ttm]),
                                                "cross-generator, shared real"),
    }

    log("\n  per generator (each against the same real test set):")
    per_gen = {}
    for name in TTM:
        gen = ttm[ttm["source"] == f"ttm_{name}"]
        per_gen[name] = evaluate(pd.concat([real_test, gen]), f"    {name}")
    results["per_generator"] = per_gen
    results["calibration"] = calib
    results["bandwidth"] = bw
    results["n_train"] = int((data["split"] == "train").sum())

    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    pd.DataFrame([
        {"setting": k, **{m: v[m] for m in ("f1", "auroc", "fpr", "ece", "n",
                                            "all_positive_f1")}}
        for k, v in list(results.items())[:2]
    ] + [{"setting": f"ttm_{k}", **{m: v[m] for m in ("f1", "auroc", "fpr", "ece", "n",
                                                      "all_positive_f1")}}
         for k, v in per_gen.items()]).to_csv(args.out / "results.csv", index=False)

    a = results["in_distribution"]["auroc"]
    b = results["cross_generator_shared_real"]["auroc"]
    log(f"\n  AUROC {a:.4f} -> {b:.4f} ({b - a:+.4f}) with the real class held fixed")
    log(f"  FPR on the same real recordings: {results['in_distribution']['fpr']:.3f} -> "
        f"{results['cross_generator_shared_real']['fpr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
