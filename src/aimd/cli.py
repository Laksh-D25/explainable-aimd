"""Command-line entry points.

    python -m aimd.cli info
    python -m aimd.cli prepare-data --real-csv ... --fake-csv ... --audio-root ...
    python -m aimd.cli train --manifest manifest.csv --out artifacts/run1
    python -m aimd.cli evaluate --manifest manifest.csv --checkpoint artifacts/run1/best.pt
    python -m aimd.cli explain --manifest manifest.csv --checkpoint ... --song-id s0042
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import typer

app = typer.Typer(add_completion=False, help="Explainable AI-generated music detection.")


@app.command()
def info() -> None:
    """Report the environment and what perturbation backends are usable."""
    from .data.perturb import EVAL_PERTURBATIONS, TRAIN_PERTURBATIONS, available

    typer.echo(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        typer.echo(f"  {props.name}, {props.total_memory / 1024**3:.1f} GB")

    status = {**available(TRAIN_PERTURBATIONS), **available(EVAL_PERTURBATIONS)}
    missing = [k for k, ok in status.items() if not ok]
    typer.echo(f"perturbations: {len(status) - len(missing)}/{len(status)} usable")
    if missing:
        typer.secho(f"  MISSING: {missing}", fg="red")
        typer.echo("  MP3 needs fast_mp3_augment; reverb needs pyroomacoustics.")


@app.command("prepare-data")
def prepare_data(
    real_csv: Path = typer.Option(..., help="SONICS real_songs.csv"),
    fake_csv: Path = typer.Option(..., help="SONICS fake_songs.csv"),
    audio_root: Path = typer.Option(None, help="prefix for relative audio paths"),
    out: Path = typer.Option(Path("manifest.csv")),
    seed: int = 1337,
) -> None:
    """Build a song-level manifest with leak-free, stratified splits."""
    from .data.manifest import load_sonics_manifest, split_by_song, summarize

    manifest = load_sonics_manifest(str(real_csv), str(fake_csv))
    if audio_root:
        manifest["path"] = manifest["path"].apply(lambda p: str(Path(audio_root) / str(p)))

    manifest = split_by_song(manifest, seed=seed)  # asserts no leakage internally
    manifest.to_csv(out, index=False)
    typer.echo(summarize(manifest).to_string())
    typer.secho(f"\nwrote {len(manifest)} songs to {out}", fg="green")


@app.command()
def train(
    manifest: Path = typer.Option(...),
    out: Path = typer.Option(Path("artifacts/run")),
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    clip_seconds: float = 10.0,
    clips_per_song: int = 8,
    limit_songs: int = typer.Option(None, help="cap training songs (sanity gate)"),
    no_augment: bool = typer.Option(False, "--no-augment"),
    layer_mode: str = "static",
    frame_pool: str = "attn",
    song_pool: str = "attn",
    self_attn: bool = False,
    device: str = "auto",
) -> None:
    """Train the detector head on a prepared manifest."""
    from .pipeline import TrainConfig, build_model, calibrate_and_threshold, resolve_device, train_model

    device = resolve_device(device)
    rows = pd.read_csv(manifest)
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        clip_seconds=clip_seconds,
        clips_per_song=clips_per_song,
        augment=not no_augment,
        limit_songs=limit_songs,
        amp=device.startswith("cuda"),
    )
    model = build_model(
        device,
        layer_mode=layer_mode,
        frame_pool=frame_pool,
        song_pool=song_pool,
        use_self_attn=self_attn,
    )
    typer.echo(f"device={device}  trainable={sum(p.numel() for p in model.trainable_parameters()):,}")

    result = train_model(model, rows, config, device=device, checkpoint_dir=out, log=typer.echo)
    out.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(out / "history.csv", index=False)

    _, _, calibration = calibrate_and_threshold(
        model, rows, config.spec(), device=device, amp=config.amp
    )
    typer.echo(
        f"\ncalibration: T={calibration['temperature']:.3f}  "
        f"ECE {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}  "
        f"threshold={calibration['threshold']:.4f}"
    )
    typer.secho(f"best val F1 {result['best']['f1']:.4f} (epoch {result['best']['epoch'] + 1})", fg="green")


@app.command()
def evaluate(
    manifest: Path = typer.Option(..., help="SONICS manifest (splits used for calibration)"),
    checkpoint: Path = typer.Option(...),
    cross_generator_manifest: Path = typer.Option(None, help="FakeMusicCaps manifest"),
    out: Path = typer.Option(Path("artifacts/eval")),
    target_fpr: float = 0.01,
    skip_robustness: bool = False,
    device: str = "auto",
) -> None:
    """Run the in-distribution, cross-generator and robustness protocols."""
    from .eval.protocols import (
        evaluate_cross_generator,
        evaluate_in_distribution,
        evaluate_robustness,
        results_table,
    )
    from .pipeline import TrainConfig, build_model, calibrate_and_threshold, load_checkpoint, resolve_device

    device = resolve_device(device)
    rows = pd.read_csv(manifest)
    model = build_model(device)
    payload = load_checkpoint(model, checkpoint, device)
    config = TrainConfig(**payload["config"])
    spec, amp = config.spec(), config.amp and device.startswith("cuda")

    scaler, threshold, calibration = calibrate_and_threshold(
        model, rows, spec, target_fpr=target_fpr, device=device, amp=amp
    )
    typer.echo(f"calibration: {calibration}\n")

    out.mkdir(parents=True, exist_ok=True)
    results = [evaluate_in_distribution(model, rows, spec, scaler, threshold, device=device, amp=amp)]

    if cross_generator_manifest:
        cross = evaluate_cross_generator(
            model, pd.read_csv(cross_generator_manifest), spec, scaler, threshold,
            device=device, amp=amp,
        )
        results.append(cross)
        typer.secho(
            f"cross-generator F1 {cross.metrics['f1']:.4f}  "
            f"(base paper {cross.extra['baseline_f1']}, "
            f"delta {cross.extra['delta_vs_baseline']:+.4f})",
            fg="cyan",
        )
    else:
        typer.secho("no --cross-generator-manifest: the headline number was NOT produced", fg="yellow")

    table = results_table(results)
    table.to_csv(out / "protocols.csv")
    typer.echo("\n" + table.to_string())

    if not skip_robustness:
        robustness = evaluate_robustness(model, rows, spec, scaler, threshold, device=device, amp=amp)
        robustness.to_csv(out / "robustness.csv")
        typer.echo("\n" + robustness[["f1", "f1_drop", "family"]].to_string())


@app.command()
def explain(
    manifest: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    song_id: str = typer.Option(..., help="song_id from the manifest"),
    out: Path = typer.Option(Path("artifacts/explain")),
    tf_occlusion: bool = typer.Option(False, help="also run level 2b (slow)"),
    device: str = "auto",
) -> None:
    """Explain one song across all three XAI levels, with a faithfulness check."""
    import numpy as np

    from .data.datasets import SongClipsDataset
    from .pipeline import TrainConfig, build_model, load_checkpoint, resolve_device
    from .xai.faithfulness import deletion_curve
    from .xai.relevance import temporal_relevance
    from .xai.shap_layers import agreement_with_attention, layer_shapley

    device = resolve_device(device)
    rows = pd.read_csv(manifest)
    match = rows[rows["song_id"].astype(str) == song_id]
    if match.empty:
        raise typer.BadParameter(f"song_id {song_id!r} not in manifest")

    model = build_model(device)
    payload = load_checkpoint(model, checkpoint, device)
    spec = TrainConfig(**payload["config"]).spec()
    model.eval()

    item = SongClipsDataset(match, spec, split="test")[0]
    wav = item["clips"].unsqueeze(0).to(device)
    mask = item["clip_mask"].unsqueeze(0).to(device)

    relevance = temporal_relevance(model, wav, mask)
    hidden = model.backbone(wav.flatten(0, 1), no_grad=True)
    shapley = layer_shapley(model, hidden.view(*wav.shape[:2], *hidden.shape[1:]), mask)
    attention = model.layer_attn.logits.softmax(0) if model.layer_attn.mode == "static" else None

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{song_id}.npz", relevance=relevance, shapley=shapley)

    typer.echo(f"L1 layer attention : {None if attention is None else attention.detach().cpu().numpy().round(3)}")
    typer.echo(f"L2a relevance      : {relevance.shape}, peak frame {relevance.argmax()}")
    typer.echo(f"L3 shapley         : top layers {np.argsort(np.abs(shapley))[::-1][:3].tolist()}")
    if attention is not None:
        agreement = agreement_with_attention(shapley, attention)
        typer.echo(f"L1 vs L3 agreement : {agreement:+.3f}")
        if agreement < 0.3:
            typer.secho(
                "  attention weights disagree with what moved the decision; "
                "report this rather than presenting L1 as the explanation.",
                fg="yellow",
            )

    curve = deletion_curve(model, wav, relevance, mask, steps=8)
    typer.echo(f"faithfulness gap   : {curve.gap:+.4f} "
               f"({'beats chance' if curve.gap < 0 else 'NO better than chance'})")

    if tf_occlusion:
        from .xai.tf_occlusion import tf_occlusion_map

        tf_map = tf_occlusion_map(model, wav, mask)
        (f_lo, f_hi), (t_lo, t_hi) = tf_map.top_tile()
        typer.echo(f"L2b top tile       : {f_lo:.0f}-{f_hi:.0f} Hz at {t_lo:.2f}-{t_hi:.2f} s")
        np.save(out / f"{song_id}_tf.npy", tf_map.importance)


if __name__ == "__main__":
    app()
