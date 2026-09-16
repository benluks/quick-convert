from pathlib import Path

from quick_convert.cli import export as export_module


def test_export_command_forwards_arguments(monkeypatch, tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    destination = tmp_path / "model"
    received = None

    def fake_export(*args, **kwargs):
        nonlocal received
        received = (args, kwargs)
        return destination

    monkeypatch.setattr(export_module, "export_inference_artifact", fake_export)

    result = export_module.main(
        [
            str(run_dir),
            str(destination),
            "--checkpoint",
            "checkpoints/best.ckpt",
            "--device",
            "cuda",
            "--overwrite",
        ]
    )

    assert result == destination
    assert received == (
        (run_dir, destination),
        {
            "checkpoint": "checkpoints/best.ckpt",
            "config": "config.yaml",
            "map_location": "cuda",
            "overwrite": True,
        },
    )
    assert capsys.readouterr().out == f"{destination}\n"
