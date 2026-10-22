"""A dry controller run must launch only the two requested joint placements."""
import json
from types import SimpleNamespace

import torch
import pytest

from tactilelatent import fgw_pilot as pilot


@pytest.mark.parametrize('endpoint', [20, 30])
def test_pilot_has_two_joint_runs_no_screening_or_extra_seeds(tmp_path, monkeypatch, endpoint):
    sources = [f"source{i}" for i in range(11)]
    native = {s: {"normalized_mae": .1} for s in sources}
    parent = tmp_path / "parent.pt"
    data = {"native": "verified"}
    torch.save({"epoch": 9, "config": {"seed": 45}, "provenance": {"data": data},
                "validation": native}, parent)
    base = tmp_path / "base.json"; base.write_text(json.dumps({"epochs": 50, "heldout_source": "ftp_motiontrans",
                                                              "manifest": "fixture.json", "workers": 0}))
    monkeypatch.setattr(pilot, "BASE_CONFIG", str(base))
    monkeypatch.setattr(pilot, "EXISTING_PARENT", str(parent))
    monkeypatch.setattr(pilot, "FULL_ROOT", str(tmp_path / "superseded"))
    monkeypatch.setattr(pilot, "source_context", lambda cfg: (data, {}, sources, {}, {}, {}, {}))
    monkeypatch.setattr(pilot, "signature", lambda: {"test": "fixed"})
    monkeypatch.setattr(pilot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    spec = pilot.prepare(tmp_path / "pilot")
    if endpoint != 20:
        spec.update(completed_epoch=endpoint, additional_joint_epochs_per_run=endpoint - 10)
        pilot.write_json(tmp_path / 'pilot/configs/visual.json', pilot.joint_config(spec, 'visual'))
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)
            module = command[command.index("-m") + 1]
            def arg(key):
                return command[command.index(key) + 1]
            if "--config" in command and "--cache-only" not in command:
                from pathlib import Path
                cfg = json.loads(Path(arg("--config")).read_text())
                folder = Path(cfg["output"]); folder.mkdir(parents=True, exist_ok=True)
                epoch = 69 if module.endswith("tactile_teacher") else cfg['epochs'] - 1
                saved = {"config": cfg, "epoch": epoch, "validation": native}
                torch.save(saved, folder / "last.pt")
                if module.endswith('train_alignment'):
                    torch.save(saved, folder / f"epoch_{epoch + 1:03d}.pt")
            elif module.endswith("evaluate_holdout_decoder"):
                from pathlib import Path
                folder = Path(arg("--output")); folder.mkdir(parents=True, exist_ok=True)
                start = command.index("--hidden-dims") + 1
                end = command.index("--seed")
                value = {"checkpoint_sha256": pilot.file_hash(arg("--checkpoint")),
                    "seed": 45, "head": "-".join(["32", *command[start:end], "6"]),
                    "train_fraction": 1., "validation": {"sample_count": 64, "deterministic_mae": .05}}
                (folder / "results.json").write_text(json.dumps(value))

        def wait(self):
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(pilot.subprocess, "Popen", Process)
    pilot.run(spec)
    training = [c for c in calls if "tactilelatent.train_alignment" in c]
    assert len(training) == 2
    assert len([c for c in calls if "tactilelatent.evaluate_holdout_decoder" in c]) == 4
    assert len([c for c in calls if "tactilelatent.tactile_teacher" in c and "--cache-only" not in c]) == 1
    assert all("--stop-after-epoch" not in c for c in calls)
    assert spec["seed"] == 45 and not spec["screening"] and not spec["finalist_selection"]
    assert spec["additional_joint_epochs_per_run"] == endpoint - 10 and not spec["additional_seeds"]
    assert all(f'epoch_{endpoint:03d}.pt' in c[c.index('--checkpoint') + 1]
               for c in calls if 'tactilelatent.evaluate_holdout_decoder' in c)
    calls.clear(); pilot.run(spec)
    assert not any("tactilelatent.train_alignment" in c for c in calls)  # Completed models are reused.
