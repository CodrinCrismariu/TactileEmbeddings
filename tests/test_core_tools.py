"""Declared tools: safe help, real API routing, explicit bounded tuning and paths."""

import json
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from tactilelatent_tools.cli import main
from tactilelatent_tools.commands.tune import resolve_plan

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_code_resolves_archives_from_explicit_project(monkeypatch, tmp_path):
    import tactilelatent.configuration as configuration

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = tmp_path / "code"
    monkeypatch.setattr(
        configuration, "__file__", str(checkout / "src/tactilelatent/configuration.py")
    )
    assert configuration.metadata_root(workspace) == workspace
    (checkout / "legacy/v1").mkdir(parents=True)
    assert configuration.metadata_root(workspace) == checkout


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    configs = root / "configs"
    configs.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    components = {
        "model": {"latent_dim": 32, "sensor_code_dims": {"z_source": 8, "a_source": 16}},
        "datasets": {"sources": ["source_a"], "heldout_source": "ftp_motiontrans"},
        "training": {"output": "processing/runs/current", "epochs": 30, "seed": 45},
        "transfer": {"hidden_dims": [512, 512, 512]},
    }
    for name, content in components.items():
        (configs / f"{name}.json").write_text(json.dumps(content))
    (configs / "experiment.json").write_text(
        json.dumps({name: f"{name}.json" for name in components})
    )
    return root


def fake_api(monkeypatch, qualified, **functions):
    module = types.ModuleType(qualified)
    module.__dict__.update(functions)
    monkeypatch.setitem(sys.modules, qualified, module)


def invoke(project, *args):
    return main(["--project-root", str(project), *args])


@pytest.mark.parametrize(
    "arguments",
    [[], ["train"], ["evaluate"], ["tune"], ["data"], ["data", "download"], ["view"], ["status"]],
)
def test_help_never_imports_research_gpu_or_network_modules(tmp_path, arguments):
    script = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tactilelatent','torch','numpy','av','httpx','requests','huggingface_hub'}:
            raise AssertionError('Unexpected help import: '+fullname)
sys.meta_path.insert(0, Block())
from tactilelatent_tools.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "tools")]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    before = list(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert list(tmp_path.iterdir()) == before


def test_train_routes_real_current_core_api(project, monkeypatch, capsys):
    calls = []

    def train(experiment, **kwargs):
        calls.append((experiment, kwargs))
        return {"stage": "preflight_passed"}

    fake_api(monkeypatch, "tactilelatent.training.engine", train_experiment=train)
    assert invoke(project, "train", "--resume", "processing/runs/previous", "--preflight-only") == 0
    assert calls == [
        (
            project / "configs/experiment.json",
            {
                "project_root": project,
                "resume": project / "processing/runs/previous",
                "preflight_only": True,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["stage"] == "preflight_passed"


def test_explicit_checkpoint_import_routes_to_current_trainer(project, monkeypatch):
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.training.engine",
        train_experiment=lambda config, **kwargs: calls.append((config, kwargs)),
    )
    assert (
        invoke(
            project,
            "train",
            "--import-checkpoint",
            "processing/runs/parent/last.pt",
            "--import-only",
        )
        == 0
    )
    assert calls[0][1] == {
        "project_root": project,
        "resume": None,
        "preflight_only": False,
        "import_checkpoint": project / "processing/runs/parent/last.pt",
        "import_only": True,
    }


def test_checkpoint_import_modes_are_explicit_and_mutually_exclusive(project, monkeypatch):
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.training.engine",
        train_experiment=lambda *args, **kwargs: calls.append(kwargs),
    )
    assert invoke(project, "train", "--import-only") == 2
    assert invoke(project, "train", "--import-checkpoint", "last.pt", "--preflight-only") == 2
    with pytest.raises(SystemExit):
        invoke(
            project,
            "train",
            "--resume",
            "processing/runs/previous",
            "--import-checkpoint",
            "last.pt",
        )
    assert calls == []


def test_evaluation_routes_run_epochs_protocol(project, monkeypatch):
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.evaluation.transfer",
        evaluate_experiment=lambda experiment, **kwargs: calls.append((experiment, kwargs)),
    )
    assert (
        invoke(
            project,
            "evaluate",
            "--run",
            "processing/runs/current",
            "--epochs",
            "5",
            "10",
            "--protocol",
            "dataset-probe",
        )
        == 0
    )
    assert calls[0][1] == {
        "project_root": project,
        "run": project / "processing/runs/current",
        "epochs": [5, 10],
        "protocol": "dataset-probe",
    }
    assert invoke(project, "evaluate", "--run", "processing/runs/current", "--epochs", "0") == 2
    assert len(calls) == 1


def test_data_management_never_imports_model_trainer(project, monkeypatch):
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.dataset_management.api",
        manage=lambda action, **kwargs: calls.append((action, kwargs)),
    )
    assert invoke(project, "data", "verify", "--source", "source_a", "--sample", "--full") == 0
    assert calls == [
        (
            "verify",
            {
                "config": {"sources": ["source_a"], "heldout_source": "ftp_motiontrans"},
                "project_root": project,
                "source": "source_a",
                "output": None,
                "sample": True,
                "full": True,
            },
        )
    ]
    assert invoke(project, "data", "verify", "--url", "https://invalid.example/file") == 2
    assert len(calls) == 1


def test_view_routes_explicit_root_and_experiment(project, monkeypatch):
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.visualization.server",
        serve=lambda **kwargs: calls.append(kwargs),
    )
    assert invoke(project, "view", "--host", "0.0.0.0", "--port", "8081") == 0
    assert calls == [
        {
            "project_root": project,
            "host": "0.0.0.0",
            "port": 8081,
            "experiment": project / "configs/experiment.json",
        }
    ]


def test_status_uses_project_root_from_private_task_cwd(project, tmp_path, monkeypatch, capsys):
    task = tmp_path / "private-task"
    task.mkdir()
    monkeypatch.chdir(task)
    before = sorted(project.rglob("*"))
    assert invoke(project, "status") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == {"stage": "not_started"}
    assert result["run"] == str(project / "processing/runs/current")
    assert result["writes"] is False
    assert sorted(project.rglob("*")) == before
    assert not list(task.iterdir())


def test_status_tolerates_one_incomplete_live_metrics_line(project, capsys):
    run = project / "processing/runs/historical"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"stage": "training", "epoch": 4}))
    (run / "metrics.jsonl").write_text('{"epoch":3,"loss":0.2}\n{"epoch":')
    assert invoke(project, "status", "--run", "processing/runs/historical") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["last_metrics"]["epoch"] == 3


def tune_plan(project, **updates):
    plan = {
        "candidates": ["configs/experiment.json"],
        "seeds": [45],
        "budget": {"epochs_per_trial": 2, "max_trials": 1},
        "output": "processing/tuning/test",
    }
    plan.update(updates)
    path = project / "plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.mark.parametrize(
    "updates",
    [
        {"candidates": []},
        {"seeds": []},
        {"seeds": [45, 46]},
        {"seeds": [True]},
        {"budget": {"max_trials": 1}},
        {"budget": {"epochs_per_trial": 0, "max_trials": 1}},
        {"output": "configs/rewrite"},
        {"output": "../outside"},
    ],
)
def test_tuning_requires_explicit_finite_valid_budget(project, updates):
    path = tune_plan(project, **updates)
    with pytest.raises(ValueError):
        resolve_plan(path, project)
    assert not (project / "processing").exists()


def test_tuning_dry_run_writes_nothing_and_imports_no_trainer(project, monkeypatch, capsys):
    path = tune_plan(project)
    before = sorted(project.rglob("*"))
    assert invoke(project, "tune", "--plan", str(path), "--dry-run") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["writes"] is False
    assert result["trial_count"] == 1
    assert result["total_epoch_budget"] == 2
    assert sorted(project.rglob("*")) == before


def test_tuning_materializes_only_explicit_trials_preserving_head_order(project, monkeypatch):
    path = tune_plan(project, seeds=[45, 46], budget={"epochs_per_trial": 3, "max_trials": 2})
    calls = []

    def train(config, **kwargs):
        document = json.loads(config.read_text())
        calls.append((document, kwargs))
        assert list(document["model"]["sensor_code_dims"]) == ["z_source", "a_source"]
        assert document["training"]["epochs"] == 3
        assert document["training"]["seed"] in (45, 46)
        return {"stage": "complete"}

    fake_api(monkeypatch, "tactilelatent.training.engine", train_experiment=train)
    assert invoke(project, "tune", "--plan", str(path)) == 0
    assert len(calls) == 2
    assert {call[0]["training"]["seed"] for call in calls} == {45, 46}
    assert invoke(project, "tune", "--plan", str(path)) == 2
    assert len(calls) == 2
    # Same plan hash with changed candidate data must still refuse resume.
    model = project / "configs/model.json"
    model.write_text(json.dumps({"latent_dim": 16}))
    assert invoke(project, "tune", "--plan", str(path), "--resume") == 2
    assert len(calls) == 2


def test_checked_in_configuration_preserves_reference_architecture_and_selection():
    configuration = json.loads((ROOT / "configs/model.json").read_text())
    baseline = json.loads(
        (ROOT / "legacy/v1/configs/joint_vjepa21_all_sources_32d_mirror_v2.json").read_text()
    )
    datasets = json.loads((ROOT / "configs/datasets.json").read_text())
    training = json.loads((ROOT / "configs/training.json").read_text())
    assert configuration["latent_dim"] == baseline["latent_dim"] == 32
    assert configuration["adapter_hidden_dims"] == baseline["adapter_hidden_dims"]
    for key in ("width", "heads", "layers", "feedforward", "dropout", "grid_size"):
        assert configuration[key] == baseline["attention"][key]
    assert "sensor_code_dims" not in configuration and "numeric_shapes" not in configuration
    assert len(set(datasets["sources"])) == len(datasets["sources"])
    assert configuration["statistics_artifact"] == datasets["registry_paths"]["model_config"]
    assert len(datasets["sources"]) == 17
    assert datasets["heldout_source"] == "ftp_motiontrans"
    assert "ftp_motiontrans" not in datasets["sources"]
    assert set(datasets["excluded_sources"]) == {"freetacman", "tacquad", "tdex"}
    assert datasets["mirror"]["sources"] == baseline["mirror_augmentation"]["sources"]
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "accumulation",
        "learning_rate",
        "sam_rho",
        "wrist_dropout",
    ):
        assert training[key] == baseline[key]
    assert training["output"] != baseline["output"]


def test_packaging_declares_small_cli_and_never_discovers_legacy():
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert document["project"]["scripts"] == {"tl": "tactilelatent_tools.cli:main"}
    assert document["tool"]["setuptools"]["packages"]["find"]["where"] == ["src", "tools"]
    assert document["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_watch_status_has_finite_readonly_samples(project, monkeypatch, capsys):
    import tactilelatent_tools.commands.status as status_command

    calls = []
    monkeypatch.setattr(status_command.time, "sleep", lambda seconds: calls.append(seconds))
    before = sorted(project.rglob("*"))
    assert invoke(project, "status", "--watch", "--interval", "0.25", "--samples", "2") == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["writes"] is False for line in lines)
    assert calls == [0.25]
    assert sorted(project.rglob("*")) == before
    assert invoke(project, "status", "--watch", "--interval", "0", "--samples", "1") == 2


def test_train_last_checkpoint_is_normalized_to_run_directory(project, monkeypatch):
    run = project / "processing/runs/current"
    run.mkdir(parents=True)
    (run / "last.pt").write_text("inert fixture")
    calls = []
    fake_api(
        monkeypatch,
        "tactilelatent.training.engine",
        train_experiment=lambda *args, **kwargs: calls.append(kwargs),
    )
    assert invoke(project, "train", "--resume", str(run / "last.pt")) == 0
    assert calls[0]["resume"] == run
