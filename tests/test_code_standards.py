"""Exercise maintenance checks against isolated fixtures without research imports."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tactilelatent_tools.checks.code_standards import check_project, main

SCRIPT_DOC = '''"""Describe the operation.

Inputs:
    Explicit arguments from the command line.
Outputs:
    A short printed result; no persistent writes.
Example:
    python -m example --help
"""
'''

JAVASCRIPT_DOC = """/**
 * Control the native sensor viewer in the browser.
 *
 * Inputs:
 *   User selections and read-only viewer API responses.
 * Outputs:
 *   Updated page contents and event handlers.
 * Example:
 *   tl view --port 8080
 */
"""


def write(root, relative, content):
    """Write a fixture, keeping every test artifact under pytest's temporary root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path):
    write(tmp_path, "README.md", "# Fixture\n")
    write(tmp_path, "pyproject.toml", '[project]\nname = "fixture"\n')
    write(tmp_path, "src/tactilelatent/__init__.py", '"""Fixture package."""\n')
    return tmp_path


def rules(root):
    return {finding.rule for finding in check_project(root)}


def test_readable_library_and_portable_paths_pass(project):
    write(
        project,
        "src/tactilelatent/models/attention.py",
        '''
        """A shared attention implementation with generic inputs."""
        from pathlib import Path

        def paths(project_root):
            return Path(project_root) / "processing" / "features"
    ''',
    )
    write(project, "configs/model.json", '{"weights": "processing/models/backbone.pt"}')
    assert check_project(project) == []


def test_docstring_required_for_every_maintained_python_module(project):
    for location in (
        "src/tactilelatent/core.py",
        "tools/inspect_data.py",
        "tests/test_masks.py",
        "examples/attention.py",
    ):
        write(project, location, "value = 1\n")
    findings = [finding for finding in check_project(project) if finding.rule == "module-docstring"]
    assert {finding.path for finding in findings} == {
        "src/tactilelatent/core.py",
        "tools/inspect_data.py",
        "tests/test_masks.py",
        "examples/attention.py",
    }


def test_archives_private_outputs_and_research_assets_are_not_scanned(project):
    for directory in ("legacy/v1", "ai_work/scratch/task", "processing", "datasets", ".venv"):
        write(project, f"{directory}/test2_final_v3.py", "this is invalid python\n")
    assert check_project(project) == []


@pytest.mark.parametrize(
    "filename",
    ["test2_final_v3.py", "tmp_report.py", "script12.py", "analysis_final.py", "MixedCase.py"],
)
def test_descriptive_names_are_required(project, filename):
    write(project, f"tools/{filename}", '"""A temporary-looking module."""\n')
    assert "descriptive-name" in rules(project)


def test_root_launchers_belong_in_descriptive_maintained_folders(project):
    write(project, "evaluate.py", '"""A misplaced command."""\n')
    assert "source-location" in rules(project)


def test_constant_signal_text_and_one_line_compound_statements(project):
    write(
        project,
        "src/tactilelatent/helpers.py",
        '''
        """Detect statements without interpreting quoted semicolons as code."""
        message = "alpha; beta"
        first = 1; second = 2
        if first: second = 3
    ''',
    )
    findings = [
        finding for finding in check_project(project) if finding.rule == "one-statement-per-line"
    ]
    assert [finding.line for finding in findings] == [4, 5]


def test_machine_paths_are_rejected_but_routes_and_os_interfaces_are_allowed(project):
    machine_path = "/" + "home/researcher/data"
    write(
        project,
        "src/tactilelatent/paths.py",
        f'"""Path fixture."""\nvalue = {machine_path!r}\nroute = "/api/models"\nproc = "/proc/self/status"\n',
    )
    write(
        project, "configs/datasets.json", '{"artifacts": [{"path": "C:\\\\research\\\\samples"}]}'
    )
    findings = [finding for finding in check_project(project) if finding.rule == "portable-path"]
    assert len(findings) == 2
    assert {finding.path for finding in findings} == {
        "src/tactilelatent/paths.py",
        "configs/datasets.json",
    }


@pytest.mark.parametrize("prefix", ["data", "opt", "srv", "root", "work", "Volumes"])
def test_other_machine_artifact_roots_are_rejected(project, prefix):
    value = "/" + prefix + "/research/input"
    write(project, "src/tactilelatent/paths.py", f'"""Path fixture."""\nartifact = {value!r}\n')
    assert "portable-path" in rules(project)


def test_literal_home_paths_are_rejected(project):
    value = "~" + "/research/input"
    write(project, "src/tactilelatent/paths.py", f'"""Path fixture."""\nartifact = {value!r}\n')
    assert "portable-path" in rules(project)


@pytest.mark.parametrize(
    "source",
    [
        "from tactilelatent_tools import cli",
        "import tactilelatent_tools.commands.train",
    ],
)
def test_core_cannot_depend_on_operational_tools(project, source):
    write(project, "src/tactilelatent/core.py", f'"""Research module."""\n{source}\n')
    assert "dependency-direction" in rules(project)


def test_literal_dynamic_imports_cannot_bypass_dependency_direction(project):
    write(
        project,
        "src/tactilelatent/core.py",
        '"""Research module."""\nimport importlib\nmodule = importlib.import_module("tactilelatent_tools.cli")\n',
    )
    assert "dependency-direction" in rules(project)


@pytest.mark.parametrize(
    "source",
    [
        "from ..datasets import sources",
        "from .. import training",
        "from tactilelatent.dataset_management import registry",
    ],
)
def test_model_boundary_handles_absolute_and_relative_imports(project, source):
    write(project, "src/tactilelatent/models/attention.py", f'"""Model fixture."""\n{source}\n')
    assert "model-boundary" in rules(project)


@pytest.mark.parametrize(
    "source",
    [
        "import ai_work.scratch.temporary",
        "from ai_work.scratch import helper",
        'import runpy\nrunpy.run_path("ai_work/scratch/task/program.py")',
        'import subprocess\nsubprocess.run(["python", "ai_work/scratch/task/program.py"])',
    ],
)
def test_disposable_programs_cannot_be_imported_or_executed(project, source):
    write(project, "tools/operation.py", f'"""Operational fixture."""\n{source}\n')
    assert "private-dependency" in rules(project)


def test_main_guards_require_structured_docs_and_root_readme_links(project):
    script = "examples/run_attention.py"
    write(
        project,
        script,
        '"""Run the example."""\nif __name__ == "__main__":\n    print("example")\n',
    )
    assert {"script-docstring", "command-index"} <= rules(project)
    write(project, script, SCRIPT_DOC + 'if __name__ == "__main__":\n    print("example")\n')
    write(project, "README.md", f"# Commands\n[Run attention](<{script}#L1>)\n")
    assert check_project(project) == []


def test_empty_doc_sections_do_not_pass_by_matching_the_next_heading(project):
    script = "examples/run_attention.py"
    source = SCRIPT_DOC.replace("    Explicit arguments from the command line.\n", "")
    write(project, script, source + 'if __name__ == "__main__":\n    print("example")\n')
    write(project, "README.md", f"[Example]({script})\n")
    findings = [finding for finding in check_project(project) if finding.rule == "script-docstring"]
    assert len(findings) == 1
    assert "Inputs:" in findings[0].message


def test_entrypoints_are_read_from_metadata_and_menu_without_importing(project):
    cli = "tools/tactilelatent_tools/cli.py"
    command = "tools/tactilelatent_tools/commands/evaluate.py"
    write(project, "pyproject.toml", '[project.scripts]\ntl = "tactilelatent_tools.cli:main"\n')
    write(
        project,
        cli,
        SCRIPT_DOC
        + 'COMMANDS = {"evaluate": "tactilelatent_tools.commands.evaluate"}\nraise RuntimeError("MUST NEVER IMPORT")\n',
    )
    write(project, command, SCRIPT_DOC + 'raise RuntimeError("MUST NEVER IMPORT")\n')
    findings = [finding for finding in check_project(project) if finding.rule == "command-index"]
    assert len(findings) == 2
    write(project, "README.md", f"[CLI]({cli})\n[Evaluate]({command})\n")
    assert check_project(project) == []


def test_annotated_command_maps_still_discover_handlers(project):
    cli = "tools/tactilelatent_tools/cli.py"
    command = "tools/tactilelatent_tools/commands/evaluate.py"
    write(
        project,
        cli,
        SCRIPT_DOC
        + 'COMMANDS: dict[str, str] = {"evaluate": "tactilelatent_tools.commands.evaluate"}\n',
    )
    write(project, command, '"""A handler missing operational documentation."""\n')
    write(project, "README.md", f"[Evaluate]({command})\n")
    assert "script-docstring" in rules(project)


@pytest.mark.parametrize("wrapper", ["```markdown\n{}\n```\n", "<!-- {} -->\n"])
def test_hidden_or_fenced_links_do_not_satisfy_the_visible_command_index(project, wrapper):
    script = "examples/run_attention.py"
    write(project, script, SCRIPT_DOC + 'if __name__ == "__main__":\n    print("example")\n')
    write(project, "README.md", wrapper.format(f"[Example]({script})"))
    assert "command-index" in rules(project)


def test_symlinked_private_source_is_rejected_without_reading_it(project):
    private = write(
        project,
        "ai_work/scratch/task/program.py",
        "invalid Python deliberately outside the checking scope\n",
    )
    destination = project / "src/tactilelatent/external.py"
    destination.symlink_to(private)
    assert "maintained-symlink" in rules(project)
    assert "python-source" not in rules(project)


@pytest.mark.parametrize("name", ["tools", "configs"])
def test_symlinked_roots_do_not_traverse_archived_code_or_config(project, name):
    archive = project / "legacy/v1" / name
    write(project, f"legacy/v1/{name}/invalid.py", "not valid Python\n")
    write(project, f"legacy/v1/{name}/invalid.json", "not valid JSON\n")
    (project / name).symlink_to(archive, target_is_directory=True)
    assert rules(project) == {"maintained-symlink"}


def test_internal_source_file_links_are_allowed(project):
    source = write(project, "src/tactilelatent/shared.py", '"""A shared maintained module."""\n')
    (project / "src/tactilelatent/alias.py").symlink_to(source)
    assert check_project(project) == []


def test_private_command_declarations_cannot_escape_maintained_sources(project):
    write(project, "ai_work/scratch/program.py", SCRIPT_DOC)
    write(
        project,
        "tools/tactilelatent_tools/cli.py",
        SCRIPT_DOC + 'COMMANDS = {"private": "ai_work.scratch.program"}\n',
    )
    assert "entrypoint-target" in rules(project)


def test_browser_scripts_require_headers_and_visible_readme_links(project):
    script = "src/tactilelatent/visualization/viewer.js"
    write(project, script, "const enabled = true;\n")
    assert {"javascript-docstring", "command-index"} <= rules(project)
    write(project, script, JAVASCRIPT_DOC + "const enabled = true;\n")
    write(project, "README.md", f"[Viewer controls]({script})\n")
    assert check_project(project) == []


def test_browser_headers_require_nonempty_sections(project):
    script = "src/tactilelatent/visualization/viewer.js"
    source = JAVASCRIPT_DOC.replace(
        " *   User selections and read-only viewer API responses.\n", ""
    )
    write(project, script, source + "const enabled = true;\n")
    write(project, "README.md", f"[Viewer controls]({script})\n")
    assert "javascript-docstring" in rules(project)


def test_packaged_vendor_assets_are_not_treated_as_maintained_browser_scripts(project):
    write(project, "src/tactilelatent/visualization/assets/vendor.js", "external browser asset\n")
    assert check_project(project) == []


def test_deleted_browser_script_links_are_reported(project):
    write(project, "README.md", "[Viewer](src/tactilelatent/visualization/deleted.js)\n")
    assert "stale-command-link" in rules(project)


def test_renames_and_deletions_invalidate_the_command_index(project):
    old = "examples/old_attention.py"
    new = "examples/attention_walkthrough.py"
    write(project, old, SCRIPT_DOC + 'if __name__ == "__main__":\n    print("example")\n')
    write(project, "README.md", f"[Example]({old})\n")
    assert check_project(project) == []
    (project / old).rename(project / new)
    assert {"stale-command-link", "command-index"} <= rules(project)
    write(project, "README.md", f"[Example]({new})\n")
    assert check_project(project) == []
    (project / new).unlink()
    assert "stale-command-link" in rules(project)


def test_missing_declared_handler_is_reported(project):
    write(
        project,
        "tools/tactilelatent_tools/cli.py",
        SCRIPT_DOC + 'COMMANDS = {"data": "tactilelatent_tools.commands.missing"}\n',
    )
    assert "entrypoint-target" in rules(project)


def test_invalid_python_and_configuration_produce_readable_findings(project):
    write(project, "src/tactilelatent/broken.py", "def invalid(:\n")
    write(project, "configs/model.json", '{"width":')
    assert {"python-source", "configuration"} <= rules(project)


def test_main_reports_an_exit_status_and_does_not_write(project, capsys):
    before = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert main(["--project-root", str(project)]) == 0
    assert "passed" in capsys.readouterr().out
    after = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert before == after
    write(project, "src/tactilelatent/missing_doc.py", "value = 1\n")
    assert main(["--project-root", str(project)]) == 1
    assert "module-docstring" in capsys.readouterr().out


def test_check_is_standard_library_only_in_an_isolated_python_process(project):
    tools = Path(__file__).resolve().parents[1] / "tools"
    program = """
import importlib.abc
import sys
class NoResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "tactilelatent", "requests", "httpx"}:
            raise RuntimeError("forbidden import: " + fullname)
sys.meta_path.insert(0, NoResearchImports())
from tactilelatent_tools.checks.code_standards import main
raise SystemExit(main(["--project-root", sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(project)],
        env={**os.environ, "PYTHONPATH": str(tools)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


def test_maintained_checkout_satisfies_code_standards():
    project_root = Path(__file__).resolve().parents[1]
    assert check_project(project_root) == []
