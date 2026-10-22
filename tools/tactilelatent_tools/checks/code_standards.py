"""Check readable source structure and operational documentation without imports.

Inputs:
    A project root containing maintained Python, configurations and README.md.
Outputs:
    Human-readable findings on stdout; exit 1 for violations and 0 for success.
    The check reads files only, without importing research code or writing files.
Example:
    python -m tactilelatent_tools.checks.code_standards --project-root .
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

PYTHON_ROOTS = ("src", "tools", "tests", "examples")
MACHINE_PATH = re.compile(
    r"^(?:/(?:home|Users|Volumes|mnt|media|scratch|workspace|work|root|data|datasets|srv|opt)(?:/|$)|~[^/\\]*[/\\]|[A-Za-z]:[\\/]|\\\\)"
)
TEMPORARY_NAME = re.compile(
    r"^(?:tmp|temp|scratch|new|test\d+|script\d+)(?:_|$)|(?:^|_)final(?:_|$)"
)
PYTHON_NAME = re.compile(r"(?:__[a-z]+__|_?[a-z][a-z0-9]*(?:_[a-z0-9]+)*)\.py$")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+['\"][^)]*)?\)")
DOC_SECTIONS = ("Inputs", "Outputs", "Example")


@dataclass(frozen=True, order=True)
class Finding:
    """A location and actionable explanation; paths are relative to the project."""

    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


def maintained_python(root: Path) -> list[Path]:
    """Discover maintained files without walking private work or research data."""
    paths = set(root.glob("*.py"))
    for name in PYTHON_ROOTS:
        location = root / name
        if location.is_dir() and not location.is_symlink():
            paths.update(location.rglob("*.py"))
    return sorted(
        path
        for path in paths
        if "__pycache__" not in path.parts and _within_source_scope(path, root)
    )


def _within_source_scope(path: Path, root: Path) -> bool:
    """Do not read files redirected into private, historical or external trees."""
    try:
        target = path.resolve(strict=True)
        return not path.is_symlink() or any(
            target.is_relative_to(root / name) for name in PYTHON_ROOTS
        )
    except (OSError, RuntimeError):
        return False


def _javascript_findings(root: Path) -> tuple[set[Path], list[Finding]]:
    """Require readable headers on maintained browser scripts, without parsing JS."""
    scripts = set(root.glob("*.js"))
    for name in ("src", "tools", "examples"):
        location = root / name
        if location.is_dir() and not location.is_symlink():
            scripts.update(location.rglob("*.js"))
    scripts = {
        path
        for path in scripts
        if not {"assets", "vendor", "node_modules"}.intersection(path.relative_to(root).parts)
        and _within_source_scope(path, root)
    }
    findings = []
    for path in sorted(scripts):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError) as error:
            findings.append(Finding(relative, 1, "javascript-source", str(error)))
            continue
        header = re.match(r"\s*/\*\*(.*?)\*/", source, flags=re.DOTALL)
        if not header:
            findings.append(
                Finding(
                    relative,
                    1,
                    "javascript-docstring",
                    "Start the script with a JSDoc purpose, Inputs:, Outputs: and Example: header.",
                )
            )
            continue
        document = re.sub(r"(?m)^[ \t]*\* ?", "", header.group(1)).strip()
        missing = _missing_doc_sections(document)
        if not document or document.splitlines()[0].split(":", 1)[0] in DOC_SECTIONS or missing:
            findings.append(
                Finding(
                    relative,
                    1,
                    "javascript-docstring",
                    "Document the script purpose and nonempty Inputs:, Outputs: and Example: sections.",
                )
            )
        if TEMPORARY_NAME.search(path.stem):
            findings.append(
                Finding(
                    relative,
                    1,
                    "descriptive-name",
                    "Use a descriptive script name without temporary/final labels.",
                )
            )
        if path.parent == root:
            findings.append(
                Finding(
                    relative,
                    1,
                    "source-location",
                    "Place maintained browser scripts in a descriptive source/tools/example folder.",
                )
            )
    return scripts, findings


def _missing_doc_sections(docstring: str) -> list[str]:
    return [
        section
        for section in DOC_SECTIONS
        if not re.search(rf"(?m)^{section}:[ \t]*(?:\S[^\n]*|\n[ \t]+\S)", docstring)
    ]


def _symlink_findings(root: Path) -> list[Finding]:
    findings = []
    for path in root.glob("*.py"):
        if path.is_symlink() and not _within_source_scope(path, root):
            findings.append(
                Finding(
                    path.name,
                    1,
                    "maintained-symlink",
                    "A maintained Python file must not redirect outside the source folders.",
                )
            )
    for name in (*PYTHON_ROOTS, "configs"):
        location = root / name
        if not location.exists() and not location.is_symlink():
            continue
        if location.is_symlink():
            findings.append(
                Finding(
                    name,
                    1,
                    "maintained-symlink",
                    "The maintained source/configuration root must be a real directory, not an archive/private shortcut.",
                )
            )
            continue
        for directory, folders, files in os.walk(location, followlinks=False):
            folders[:] = [folder for folder in folders if folder != "__pycache__"]
            for entry in (*folders, *files):
                path = Path(directory) / entry
                if not path.is_symlink():
                    continue
                try:
                    target = path.resolve(strict=True)
                    allowed = ("configs",) if name == "configs" else PYTHON_ROOTS
                    valid = any(target.is_relative_to(root / item) for item in allowed)
                except (OSError, RuntimeError):
                    valid = False
                if not valid:
                    findings.append(
                        Finding(
                            path.relative_to(root).as_posix(),
                            1,
                            "maintained-symlink",
                            "Maintained code/configuration must not redirect into private work, archives or external files.",
                        )
                    )
    return findings


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[0] in ("src", "tools"):
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_names(node: ast.AST, path: Path, root: Path) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.Call):
        function = ast.unparse(node.func).rsplit(".", 1)[-1]
        if function in {"import_module", "__import__", "run_module"} and node.args:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                return [argument.value]
        return []
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        package = _module_name(path, root).split(".")
        if path.name != "__init__.py":
            package.pop()
        keep = len(package) - node.level + 1
        module = ".".join(package[: max(keep, 0)] + ([module] if module else []))
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    operands = (test.left, *test.comparators)
    return any(
        isinstance(value, ast.Name) and value.id == "__name__" for value in operands
    ) and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in operands)


def _source_findings(path: Path, root: Path, source: str, tree: ast.Module) -> list[Finding]:
    relative = path.relative_to(root).as_posix()
    findings = []

    def report(node: ast.AST, rule: str, message: str) -> None:
        findings.append(Finding(relative, getattr(node, "lineno", 1), rule, message))

    if not ast.get_docstring(tree):
        report(tree, "module-docstring", "Start the module with a purpose docstring.")
    if not PYTHON_NAME.fullmatch(path.name) or TEMPORARY_NAME.search(path.stem):
        report(
            tree,
            "descriptive-name",
            "Use a descriptive lower_snake_case filename, without temporary/final labels.",
        )
    if path.parent == root:
        report(
            tree, "source-location", "Place maintained Python in src/, tools/, tests/ or examples/."
        )

    in_core = relative.startswith("src/tactilelatent/")
    in_model = relative.startswith("src/tactilelatent/models/")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and MACHINE_PATH.match(node.value)
        ):
            report(
                node,
                "portable-path",
                "Replace the machine-specific absolute path with an argument or configuration value.",
            )
        for name in _import_names(node, path, root):
            if "ai_work" in name.split("."):
                report(
                    node,
                    "private-dependency",
                    "Maintained code must not import disposable ai_work modules.",
                )
            elif in_core and name.split(".")[0] == "tactilelatent_tools":
                report(
                    node, "dependency-direction", "Research code must not import operational tools."
                )
            elif in_model and any(
                name == module or name.startswith(module + ".")
                for module in (
                    "tactilelatent.datasets",
                    "tactilelatent.dataset_management",
                    "tactilelatent.training",
                )
            ):
                report(
                    node,
                    "model-boundary",
                    "Models receive generic contracts; dataset management and training belong outside models/.",
                )
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            if getattr(node, "lineno", None) == body[0].lineno:
                report(
                    node,
                    "one-statement-per-line",
                    "Put the compound statement body on a separate indented line.",
                )
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            executable = function.rsplit(".", 1)[-1] in {
                "run_path",
                "run_module",
                "import_module",
                "__import__",
                "spec_from_file_location",
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
                "system",
                "execv",
                "execve",
            }
            strings = [
                value.value
                for value in ast.walk(node)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            if executable and any("ai_work" in re.split(r"[/\\.]", value) for value in strings):
                report(
                    node,
                    "private-dependency",
                    "Maintained execution must not depend on a disposable ai_work program.",
                )
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.OP and token.string == ";":
            findings.append(
                Finding(
                    relative,
                    token.start[0],
                    "one-statement-per-line",
                    "Split semicolon-separated statements onto separate lines.",
                )
            )
    return findings


def _module_path(module: str, root: Path) -> Path | None:
    relative = Path(*module.split("."))
    for location in ("src", "tools", ""):
        for candidate in (
            root / location / relative.with_suffix(".py"),
            root / location / relative / "__init__.py",
        ):
            if candidate.is_file():
                return candidate
    return None


def _entrypoints(root: Path, trees: dict[Path, ast.Module]) -> tuple[set[Path], list[Finding]]:
    """Read declared commands as data; never import command implementations."""
    entries = {
        path for path, tree in trees.items() if any(_is_main_guard(node) for node in tree.body)
    }
    findings = []
    declarations = []
    project_file = root / "pyproject.toml"
    if project_file.is_file():
        try:
            metadata = tomllib.loads(project_file.read_text(encoding="utf-8"))
            declarations.extend(
                (project_file, 1, str(target).split(":", 1)[0])
                for kind in ("scripts", "gui-scripts")
                for target in metadata.get("project", {}).get(kind, {}).values()
            )
        except (tomllib.TOMLDecodeError, OSError) as error:
            findings.append(Finding("pyproject.toml", 1, "project-metadata", str(error)))
    for path, tree in trees.items():
        if path.relative_to(root).as_posix() != "tools/tactilelatent_tools/cli.py":
            continue
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(isinstance(target, ast.Name) and target.id == "COMMANDS" for target in targets):
                try:
                    commands = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    findings.append(
                        Finding(
                            path.relative_to(root).as_posix(),
                            node.lineno,
                            "command-menu",
                            "COMMANDS must be a literal mapping so help and discovery stay inspectable.",
                        )
                    )
                    continue
                if not isinstance(commands, dict) or not all(
                    isinstance(value, str) for value in commands.values()
                ):
                    findings.append(
                        Finding(
                            path.relative_to(root).as_posix(),
                            node.lineno,
                            "command-menu",
                            "COMMANDS must map command names to module strings.",
                        )
                    )
                    continue
                declarations.extend((path, node.lineno, module) for module in commands.values())
    for declaration, line, module in declarations:
        path = _module_path(module, root)
        if path is None or path not in trees:
            findings.append(
                Finding(
                    declaration.relative_to(root).as_posix(),
                    line,
                    "entrypoint-target",
                    f"Declared operational module must resolve to maintained Python: {module}",
                )
            )
        else:
            entries.add(path)
    return entries, findings


def _entrypoint_findings(
    root: Path, trees: dict[Path, ast.Module], browser_scripts: set[Path]
) -> list[Finding]:
    entries, findings = _entrypoints(root, trees)
    entries.update(browser_scripts)
    for path in sorted(entries):
        if path not in trees:
            continue
        docstring = ast.get_docstring(trees[path]) or ""
        missing = _missing_doc_sections(docstring)
        if missing:
            findings.append(
                Finding(
                    path.relative_to(root).as_posix(),
                    1,
                    "script-docstring",
                    f"Document nonempty {' / '.join(section + ':' for section in missing)} sections in the module docstring.",
                )
            )

    readme = root / "README.md"
    if not readme.is_file():
        return [
            *findings,
            Finding(
                "README.md",
                1,
                "command-index",
                "Add a root README with links to maintained executable modules.",
            ),
        ]
    links = set()
    text = readme.read_text(encoding="utf-8")
    visible = re.sub(
        r"<!--.*?-->", lambda match: re.sub(r"[^\n]", " ", match.group()), text, flags=re.DOTALL
    )
    lines = visible.splitlines(keepends=True)
    fence = None
    for index, line in enumerate(lines):
        marker = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if marker and fence is None:
            fence = marker.group(1)
        elif (
            marker
            and fence is not None
            and marker.group(1)[0] == fence[0]
            and len(marker.group(1)) >= len(fence)
        ):
            fence = None
            lines[index] = re.sub(r"[^\n]", " ", line)
        if fence is not None:
            lines[index] = re.sub(r"[^\n]", " ", line)
    for match in MARKDOWN_LINK.finditer("".join(lines)):
        target = next(value for value in match.groups() if value is not None)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        normalized = Path(unquote(parsed.path)).as_posix().removeprefix("./")
        links.add(normalized)
        if (
            normalized.endswith((".py", ".js"))
            and normalized.split("/")[0] in PYTHON_ROOTS
            and not (root / normalized).is_file()
        ):
            findings.append(
                Finding(
                    "README.md",
                    text.count("\n", 0, match.start()) + 1,
                    "stale-command-link",
                    f"Remove or repair the missing maintained script link: {normalized}",
                )
            )
    for path in sorted(entries):
        relative = path.relative_to(root).as_posix()
        if relative not in links:
            findings.append(
                Finding(
                    "README.md",
                    1,
                    "command-index",
                    f"Add a Markdown link to the maintained entrypoint: {relative}",
                )
            )
    return findings


def _config_findings(root: Path) -> list[Finding]:
    findings = []
    location = root / "configs"
    if location.is_symlink():
        return findings
    for path in sorted(location.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            try:
                if not path.resolve(strict=True).is_relative_to(root / "configs"):
                    continue
            except (OSError, RuntimeError):
                continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            findings.append(Finding(relative, 1, "configuration", str(error)))
            continue
        pending = [("", value)]
        while pending:
            key, value = pending.pop()
            if isinstance(value, str) and MACHINE_PATH.match(value):
                findings.append(
                    Finding(
                        relative,
                        1,
                        "portable-path",
                        f"Replace the machine-specific absolute path at {key or '<root>'} with a portable project-relative setting.",
                    )
                )
            elif isinstance(value, dict):
                pending.extend(
                    (f"{key}.{name}".lstrip("."), child) for name, child in value.items()
                )
            elif isinstance(value, list):
                pending.extend((f"{key}[{index}]", child) for index, child in enumerate(value))
    return findings


def check_project(project_root: Path | str) -> list[Finding]:
    """Return deterministic findings without imports, subprocesses or mutations."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        return [Finding(".", 1, "project-root", "The requested project root is not a directory.")]
    findings = _symlink_findings(root)
    trees = {}
    for path in maintained_python(root):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
            trees[path] = tree
            findings.extend(_source_findings(path, root, source, tree))
        except (SyntaxError, UnicodeError, OSError, tokenize.TokenError) as error:
            findings.append(
                Finding(relative, getattr(error, "lineno", 1) or 1, "python-source", str(error))
            )
    findings.extend(_config_findings(root))
    browser_scripts, javascript_findings = _javascript_findings(root)
    findings.extend(javascript_findings)
    findings.extend(_entrypoint_findings(root, trees, browser_scripts))
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    """Run the read-only check against an explicit checkout or the current folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project checkout to inspect; defaults to the working directory.",
    )
    args = parser.parse_args(argv)
    findings = check_project(args.project_root)
    for finding in findings:
        print(finding)
    if findings:
        print(f"Code standards: {len(findings)} finding(s).")
        return 1
    print("Code standards: passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
