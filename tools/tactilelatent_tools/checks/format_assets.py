"""Check or explicitly format maintained HTML, JavaScript, and CSS assets.

Inputs:
    --project-root selects the checkout; pinned .[dev] beautifiers are required.
Outputs:
    A report on stdout and a failing exit code for unformatted assets. Files are
    changed only with --write. Historical and generated assets are never visited.
Example:
    python -m tactilelatent_tools.checks.format_assets --project-root .
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

HTML_DEFAULTS = (
    "--profile",
    "html",
    "--indent",
    "2",
    "--max-line-length",
    "100",
    "--max-attribute-length",
    "70",
    "--format-css",
    "--format-js",
    "--indent-css",
    "2",
    "--indent-js",
    "2",
    "--ignore-case",
    "--no-entity-formatting",
)


def formatted_asset(source: str, suffix: str, *, project_root: Path | None = None) -> str:
    """Return canonical source using the pinned, language-aware beautifier."""
    if suffix == ".html":
        project_root = project_root.resolve() if project_root is not None else None
        if find_spec("djlint") is None:
            raise ModuleNotFoundError("No module named 'djlint'", name="djlint")
        # stdin prevents djLint from changing checkout files even in --reformat
        # mode. Only check_assets(write=True) may persist the returned text.
        command = [sys.executable, "-m", "djlint", "-", "--reformat", "--quiet"]
        config = project_root / "pyproject.toml" if project_root else None
        if config is not None and config.is_file():
            command.extend(("--configuration", str(config), "--prefer-configuration"))
        else:
            command.extend(HTML_DEFAULTS)
        result = subprocess.run(
            command,
            input=source,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            cwd=project_root,
        )
        return result.stdout.rstrip("\n") + "\n"
    if suffix == ".js":
        import jsbeautifier

        options = jsbeautifier.default_options()
        options.indent_size = 2
        options.end_with_newline = True
        return jsbeautifier.beautify(source, options)
    if suffix == ".css":
        import cssbeautifier

        options = cssbeautifier.default_options()
        options.indent_size = 2
        options.end_with_newline = True
        return cssbeautifier.beautify(source, options)
    raise ValueError(f"Unsupported maintained asset type: {suffix}")


def check_assets(project_root: Path, *, write: bool = False) -> list[Path]:
    """Return differing maintained assets; optionally rewrite only those files."""
    assets = project_root / "src/tactilelatent/visualization"
    changed = []
    for path in sorted(assets.iterdir()):
        if not path.is_file() or path.suffix not in {".html", ".js", ".css"}:
            continue
        source = path.read_text(encoding="utf-8")
        formatted = formatted_asset(source, path.suffix, project_root=project_root)
        if formatted != source:
            changed.append(path)
            if write:
                path.write_text(formatted, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    """Parse arguments without importing formatters until after help exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true", help="Apply formatting explicitly")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    if not (root / "src/tactilelatent/visualization").is_dir():
        parser.error("Project root must contain src/tactilelatent/visualization")
    try:
        changed = check_assets(root, write=args.write)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Asset formatting requires .[dev]: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(2, f"HTML formatting failed: {error.stderr.strip()}\n")
    for path in changed:
        print(f"{'Formatted' if args.write else 'Needs formatting'}: {path.relative_to(root)}")
    if not changed:
        print("Maintained HTML/JavaScript/CSS formatting passed.")
    return int(bool(changed) and not args.write)


if __name__ == "__main__":
    raise SystemExit(main())
