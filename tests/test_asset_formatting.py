"""Verify read-only web-asset checks, explicit writes, and lazy tool help."""

import re
import subprocess
import sys
from html.parser import HTMLParser
from importlib.util import find_spec

import pytest

from tactilelatent_tools.checks import format_assets


def asset_checkout(tmp_path):
    """Create only the maintained asset location and one historical control."""
    assets = tmp_path / "src/tactilelatent/visualization"
    assets.mkdir(parents=True)
    historical = tmp_path / "legacy/v1/viewer.js"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical bytes", encoding="utf-8")
    return assets, historical


def test_asset_check_is_read_only_and_ignores_history(tmp_path, monkeypatch):
    assets, historical = asset_checkout(tmp_path)
    original = {}
    for name in ("viewer.html", "viewer.js", "viewer.css"):
        path = assets / name
        path.write_text("compact", encoding="utf-8")
        original[path] = path.read_bytes()
    (assets / "README.md").write_text("keep docs", encoding="utf-8")
    monkeypatch.setattr(format_assets, "formatted_asset", lambda *args, **kwargs: "formatted\n")

    assert format_assets.check_assets(tmp_path) == sorted(original)
    assert all(path.read_bytes() == content for path, content in original.items())
    assert historical.read_text() == "historical bytes"
    assert (assets / "README.md").read_text() == "keep docs"


def test_asset_writes_are_explicit_and_idempotent(tmp_path, monkeypatch):
    assets, historical = asset_checkout(tmp_path)
    path = assets / "viewer.html"
    path.write_text("compact", encoding="utf-8")
    monkeypatch.setattr(format_assets, "formatted_asset", lambda *args, **kwargs: "formatted\n")

    assert format_assets.check_assets(tmp_path, write=True) == [path]
    assert path.read_text() == "formatted\n"
    assert format_assets.check_assets(tmp_path, write=True) == []
    assert historical.read_text() == "historical bytes"


def test_asset_cli_reports_missing_optional_dependencies(tmp_path, monkeypatch, capsys):
    assets, _ = asset_checkout(tmp_path)
    (assets / "viewer.js").touch()

    def missing_formatter(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'jsbeautifier'")

    monkeypatch.setattr(format_assets, "formatted_asset", missing_formatter)
    with pytest.raises(SystemExit) as stopped:
        format_assets.main(["--project-root", str(tmp_path)])
    assert stopped.value.code == 2
    assert ".[dev]" in capsys.readouterr().err


def test_asset_help_needs_no_formatter_imports_or_subprocesses():
    script = """
import importlib.abc
import subprocess
import sys

class BlockFormatters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'djlint', 'jsbeautifier', 'cssbeautifier'}:
            raise AssertionError('Help imported optional formatter: ' + fullname)

def forbidden(*args, **kwargs):
    raise AssertionError('Help started a subprocess')

sys.meta_path.insert(0, BlockFormatters())
subprocess.run = forbidden
from tactilelatent_tools.checks.format_assets import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--write" in result.stdout


@pytest.mark.parametrize("suffix,package", [(".js", "jsbeautifier"), (".css", "cssbeautifier")])
def test_language_formatter_is_idempotent(suffix, package):
    pytest.importorskip(package)
    source = (
        "function read(x){return x+1;}" if suffix == ".js" else "body{color:red;background:black}"
    )
    formatted = format_assets.formatted_asset(source, suffix)
    assert formatted != source
    assert formatted.endswith("\n")
    assert format_assets.formatted_asset(formatted, suffix) == formatted


class HtmlStructure(HTMLParser):
    """Record element/attribute order and visible text, ignoring layout space."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.events = []

    def handle_starttag(self, tag, attrs):
        self.events.append(("start", tag, attrs))

    def handle_startendtag(self, tag, attrs):
        self.events.append(("empty", tag, attrs))

    def handle_endtag(self, tag):
        self.events.append(("end", tag))

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self.events.append(("text", text))

    def handle_entityref(self, name):
        self.events.append(("entity", name))

    def handle_charref(self, name):
        self.events.append(("character", name))


@pytest.mark.skipif(
    find_spec("djlint") is None, reason="HTML formatter is an optional .[dev] dependency"
)
def test_html_formatter_preserves_elements_attributes_and_text(tmp_path):
    source = (
        '<!doctype html><html lang="en"><head><title>Sensor viewer</title></head>'
        '<body><label class="check"><input id="mirror" type="checkbox" checked>'
        'Mirror &amp; compare</label><p data-source="pressure">224 &times; 224</p>'
        '<script src="viewer.js"></script></body></html>'
    )
    formatted = format_assets.formatted_asset(source, ".html", project_root=tmp_path)
    original, actual = HtmlStructure(), HtmlStructure()
    original.feed(source)
    actual.feed(formatted)
    assert actual.events == original.events
    assert format_assets.formatted_asset(formatted, ".html", project_root=tmp_path) == formatted
    assert not list(tmp_path.iterdir()), "stdin formatting must never write temporary project files"


def test_html_formatting_uses_stdin_and_explicit_configuration(tmp_path, monkeypatch):
    config = tmp_path / "pyproject.toml"
    config.write_text("[tool.djlint]\nindent = 2\n", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(format_assets, "find_spec", lambda name: object())

    def format_subprocess(command, **kwargs):
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="<p>sensor</p>")

    monkeypatch.setattr(format_assets.subprocess, "run", format_subprocess)
    assert (
        format_assets.formatted_asset("<p>sensor</p>", ".html", project_root=tmp_path)
        == "<p>sensor</p>\n"
    )
    assert observed["command"][:4] == [sys.executable, "-m", "djlint", "-"]
    assert observed["input"] == "<p>sensor</p>"
    assert observed["command"][observed["command"].index("--configuration") + 1] == str(config)
    assert observed["check"] is True
    assert observed["cwd"] == tmp_path
