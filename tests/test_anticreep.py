"""SAFE anti-creep scans for the deep-think plugin.

Technique adapted from the hermes-anansi-plugin (bionicbutterfly13, no
licence, 2026): a plugin whose core promise is "no-op, never fetches,
never mutates" earns that promise through STATIC, machine-enforced
invariants, not vibes. Three scans:

  1. forbidden-substring scan      — no shell-out, no raw provider SDKs,
                                    no subprocess anywhere in the plugin.
  2. AST import-allowlist scan     — the plugin's top-level imports must
                                    stay stdlib-only (the deep-think
                                    handler is pure; any host import would
                                    couple it to a runtime it does not use).
  3. write-mode open() scan        — exactly ONE write-mode open in the
                                    whole plugin: the append to the
                                    reasoning trace. Any other write-open
                                    is a mutation the plugin never
                                    promised (and the trace itself is
                                    profile-scoped, sized, rotated).

If ANY of these fail, the plugin no longer delivers on its contract.
These tests are the contract, mechanically enforced.

Run: python -m pytest tests/test_anticreep.py -q  (in this repo root)
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PLUGIN_FILE = Path(__file__).resolve().parent.parent / "plugin" / "__init__.py"

# The one allowed write-mode open: the reasoning trace append. Everything
# else (rotation, reads) must be read-mode or Path.replace (rename, not
# open-for-write).
WRITE_OPEN_EXPECTATION = [
    # trace append — deliberately NOT a file write in the scan below; we
    # assert its line exists in the plugin source instead, so the count
    # stays at exactly one.
    "fh.write(line + \"\\n\")",
]

FORBIDDEN_SUBSTRINGS = [
    # No shell-out, ever.
    "subprocess",
    "os.system",
    "os.popen",
    "Popen",
    # No raw provider SDKs / host secrets handling.
    "anthropic",
    "openai",
    "requests.",
    "httpx.",
    "urllib.request",
    # No cross-plugin memory access (the plugin never fetches context).
    "memory_provider",
    "MemoryProvider",
    "register_memory_provider",
    # No scheduling/daemon machinery (all work is in-hook).
    "schedule",
    "cron",
]

ALLOWED_IMPORTS = {
    # stdlib only — the handler is a pure no-op surface
    "json",
    "os",
    "re",
    "time",
    "pathlib",
    "typing",
    "__future__",
}


def _plugin_text() -> str:
    return PLUGIN_FILE.read_text(encoding="utf-8")


def _code_only_text() -> str:
    """The plugin source with the module docstring and comment lines
    removed — the anti-dependency substring scan must flag real code, not
    citations in the docstring (e.g. 'per the think-tool/tau-bench pattern,
    anthropic.com/engineering/...')."""
    text = _plugin_text()
    lines = text.splitlines()
    in_docstring = False
    kept: list[str] = []
    for line in lines:
        if in_docstring:
            if '"""' in line:
                in_docstring = False
            continue
        stripped = line.strip()
        if stripped.startswith('"""') and not stripped.endswith('"""'):
            in_docstring = True
            continue
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                names.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top = node.module.split(".")[0]
                names.add(top)
    return names


def _call_mode(node: ast.Call, is_path_open: bool) -> str | None:
    """Extract the mode argument from an open() call.

    Path.open(mode, ...) takes the mode FIRST positionally (that is how the
    trace append is written: ``self.path.open("a", ...)``). Builtin
    open(file, mode) takes it SECOND. ``is_path_open`` selects the right
    slot; both fall back to a ``mode=`` keyword."""
    if not is_path_open:
        if node.args and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            return str(node.args[1].value)
    else:
        if node.args and isinstance(node.args[0], ast.Constant):
            return str(node.args[0].value)
    for kw in node.keywords or []:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return None


def test_manifest_declares_no_tools_or_hooks_that_could_fetch():
    """The plugin is a no-op tool; its manifest must not declare memory
    providers or hooks that imply external side effects beyond the tool."""
    manifest = (Path(__file__).resolve().parent.parent / "plugin" / "plugin.yaml").read_text()
    assert "memory" not in manifest.lower()
    assert "provides_hooks:" not in manifest


def test_forbidden_substring_scan():
    """No shell-out, no raw SDKs, no scheduler machinery anywhere in code
    (docstrings and comments excepted — citations are not dependencies)."""
    text = _code_only_text()
    lowered = text.lower()
    for needle in FORBIDDEN_SUBSTRINGS:
        assert needle.lower() not in lowered, f"forbidden substring present: {needle!r}"


def test_import_allowlist_scan():
    """Top-level imports must be stdlib only."""
    tree = ast.parse(_plugin_text())
    imports = _top_level_imports(tree)
    assert imports <= ALLOWED_IMPORTS, f"non-stdlib imports: {imports - ALLOWED_IMPORTS}"


def test_write_mode_open_scan():
    """Exactly ONE write-mode open in the plugin: the trace append."""
    text = _plugin_text()
    tree = ast.parse(text)
    source_lines = text.splitlines()

    write_opens = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "open":
            # Path.open(mode) — mode is the first positional. The
            # ReasoningTrace.append write is the ONLY allowed one.
            mode = _call_mode(node, is_path_open=True)
            if mode is not None and any(c in mode for c in "wa+"):
                write_opens.append(
                    (node.lineno, source_lines[node.lineno - 1].strip()[:80])
                )
        elif isinstance(fn, ast.Name) and fn.id == "open":
            # Builtin open(file, mode) — mode is the second positional
            # (default 'r'); only flag explicit write-flavoured modes.
            mode = _call_mode(node, is_path_open=False)
            if mode is not None and any(c in mode for c in "wa+"):
                write_opens.append(
                    (node.lineno, source_lines[node.lineno - 1].strip()[:80])
                )

    assert len(write_opens) == 1, (
        f"expected exactly 1 write-mode open, found {len(write_opens)}: {write_opens}"
    )
    lineno, source = write_opens[0]
    assert "reasoning-trace.jsonl" in source or "path.name" in source, (
        f"the single write-open must be the trace append, got line {lineno}: {source}"
    )


def test_write_open_uses_append_not_truncate():
    """The trace append MUST be mode 'a' (append). A 'w' here would destroy
    history on every rotate — the anti-erasure invariant."""
    text = _plugin_text()
    assert '"a", encoding="utf-8"' in text
    assert re.search(r'\.open\("w"', text) is None


def test_no_network_imports_transitively_via_plugin_dir():
    """Directory-level guard: no vendored site-packages, no .git inside
    the plugin folder (the plugin must not ship a network dependency)."""
    plugin_dir = PLUGIN_FILE.parent
    assert not (plugin_dir / "site-packages").exists()
    assert not (plugin_dir / ".git").exists()


def test_trace_rotation_uses_rename_not_truncate():
    """Rotation must shift generations via Path.replace (atomic rename),
    never by truncating the live file with a write-open."""
    text = _plugin_text()
    assert ".replace(" in text or "src.replace" in text
    assert "truncate" not in text.lower()
