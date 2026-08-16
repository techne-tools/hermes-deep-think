"""Tests for the deep-think plugin.

Run: python -m pytest tests/ -q
(stdlib + pytest only; no network, no Hermes runtime required)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "deep_think_plugin", Path(__file__).resolve().parent.parent / "plugin" / "__init__.py"
)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class TestHandler:
    def test_valid_thought_returns_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = json.loads(plugin.deep_think_handler({"thought": "Consider option A vs B."}))
        assert result["status"] == "ok"
        assert result["recorded_chars"] == len("Consider option A vs B.")
        assert result["trace"].endswith("reasoning-trace.jsonl")

    def test_empty_thought_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = json.loads(plugin.deep_think_handler({"thought": "   "}))
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_missing_thought_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = json.loads(plugin.deep_think_handler({}))
        assert result["status"] == "error"

    def test_non_string_thought_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = json.loads(plugin.deep_think_handler({"thought": 42}))
        assert result["status"] == "error"

    def test_valid_phase_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        json.loads(plugin.deep_think_handler({"thought": "check", "phase": "verify"}))
        rows = plugin.ReasoningTrace(plugin._local_trace_dir()).tail()
        assert rows[-1]["phase"] == "verify"

    def test_invalid_phase_is_normalized_to_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        json.loads(plugin.deep_think_handler({"thought": "check", "phase": "daydream"}))
        rows = plugin.ReasoningTrace(plugin._local_trace_dir()).tail()
        assert rows[-1]["phase"] is None

    def test_result_is_always_json_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        out = plugin.deep_think_handler({"thought": "x"})
        assert isinstance(out, str)
        json.loads(out)  # must parse


# ---------------------------------------------------------------------------
# Trace: append + rotation
# ---------------------------------------------------------------------------

class TestReasoningTrace:
    def _trace(self, tmp_path, **kw):
        return plugin.ReasoningTrace(tmp_path, **kw)

    def test_append_creates_file_and_records(self, tmp_path):
        trace = self._trace(tmp_path)
        trace.append("first thought", phase="plan")
        assert trace.path.exists()
        rows = trace.tail()
        assert len(rows) == 1
        assert rows[0]["thought"] == "first thought"
        assert rows[0]["phase"] == "plan"

    def test_tail_returns_newest_last(self, tmp_path):
        trace = self._trace(tmp_path)
        for i in range(4):
            trace.append(f"thought {i}")
        rows = trace.tail(2)
        assert [r["thought"] for r in rows] == ["thought 2", "thought 3"]

    def test_tail_on_missing_file_returns_empty(self, tmp_path):
        assert self._trace(tmp_path).tail() == []

    def test_oversized_thought_is_truncated(self, tmp_path):
        trace = self._trace(tmp_path)
        trace.append("x" * (plugin._MAX_THOUGHT_CHARS + 5000))
        rows = trace.tail()
        assert rows[0]["chars"] == plugin._MAX_THOUGHT_CHARS + 5000  # original length kept
        assert len(rows[0]["thought"]) == plugin._MAX_THOUGHT_CHARS  # payload truncated

    def test_rotation_keeps_generation_count(self, tmp_path):
        trace = self._trace(tmp_path, max_bytes=200, max_rotations=3)
        for i in range(50):
            trace.append(f"generation-content-{i:03d} " + "y" * 40)
        # Live file + exactly max_rotations generations must exist.
        assert trace.path.exists()
        gens = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("reasoning-trace.jsonl.")
        )
        assert len(gens) == 3, f"expected 3 rotated generations, found: {gens}"

    def test_rotation_total_size_stays_bounded(self, tmp_path):
        trace = self._trace(tmp_path, max_bytes=200, max_rotations=3)
        for i in range(200):
            trace.append("z" * 150)
        total = sum(p.stat().st_size for p in tmp_path.iterdir())
        assert total <= 4 * 200 + 200 * 160  # live + 3 gens + slack for pending append

    def test_timestamps_are_iso_utc(self, tmp_path):
        trace = self._trace(tmp_path)
        trace.append("ts check")
        row = trace.tail()[0]
        assert plugin._TIMESTAMP_RE.match(row["ts"]), row["ts"]
        assert row["ts"].endswith("Z")

    def test_unicode_thought_survives_roundtrip(self, tmp_path):
        trace = self._trace(tmp_path)
        text = "deep thought — ünïcödé ✓ 中文 🧠"
        trace.append(text)
        assert trace.tail()[0]["thought"] == text


# ---------------------------------------------------------------------------
# register() against a stub PluginContext
# ---------------------------------------------------------------------------

class StubRegistration:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class StubContext:
    """Minimal stand-in for hermes PluginContext."""

    def __init__(self):
        self.tools = {}
        self.sections = {}

    def register_tool(self, *, name, toolset, schema, handler, check_fn=None, **kw):
        self.tools[name] = {
            "toolset": toolset, "schema": schema, "handler": handler,
            "check_fn": check_fn, "extra": kw,
        }

    def register_system_prompt_section(self, id, content, **kw):
        self.sections[id] = {"content": content, "kw": kw}


class TestRegister:
    def test_registers_tool_with_tau_bench_shape(self):
        ctx = StubContext()
        plugin.register(ctx)
        tool = ctx.tools["deep_think"]
        schema = tool["schema"]
        assert schema["name"] == "deep_think"
        assert schema["parameters"]["required"] == ["thought"]
        assert schema["parameters"]["properties"]["thought"]["type"] == "string"
        assert callable(tool["handler"])
        assert callable(tool["check_fn"]) and tool["check_fn"]()

    def test_registers_prompt_section_within_limit(self):
        ctx = StubContext()
        plugin.register(ctx)
        section = ctx.sections["deep_think"]
        assert section["kw"]["position"] == "after_memory"
        assert 0 < len(section["content"]) <= 4000

    def test_prompt_section_mentions_when_not_to_use(self):
        ctx = StubContext()
        plugin.register(ctx)
        assert "Do NOT use" in ctx.sections["deep_think"]["content"]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_declares_tool_and_section(self):
        manifest = (Path(__file__).resolve().parent.parent / "plugin" / "plugin.yaml").read_text()
        assert "provides_tools:" in manifest
        assert "- deep_think" in manifest
        assert "name: deep-think" in manifest
        assert "version:" in manifest

    def test_manifest_description_is_quoted_and_short(self):
        manifest = (Path(__file__).resolve().parent.parent / "plugin" / "plugin.yaml").read_text()
        import re
        m = re.search(r'^description: "(.+)"$', manifest, re.M)
        assert m, "description must be a double-quoted YAML string"
        assert len(m.group(1)) < 600
