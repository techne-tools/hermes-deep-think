"""deep-think — a no-op reasoning outlet for Hermes Agent.

Exposes a ``deep_think`` tool the model can call with its working thoughts.
The handler acknowledges and records; it never fetches, mutates, or decides.
This is the τ-Bench / Anthropic "think tool" pattern (arXiv:2406.12045,
anthropic.com/engineering/claude-think-tool), under the community name
"deep_think" popularized by @_can1357 (Aug 2026).

The reasoning trace is append-only, JSONL, size-capped with rotation, and
lives under the plugin's profile-scoped data directory.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Tool schema (τ-Bench shape: a single required `thought` string)
# ---------------------------------------------------------------------------

DEEP_THINK_SCHEMA: dict[str, Any] = {
    "name": "deep_think",
    "description": (
        "Use this tool to think about something. It will not obtain new "
        "information or change anything — it just records the thought and "
        "returns an acknowledgement. Use it when complex reasoning is "
        "needed: before acting on a non-obvious plan, after surprising tool "
        "results, when checking assumptions mid-task, or when weighing "
        "trade-offs. It is a scratchpad, not a search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "Your working thoughts: assumptions, options considered, "
                    "checks to perform, conclusions and their rationale."
                ),
            },
            "phase": {
                "type": "string",
                "enum": ["plan", "verify", "reflect", "decide"],
                "description": (
                    "Optional tag for what this thought is for: plan = before "
                    "acting, verify = checking results/assumptions, reflect = "
                    "after surprising outcomes, decide = weighing options."
                ),
            },
        },
        "required": ["thought"],
    },
}

_TOOLSET = "deep_think"

# ---------------------------------------------------------------------------
# Reasoning trace (append-only JSONL, size-capped, rotated)
# ---------------------------------------------------------------------------

_MAX_TRACE_BYTES = 1 * 1024 * 1024  # 1 MiB per file
_MAX_ROTATIONS = 3  # trace.log + .1 + .2 + .3  →  4 MiB ceiling
_MAX_THOUGHT_CHARS = 100_000  # guard against pathological payloads

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class ReasoningTrace:
    """Append-only JSONL trace with rotation. Safe under concurrent calls."""

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int = _MAX_TRACE_BYTES,
        max_rotations: int = _MAX_ROTATIONS,
        now: Any | None = None,
    ) -> None:
        self._dir = Path(directory)
        self._max_bytes = max_bytes
        self._max_rotations = max_rotations
        self._now = now or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z")

    @property
    def path(self) -> Path:
        return self._dir / "reasoning-trace.jsonl"

    def append(self, thought: str, *, phase: str = "", session: str = "") -> dict[str, Any]:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": self._now(),
            "phase": phase or None,
            "session": session or None,
            "chars": len(thought),
            "thought": thought[:_MAX_THOUGHT_CHARS],
        }
        line = json.dumps(record, ensure_ascii=False)
        self._rotate_if_needed(len(line) + 1)
        with self._dir.joinpath(self.path.name).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self._max_bytes:
            return
        # Shift generations up: .{i-1} → .{i} for i in (max_rotations-1 … 1],
        # then trace → .0. Keeps the live file plus max_rotations generations;
        # the oldest is overwritten (i.e. dropped).
        for i in range(self._max_rotations - 1, 0, -1):
            src = self._dir / f"{self.path.name}.{i - 1}"
            dst = self._dir / f"{self.path.name}.{i}"
            if src.exists():
                src.replace(dst)
        self.path.replace(self._dir / f"{self.path.name}.0")

    def tail(self, n: int = 20) -> list:
        """Return the last ``n`` recorded thoughts (newest last)."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        return rows[-n:]


def _local_trace_dir() -> Path:
    """Data dir when running as a plugin (ctx.state.data_dir) or standalone."""
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home) / "plugin-data" / "deep-think"
    return Path.home() / ".hermes" / "plugin-data" / "deep-think"


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def deep_think_handler(args: dict[str, Any], **kwargs: Any) -> str:
    """Handle a ``deep_think`` call: validate, record, acknowledge.

    Returns a JSON string (all Hermes tool handlers must).
    """
    thought = args.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        return json.dumps({
            "status": "error",
            "error": "invalid_arguments",
            "message": "deep_think requires a non-empty 'thought' string.",
        })
    phase = args.get("phase")
    if not isinstance(phase, str) or phase not in {"", "plan", "verify", "reflect", "decide"}:
        phase = ""
    trace = ReasoningTrace(_local_trace_dir())
    record = trace.append(thought, phase=phase, session=str(kwargs.get("session_id", "") or ""))
    return json.dumps({
        "status": "ok",
        "message": (
            "Thought recorded. Continue with your next step; call deep_think "
            "again whenever you need to reason through something."
        ),
        "recorded_chars": record["chars"],
        "trace": str(trace.path),
    })


# ---------------------------------------------------------------------------
# System-prompt guidance (the load-bearing half — Anthropic: guidance in the
# system prompt beats guidance in the tool description)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SECTION = """\
## Plugin Context: deep_think

You have a `deep_think` tool — a private scratchpad for your working thoughts. \
It changes nothing and returns only an acknowledgement; its value is the space \
it gives you to reason before acting.

Use it before consequential or non-obvious steps, and after surprising results:

- **plan** — before a multi-step action, lay out the approach, alternatives \
rejected, and what would change your mind. One thought per plan, not per step.
- **verify** — after a tool result that surprises you, or before finishing: \
re-check assumptions, requirements coverage, and edge cases.
- **reflect** — when something failed or behaved unexpectedly: root cause \
first, then the fix.
- **decide** — when several viable options exist: name them, weigh them, pick \
one, and say why.

Keep each thought tight (a few sentences to ~15 lines) and written for \
yourself, not an audience. Do NOT use it for trivial tasks, single tool calls, \
or questions you can already answer — that wastes tokens. Every thought is \
appended to an inspectable reasoning trace (reasoning-trace.jsonl under the \
plugin data dir) and billed as ordinary output tokens.
"""


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:  # pragma: no cover - exercised via stub-ctx tests
    """Register the deep_think tool + system-prompt section (plugin loader hook)."""
    ctx.register_tool(
        name="deep_think",
        toolset=_TOOLSET,
        schema=DEEP_THINK_SCHEMA,
        handler=deep_think_handler,
        check_fn=lambda: True,
        emoji="🧠",
    )
    ctx.register_system_prompt_section(
        "deep_think",
        SYSTEM_PROMPT_SECTION,
        position="after_memory",
    )
