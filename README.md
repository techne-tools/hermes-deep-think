# hermes-deep-think 🧠

> Give your agent a place to think out loud.

A tiny [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that exposes a
**`deep_think` tool** — a no-op scratchpad the model calls with its working thoughts
instead of (or alongside) a vendor "thinking mode". Every thought is recorded to an
**inspectable, size-capped reasoning trace** on disk.

Same reasoning capability, no dedicated infrastructure.

> **This is the techne-tools fork.** It keeps the upstream mechanism (τ-Bench think-tool
> pattern, trace recording, rotation) and swaps the system-prompt guidance for a
> **creative mode**: the four phases are re-framed as *sketch / audition / reflect /
> commit* — divergence before convergence, failure as data, commitment with named risk.
> The tool schema is unchanged, so the trace format and rotation behaviour are identical
> to upstream. See [Creative mode](#creative-mode) below.

## Why

In August 2026, [@_can1357](https://x.com/_can1357/status/2087228354399265125) popularized
a neat observation:

> "guys you do know you can just disable thinking, and instead give it a `deep_think`
> tool, and it will call it with internal CoT reasoning format right?"

The interesting part isn't the hack — it's that the mechanism was already **official,
published, and benchmarked**: Anthropic's [*think tool*](https://www.anthropic.com/engineering/claude-think-tool)
(Mar 2025), derived from Sierra Research's [τ-Bench](https://arxiv.org/abs/2406.12045).
On τ-Bench's airline domain, a no-op think tool + system-prompt guidance lifted Claude
3.7 Sonnet from 0.332 → 0.584 pass¹ (+54% relative). The insight: **reasoning doesn't
need a vendor switch — the model will happily externalize its monologue through any tool
shaped like one.**

| | Vendor thinking mode | `deep_think` tool |
|---|---|---|
| Requires model/provider support | ✅ | ❌ — works on any tool-calling model |
| User-inspectable | varies (often encrypted) | ✅ `reasoning-trace.jsonl` |
| Survives thinking disabled / non-reasoning models | ❌ | ✅ |
| Billed as reasoning tokens | sometimes | no — ordinary output tokens* |
| Guidable via system prompt | limited | ✅ |

\* *Note: the thoughts are billed as ordinary output tokens, which is usually not cheaper
than reasoning tokens — this is about capability and transparency, not cost.*

## What it does

1. **Registers a `deep_think` tool** — τ-Bench schema (single required `thought` string,
   optional `phase`: `plan` / `verify` / `reflect` / `decide`).
2. **Registers a system-prompt section** teaching the model *when* and *how* to use it —
   Anthropic's benchmarks show guidance in the system prompt is the load-bearing half.
3. **Records every thought** to `~/.hermes/plugin-data/deep-think/reasoning-trace.jsonl`
   (append-only JSONL, 1 MiB × 4 generations rotation cap).

The handler itself is a pure no-op: it validates, records, and acknowledges. It never
fetches, mutates, or decides anything.

## Creative mode

This fork replaces the upstream system-prompt guidance with a creative-mode framing.
The tool, schema, trace format, and rotation are unchanged — only the guidance the
model receives about *when and how* to think is different.

| Phase | Upstream (conservative) | Creative mode |
|---|---|---|
| `plan` | approach, alternatives rejected | **sketch** — territory, not route; keep several directions alive |
| `verify` | re-check assumptions, edge cases | **audition** — try the work on; notice what you didn't expect |
| `reflect` | root cause, then fix | **keep** — what worked / what didn't / what would you do differently |
| `decide` | weigh options, pick safest | **commit** — pick the most alive option, name the risk |

The system-prompt section also ships two worked examples (a sound-design sketch and a
draft-revision reflection) demonstrating the *shape* of a good thought, mirroring the
role worked examples played in Anthropic's benchmarked "optimized prompt".

**Trade-offs, stated plainly:** this guidance will not improve τ-bench-style
policy-compliance scores — it is not designed to. It optimises for keeping the work
open, treating the trace as a rehearsal-room wall rather than a plan document. It is
a complement to, not a replacement for, the upstream conservative guidance: use the
upstream for reliability-critical work, this fork for exploratory and creative work.

## Install

**Requirements:** Hermes Agent ≥ 0.20 with the plugin system (`hermes plugins --help`
works), Python ≥ 3.11.

```bash
git clone https://github.com/macayaven/hermes-deep-think.git
cd hermes-deep-think

# 1. Copy (or symlink) the plugin into your Hermes home
mkdir -p ~/.hermes/plugins
cp -r plugin ~/.hermes/plugins/deep-think

# 2. Enable it (plugins are opt-in by design)
hermes plugins enable deep-think

# 3. Verify
hermes plugins list | grep deep-think
```

New sessions pick it up automatically (toolsets are frozen per-session to preserve
prompt caching — start a fresh session or run `/reset`).

### Uninstall

```bash
hermes plugins disable deep-think
rm -rf ~/.hermes/plugins/deep-think
```

## Usage

Nothing to operate — the model calls it on its own when the guidance tells it to. Two
ways to watch it work:

```bash
# Tail the trace live
tail -f ~/.hermes/plugin-data/deep-think/reasoning-trace.jsonl | jq .

# Ask it something worth thinking about
hermes chat -q "Design a rate limiter for a public API. Compare 3 algorithms before choosing."
```

Sample trace entry:

```json
{"ts": "2026-08-16T21:40:12Z", "phase": "plan", "session": null, "chars": 512,
 "thought": "Options: fixed window, sliding window log, token bucket..."}
```

## How it works

```
┌─────────┐  tool_call(thought)  ┌──────────────┐  append JSONL  ┌──────────────────────────┐
│  model  │ ───────────────────▶ │ deep_think   │ ─────────────▶ │ reasoning-trace.jsonl(.N)│
└─────────┘                      │  (no-op)     │                └──────────────────────────┘
      ▲        ack(status)       └──────────────┘
      └────────────────────────────────┘
```

- **Tool schema** mirrors the τ-Bench `think` tool, with a `phase` tag added for
  trace analysis.
- **System-prompt section** (`after_memory` position, ≤ 4k chars) encodes
  when-to-use / when-not-to-use guidance distilled from Anthropic's findings.
- **Trace rotation**: the live file rotates at 1 MiB, keeping 3 generations
  (4 MiB ceiling total).

## Honest limitations

- **It is not "hidden" reasoning.** There is no proof the text a model writes here is its
  *internal* chain-of-thought rather than a lookalike generated for the occasion —
  the same is true of vendor thinking traces; both are unfalsifiable from outside.
- **No cost dodge.** Thoughts are billed as output tokens. If your provider's reasoning
  tokens are priced differently, do the math for your workload.
- **Provider detection.** Some providers (e.g. Anthropic's reasoning-extraction
  classifier) actively detect patterns that look like reasoning extraction. This plugin
  is a first-party quality tool for *your own* agent sessions — not an extraction
  harness against third-party APIs, and using it against ToS is on you.
- **Guidance-dependent.** Without the system-prompt section, models call it far less
  (that's why the plugin registers both).
- **Windows note:** trace paths use `Path.home()` semantics; fine on Windows, but the
  docs assume `~/.hermes`.

## Development

```bash
# Tests (stdlib + pytest, no network, no Hermes runtime needed)
python -m pytest tests/ -q

# Lint (optional)
uvx ruff check plugin tests
```

The plugin module has **zero dependencies** beyond the standard library — it must load
in every Hermes environment without installs.

## Credits

- **Can Bölük ([@_can1357](https://x.com/_can1357))** — the community popularization
  (Aug 2026) and the name.
- **Anthropic** — the *think tool* engineering post and τ-Bench evaluation that this
  mechanism is directly derived from.
- **Sierra Research** — τ-Bench (arXiv:2406.12045).

## License

MIT — see [LICENSE](LICENSE).
