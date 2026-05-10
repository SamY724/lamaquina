# lamaquina

Local code-execution sandboxes for LLMs.

`lamaquina` exposes two execution tiers behind a single `Sandbox.execute(code)`
interface. Both return an `ExecutionResult` that records every attempted tool
call — successful, denied, or errored — so caller-side auditing and tracing
have a stable contract to read.

> Status: phase 1. Sandboxes + function registry are implemented. The
> `brokers/`, `wrappers/`, and `auditors/` packages are placeholders for
> later phases.

## Tiers at a glance

| | Tier 1 — `RestrictedSandbox` | Tier 2 — `DockerSandbox` |
|---|---|---|
| Mechanism | RestrictedPython AST transform in a `multiprocessing` spawn child | Docker container, optional gVisor (`runsc`) runtime |
| Tool surface | Allowlist via `FunctionRegistry`, proxied over `Pipe` | None built-in — caller wires their own |
| CPU/memory | `RLIMIT_CPU` + `RLIMIT_AS` in the child | `mem_limit`, `pids_limit`, container wall timeout |
| Isolation | Process boundary + AST restrictions | Container + `cap_drop=ALL` + `no-new-privileges` + read-only root + `network=none` (default) |
| Threat model | Prevent **accidents**. Not robust against an adversarial author. | Prevent **untrusted code** from touching the host. |
| Startup cost | ~50–150 ms per call (process spawn) | Container launch cost |

Both tiers share `LimitConfig` (CPU seconds, memory bytes, wall seconds) and
return `ExecutionResult { success, value, stdout, stderr, error, tool_calls }`.

## Install

```bash
uv sync
```

This installs the package (and its deps) into `.venv/`. Python 3.12+ required.

## Quickstart — Tier 1

```python
from lamaquina.sandboxes import LimitConfig, RestrictedSandbox
from lamaquina.tools.register import FunctionRegistry

registry = FunctionRegistry(state={"notes": {}})

@registry.register
def note_set(state, key, value):
    state["notes"][key] = value

@registry.register
def note_get(state, key):
    return state["notes"].get(key)

sandbox = RestrictedSandbox(
    registry=registry,
    limits=LimitConfig(cpu_seconds=2, memory_bytes=128 * 1024 * 1024),
)

out = sandbox.execute("""
note_set("greeting", "hello")
print(note_get("greeting"))
result = note_get("greeting")
""")

print(out.success)           # True
print(out.value)             # 'hello'
print(out.stdout)            # 'hello\n'
for call in out.tool_calls:
    print(call.status, call.name, call.args)
print(registry.state)        # {'notes': {'greeting': 'hello'}}
```

A runnable end-to-end demo (happy path + hostile probes + timeout) lives at
[`examples/restricted_sandbox_demo.py`](examples/restricted_sandbox_demo.py):

```bash
uv run python examples/restricted_sandbox_demo.py
```

## `FunctionRegistry`

A `FunctionRegistry` is a class-based store of callables that share a mutable
`state: dict`. Each registered function takes `state` as its first argument.
On every call the registry deep-copies state before and after, recording the
delta on `registry.history` as a `CallRecord`. Multiple independent registries
can coexist in one process — there is no module-level state.

Both sync (`registry.call(...)`) and async (`await registry.acall(...)`)
dispatch are supported. Use `acall` when you need to support either kind
transparently.

## Tier 2 — Docker

`DockerSandbox` runs `python -I -c <code>` inside a fresh container with
dropped capabilities, no-new-privileges, a read-only root filesystem, no
network by default, `pids_limit`, and an optional custom seccomp profile. For
extra hardening pass `runtime="runsc"` to use gVisor (must be installed on
the host).

This tier is intended for **untrusted code**. It does not bridge to a host
`FunctionRegistry`; if the LLM needs to call out, the caller supplies the
mechanism (a mounted helper script, an authenticated HTTP endpoint, etc.).

```python
from lamaquina.sandboxes import DockerSandbox, DockerSandboxConfig, LimitConfig

sandbox = DockerSandbox(
    config=DockerSandboxConfig(image="python:3.12-slim", network_mode="none"),
    limits=LimitConfig(cpu_seconds=10, memory_bytes=512 * 1024 * 1024),
)
out = sandbox.execute("print(2 + 2)")
print(out.stdout)  # '4\n'
```

## Threat model

* **Tier 1** assumes the *author* of sandboxed code is non-adversarial (e.g.
  a model that may make mistakes but is not actively trying to escape).
  RestrictedPython has documented escapes against a determined attacker —
  it strips dangerous builtins and rewrites attribute access, but it is not
  a security boundary on its own. The process boundary + `setrlimit` cap
  the blast radius of accidents (runaway loops, large allocations, simple
  attempts to `open(...)` or `import os`).
* **Tier 2** is the boundary you want for adversarial code. Defence in depth:
  container caps + read-only fs + seccomp + no network + (optionally) gVisor.

## Running the tests

```bash
uv run pytest
```

The unit suite covers `FunctionRegistry` semantics and the `RestrictedSandbox`
contract end-to-end (tool dispatch, denied calls, hostile-code rejection,
wall-clock timeout). The Docker tier is not exercised in CI; run it manually
with a Docker daemon available.

## Layout

```
src/lamaquina/
  sandboxes/      # base.Sandbox, RestrictedSandbox (Tier 1), DockerSandbox (Tier 2)
  tools/          # FunctionRegistry + CallRecord
  brokers/        # (placeholder — phase 2)
  wrappers/       # (placeholder — phase 2)
  auditors/       # (placeholder — phase 2)
examples/         # runnable demos
tests/unit/       # fast unit tests
tests/integration/  # (reserved for cross-tier / docker integration)
```
