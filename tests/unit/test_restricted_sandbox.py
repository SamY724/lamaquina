"""Unit tests for the Tier 1 `RestrictedSandbox`.

These exercise the contract from `sandboxes/__init__.py`:

* Safe code runs and returns a value via `result = ...`.
* `print(...)` is captured into `ExecutionResult.stdout`.
* Registered tools are callable from sandboxed code, and every attempt — ok,
  denied, or errored — is observable on `ExecutionResult.tool_calls`.
* Hostile code (filesystem, imports, dunder escape, eval) is rejected by
  RestrictedPython, not by us, but we surface the failure cleanly.
* Wall-clock timeout terminates the child.

Each test runs the sandbox in its own spawned subprocess; tests are I/O bound
on process startup (~50–150ms) so the suite stays small but representative.
"""

from __future__ import annotations

import warnings

import pytest

from lamaquina.sandboxes import LimitConfig, RestrictedSandbox
from lamaquina.tools.register import FunctionRegistry


# RestrictedPython emits a benign SyntaxWarning when sandboxed code calls
# print() but never reads `printed`. Silence it for test noise only.
pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")


@pytest.fixture
def fast_limits() -> LimitConfig:
    return LimitConfig(cpu_seconds=2, memory_bytes=128 * 1024 * 1024, wall_seconds=5)


def _registry_with_notes() -> FunctionRegistry:
    registry = FunctionRegistry(state={"notes": {}})

    @registry.register
    def note_set(state: dict, key: str, value: str) -> None:
        state["notes"][key] = value

    @registry.register
    def note_get(state: dict, key: str) -> str | None:
        return state["notes"].get(key)

    @registry.register
    def boom(state: dict) -> None:
        raise RuntimeError("planned failure")

    return registry


def test_returns_result_value(fast_limits):
    sandbox = RestrictedSandbox(limits=fast_limits)
    out = sandbox.execute("result = 1 + 2 + 3")
    assert out.success is True
    assert out.value == 6
    assert out.error is None


def test_captures_stdout(fast_limits):
    sandbox = RestrictedSandbox(limits=fast_limits)
    out = sandbox.execute("print('hello')\nprint('world')\nresult = None")
    assert out.success is True
    assert "hello" in out.stdout
    assert "world" in out.stdout


def test_tool_calls_dispatch_and_record(fast_limits):
    registry = _registry_with_notes()
    sandbox = RestrictedSandbox(registry=registry, limits=fast_limits)

    code = """
note_set("a", "1")
note_set("b", "2")
result = note_get("a")
""".strip()

    out = sandbox.execute(code)
    assert out.success is True
    assert out.value == "1"

    statuses = [(c.name, c.status) for c in out.tool_calls]
    assert statuses == [("note_set", "ok"), ("note_set", "ok"), ("note_get", "ok")]
    assert registry.state["notes"] == {"a": "1", "b": "2"}
    # The host registry recorded its own per-call history independently.
    assert [r.name for r in registry.history] == ["note_set", "note_set", "note_get"]


def test_tool_call_error_recorded_and_propagated(fast_limits):
    registry = _registry_with_notes()
    sandbox = RestrictedSandbox(registry=registry, limits=fast_limits)

    # Sandboxed code calls a registered tool that raises. The proxy re-raises
    # in the child, which propagates as an unhandled exception unless caught.
    out = sandbox.execute("boom()")
    assert out.success is False
    assert out.error is not None
    assert "RuntimeError" in out.error
    assert len(out.tool_calls) == 1
    attempt = out.tool_calls[0]
    assert attempt.name == "boom"
    assert attempt.status == "error"
    assert attempt.error is not None


def test_unregistered_tool_name_is_namerror(fast_limits):
    """A tool that was never registered isn't injected into sandbox globals,
    so the LLM's code raises NameError before any IPC happens. The point of
    this test is to nail down the *contract*: tool_calls only records calls
    that reached the parent."""
    registry = _registry_with_notes()
    sandbox = RestrictedSandbox(registry=registry, limits=fast_limits)
    out = sandbox.execute("note_delete('a')")
    assert out.success is False
    assert "NameError" in (out.error or "")
    assert out.tool_calls == []


@pytest.mark.parametrize(
    "label,code,expected_substring",
    [
        ("open", "result = open('/etc/passwd').read()", "open"),
        ("import_os", "import os\nresult = os.listdir('/')", "__import__"),
        ("dunder_attr", "result = ().__class__", "__class__"),
        ("eval", "result = eval('1+1')", "Eval"),
        ("exec", "exec('x=1')", "Exec"),
    ],
)
def test_hostile_code_is_blocked(fast_limits, label, code, expected_substring):
    sandbox = RestrictedSandbox(limits=fast_limits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sandbox.execute(code)
    assert out.success is False
    assert out.error is not None
    assert expected_substring in out.error, (
        f"{label}: error {out.error!r} did not mention {expected_substring!r}"
    )


def test_compile_error_returned_cleanly(fast_limits):
    sandbox = RestrictedSandbox(limits=fast_limits)
    out = sandbox.execute("def def def")
    assert out.success is False
    assert out.error is not None
    assert "compile error" in out.error or "SyntaxError" in out.error


def test_wall_timeout_terminates_child():
    # Wall < CPU so the parent's poll trips first and we get the deterministic
    # "timeout after Xs" message rather than a SIGXCPU-killed child.
    limits = LimitConfig(cpu_seconds=5, wall_seconds=0.5, memory_bytes=64 * 1024 * 1024)
    sandbox = RestrictedSandbox(limits=limits)
    out = sandbox.execute("while True:\n    pass")
    assert out.success is False
    assert out.error is not None
    assert "timeout" in out.error


def test_no_registry_means_no_tools(fast_limits):
    sandbox = RestrictedSandbox(limits=fast_limits)  # registry=None
    out = sandbox.execute("note_set('a', 1)")
    assert out.success is False
    assert "NameError" in (out.error or "")
