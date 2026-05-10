"""End-to-end demo of the Tier 1 (RestrictedPython) sandbox.

Run with:

    uv run python examples/restricted_sandbox_demo.py

It does three things:

1. Registers a tiny note-taking tool surface on a `FunctionRegistry`.
2. Executes LLM-style code that uses those tools, and prints the
   `ExecutionResult` (return value, stdout, every tool-call attempt).
3. Executes deliberately hostile code (filesystem, import, attribute escape)
   and prints what RestrictedPython rejected.

The point is to show the *observability* contract: the sandbox surfaces every
attempted call — successful, denied, or errored — alongside the registry's
own per-call state snapshots.
"""

from __future__ import annotations

from lamaquina.sandboxes import LimitConfig, RestrictedSandbox
from lamaquina.tools.register import FunctionRegistry


def build_registry() -> FunctionRegistry:
    registry = FunctionRegistry(state={"notes": {}})

    @registry.register
    def note_set(state: dict, key: str, value: str) -> None:
        state["notes"][key] = value

    @registry.register
    def note_get(state: dict, key: str) -> str | None:
        return state["notes"].get(key)

    @registry.register
    def note_keys(state: dict) -> list[str]:
        return sorted(state["notes"])

    return registry


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_result(result) -> None:
    print(f"success: {result.success}")
    if result.error:
        print(f"error:   {result.error}")
    print(f"value:   {result.value!r}")
    if result.stdout:
        print("stdout:")
        for line in result.stdout.splitlines():
            print(f"  | {line}")
    if result.tool_calls:
        print("tool_calls:")
        for call in result.tool_calls:
            tag = f"[{call.status}]"
            err = f"  ({call.error})" if call.error else ""
            print(f"  {tag:<9} {call.name}{call.args}{err}")


def demo_happy_path() -> None:
    section("1. Happy path — sandboxed code calls registered tools")

    registry = build_registry()
    sandbox = RestrictedSandbox(
        registry=registry,
        limits=LimitConfig(cpu_seconds=2, memory_bytes=128 * 1024 * 1024),
    )

    code = """
note_set("greeting", "hello")
note_set("subject", "world")
print(note_get("greeting") + ", " + note_get("subject"))
result = note_keys()
""".strip()

    show_result(sandbox.execute(code))

    print()
    print(f"registry final state: {registry.state}")
    print(f"registry history len: {len(registry.history)}")


def demo_denied_unknown_tool() -> None:
    section("2. Sandbox calls an unregistered tool name")

    registry = build_registry()
    sandbox = RestrictedSandbox(registry=registry)

    # `note_delete` was never registered. The proxy doesn't even exist in the
    # sandbox globals, so the code raises NameError. The attempt is *not*
    # recorded, because the call never reached the parent. To see a 'denied'
    # attempt, the parent would have had to receive a call message — see
    # demo_probe() below for the equivalent on hostile builtins.
    code = "result = note_delete('foo')"
    show_result(sandbox.execute(code))


def demo_probe() -> None:
    section("3. Hostile probes — RestrictedPython blocks them at compile time")

    registry = build_registry()
    sandbox = RestrictedSandbox(registry=registry)

    probes = [
        ("open() filesystem read", "open('/etc/passwd').read()"),
        ("import os escape", "import os\nresult = os.listdir('/')"),
        ("dunder attribute escape", "result = ().__class__.__bases__[0]"),
        ("eval", "result = eval('1+1')"),
    ]
    for label, code in probes:
        print(f"\n--- probe: {label} ---")
        show_result(sandbox.execute(code))


def demo_timeout() -> None:
    section("4. CPU/wall timeout terminates the child")

    # wall < cpu so the parent's poll trips first and we get a clean timeout
    # message rather than a SIGXCPU-killed child.
    sandbox = RestrictedSandbox(
        limits=LimitConfig(
            cpu_seconds=5, wall_seconds=0.5, memory_bytes=64 * 1024 * 1024
        ),
    )
    code = "while True:\n    pass"
    show_result(sandbox.execute(code))


if __name__ == "__main__":
    demo_happy_path()
    demo_denied_unknown_tool()
    demo_probe()
    demo_timeout()
