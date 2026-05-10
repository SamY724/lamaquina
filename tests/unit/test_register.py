"""Unit tests for `FunctionRegistry`.

Covers registration, sync/async dispatch, error capture, and the per-call
state-snapshot history that downstream auditors rely on.
"""

from __future__ import annotations

import asyncio

import pytest

from lamaquina.tools.register import FunctionRegistry


def test_register_and_call_mutates_state():
    registry = FunctionRegistry(state={"counter": 0})

    @registry.register
    def bump(state: dict, by: int = 1) -> int:
        state["counter"] += by
        return state["counter"]

    assert registry.call("bump") == 1
    assert registry.call("bump", 4) == 5
    assert registry.state == {"counter": 5}
    assert registry.names() == ["bump"]
    assert "bump" in registry


def test_register_with_explicit_name():
    registry = FunctionRegistry()

    @registry.register(name="add")
    def _adder(state: dict, a: int, b: int) -> int:
        return a + b

    assert registry.call("add", 2, 3) == 5
    assert "add" in registry
    assert "_adder" not in registry


def test_duplicate_registration_rejected():
    registry = FunctionRegistry()

    @registry.register
    def f(state: dict) -> None:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register(name="f")
        def g(state: dict) -> None:
            pass


def test_unknown_call_raises_keyerror():
    registry = FunctionRegistry()
    with pytest.raises(KeyError):
        registry.call("nope")


def test_history_snapshots_isolate_before_and_after():
    """The before-snapshot must capture pre-call state even if the function
    later mutates it. This is the property that lets auditors diff state."""
    registry = FunctionRegistry(state={"items": []})

    @registry.register
    def push(state: dict, x: int) -> None:
        state["items"].append(x)

    registry.call("push", 1)
    registry.call("push", 2)

    assert len(registry.history) == 2
    rec0, rec1 = registry.history
    assert rec0.state_before == {"items": []}
    assert rec0.state_after == {"items": [1]}
    assert rec1.state_before == {"items": [1]}
    assert rec1.state_after == {"items": [1, 2]}
    # Snapshots are deep copies — mutating the live state must not retroactively
    # alter recorded history.
    registry.state["items"].clear()
    assert rec1.state_after == {"items": [1, 2]}


def test_history_records_errors():
    registry = FunctionRegistry()

    @registry.register
    def boom(state: dict) -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        registry.call("boom")
    assert len(registry.history) == 1
    rec = registry.history[0]
    assert isinstance(rec.error, RuntimeError)
    assert rec.result is None


def test_call_rejects_async_function():
    registry = FunctionRegistry()

    @registry.register
    async def slow(state: dict) -> int:
        return 42

    with pytest.raises(TypeError, match="async"):
        registry.call("slow")


def test_acall_handles_sync_and_async():
    registry = FunctionRegistry(state={"n": 0})

    @registry.register
    def sync_inc(state: dict) -> int:
        state["n"] += 1
        return state["n"]

    @registry.register
    async def async_inc(state: dict) -> int:
        await asyncio.sleep(0)
        state["n"] += 10
        return state["n"]

    async def driver():
        a = await registry.acall("sync_inc")
        b = await registry.acall("async_inc")
        return a, b

    a, b = asyncio.run(driver())
    assert a == 1
    assert b == 11
    assert registry.state == {"n": 11}


def test_independent_registries_do_not_share_state():
    r1 = FunctionRegistry(state={"x": 1})
    r2 = FunctionRegistry(state={"x": 99})

    @r1.register
    def get1(state: dict) -> int:
        return state["x"]

    @r2.register
    def get2(state: dict) -> int:
        return state["x"]

    assert r1.call("get1") == 1
    assert r2.call("get2") == 99
    assert "get1" not in r2
    assert "get2" not in r1
