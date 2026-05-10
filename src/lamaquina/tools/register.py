"""Function registry for user-provided tools with shared mutable state.

A `FunctionRegistry` holds a collection of callables that operate on a shared
state dict. Functions registered here receive `state` as their first argument
and may read or mutate it. Around every invocation the registry snapshots the
state, so consumers (brokers, auditors, sandboxes) can see exactly how each
call changed the environment.

The registry is class-based by design: multiple independent registries can
coexist in the same process, each with its own state and history. Decoration
is done through the instance (`@registry.register`), avoiding any reliance on
module-level state.

Both synchronous and asynchronous tool functions are supported. Use `call`
for sync-only invocation and `acall` to transparently handle either.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CallRecord:
    name: str
    args: tuple
    kwargs: dict
    result: Any
    state_before: dict
    state_after: dict
    error: BaseException | None = None


class FunctionRegistry:
    def __init__(self, state: dict | None = None):
        self._functions: dict[str, Callable] = {}
        self.state: dict = state if state is not None else {}
        self.history: list[CallRecord] = []

    def register(self, fn: Callable | None = None, *, name: str | None = None):
        def wrap(f: Callable):
            key = name or f.__name__
            if key in self._functions:
                raise ValueError(f"function {key!r} already registered")
            self._functions[key] = f
            return f

        return wrap(fn) if fn is not None else wrap

    def _begin(
        self, name: str, args: tuple, kwargs: dict
    ) -> tuple[Callable, CallRecord]:
        if name not in self._functions:
            raise KeyError(f"no function named {name!r}")
        before = copy.deepcopy(self.state)
        record = CallRecord(name, args, kwargs, None, before, before)
        self.history.append(record)
        return self._functions[name], record

    def call(self, name: str, *args, **kwargs) -> Any:
        fn, record = self._begin(name, args, kwargs)
        if inspect.iscoroutinefunction(fn):
            raise TypeError(f"{name!r} is async; use acall instead")
        try:
            record.result = fn(self.state, *args, **kwargs)
            return record.result
        except BaseException as e:
            record.error = e
            raise
        finally:
            record.state_after = copy.deepcopy(self.state)

    async def acall(self, name: str, *args, **kwargs) -> Any:
        fn, record = self._begin(name, args, kwargs)
        try:
            result = fn(self.state, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            record.result = result
            return result
        except BaseException as e:
            record.error = e
            raise
        finally:
            record.state_after = copy.deepcopy(self.state)

    def names(self) -> list[str]:
        return list(self._functions)

    def __contains__(self, name: str) -> bool:
        return name in self._functions
