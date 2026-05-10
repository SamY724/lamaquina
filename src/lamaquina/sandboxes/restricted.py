"""Tier 1 — RestrictedPython sandbox.

The LLM-supplied code is AST-transformed by RestrictedPython before execution,
which strips dangerous builtins (`open`, `__import__`, `exec`, `eval`, attribute
escapes, etc.) and rewrites attribute/iteration access through guard hooks. The
only callable surface exposed beyond safe math/string ops is the host's
`FunctionRegistry` — so an allowlist, not a blocklist.

Execution happens in a `multiprocessing` child started with the *spawn* method,
so the parent's interpreter state is not shared. The child applies
`RLIMIT_AS` (memory) and `RLIMIT_CPU` (CPU seconds) before `exec`. Tool calls
are proxied back to the parent over a `Pipe`; the parent looks up the registry,
runs the call, and ships the return value back. Every attempted call (allowed,
denied, errored) is recorded on the `ExecutionResult`.

Treat this tier as *prevent accidents*, not *prevent attacks*: RestrictedPython
has documented escapes against an adversarial author. For untrusted code use
`DockerSandbox`.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Any

from RestrictedPython import (
    compile_restricted,
    limited_builtins,
    safe_builtins,
    utility_builtins,
)
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

from ..tools.register import FunctionRegistry
from .base import CallAttempt, ExecutionResult, LimitConfig, Sandbox

logger = logging.getLogger("lamaquina.sandbox.restricted")

_SAFE_BUILTINS: dict[str, Any] = {
    **safe_builtins,
    **utility_builtins,
    **limited_builtins,
}


def _child_target(
    conn: Connection,
    code: str,
    tool_names: list[str],
    cpu_seconds: float,
    memory_bytes: int,
) -> None:
    # Imported inside the child so spawn-mode workers don't pay the cost on
    # import of this module from the parent process.
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu_hard = max(int(cpu_seconds) + 1, 1)
        resource.setrlimit(resource.RLIMIT_CPU, (max(int(cpu_seconds), 1), cpu_hard))
    except (ValueError, OSError) as e:
        conn.send(
            {
                "type": "done",
                "success": False,
                "error": f"setrlimit failed: {e}",
                "stdout": "",
            }
        )
        conn.close()
        return

    def _make_proxy(tool_name: str):
        def proxy(*args: Any, **kwargs: Any) -> Any:
            conn.send(
                {"type": "call", "name": tool_name, "args": args, "kwargs": kwargs}
            )
            reply = conn.recv()
            if reply["type"] == "result":
                return reply["value"]
            raise RuntimeError(reply.get("error", "tool call failed"))

        proxy.__name__ = tool_name
        return proxy

    g: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "_getiter_": default_guarded_getiter,
        "_getitem_": default_guarded_getitem,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getattr_": safer_getattr,
        "_print_": PrintCollector,
    }
    for name in tool_names:
        g[name] = _make_proxy(name)
    locals_: dict[str, Any] = {}

    try:
        byte_code = compile_restricted(code, "<sandbox>", "exec")
    except SyntaxError as e:
        conn.send(
            {
                "type": "done",
                "success": False,
                "error": f"compile error: {e}",
                "stdout": "",
            }
        )
        conn.close()
        return

    printed = ""
    try:
        exec(byte_code, g, locals_)
        if "_print" in locals_:
            try:
                printed = locals_["_print"]()
            except Exception:
                printed = ""
        conn.send(
            {
                "type": "done",
                "success": True,
                "value": locals_.get("result"),
                "stdout": printed,
            }
        )
    except BaseException as e:
        if "_print" in locals_:
            try:
                printed = locals_["_print"]()
            except Exception:
                pass
        conn.send(
            {
                "type": "done",
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "stdout": printed,
            }
        )
    finally:
        conn.close()


class RestrictedSandbox(Sandbox):
    def __init__(
        self,
        registry: FunctionRegistry | None = None,
        limits: LimitConfig | None = None,
    ):
        self.registry = registry
        self.limits = limits or LimitConfig()

    def execute(self, code: str) -> ExecutionResult:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        tool_names = self.registry.names() if self.registry is not None else []
        proc = ctx.Process(
            target=_child_target,
            args=(
                child_conn,
                code,
                tool_names,
                self.limits.cpu_seconds,
                self.limits.memory_bytes,
            ),
        )
        proc.start()
        child_conn.close()

        attempts: list[CallAttempt] = []
        result: ExecutionResult | None = None
        wall = self.limits.wall_seconds

        try:
            while True:
                if not parent_conn.poll(wall):
                    proc.terminate()
                    proc.join(1.0)
                    if proc.is_alive():
                        proc.kill()
                    return ExecutionResult(
                        success=False,
                        error=f"timeout after {wall:.1f}s",
                        tool_calls=attempts,
                    )
                try:
                    msg = parent_conn.recv()
                except EOFError:
                    break

                kind = msg.get("type")
                if kind == "call":
                    name, args, kwargs = msg["name"], msg["args"], msg["kwargs"]
                    if self.registry is None or name not in self.registry:
                        attempts.append(
                            CallAttempt(name, args, kwargs, "denied", "unknown tool")
                        )
                        logger.warning("denied tool call: %s", name)
                        parent_conn.send(
                            {"type": "error", "error": f"unknown tool {name!r}"}
                        )
                        continue
                    try:
                        value = self.registry.call(name, *args, **kwargs)
                        attempts.append(CallAttempt(name, args, kwargs, "ok"))
                        logger.info("tool call: %s", name)
                        parent_conn.send({"type": "result", "value": value})
                    except BaseException as e:
                        attempts.append(
                            CallAttempt(name, args, kwargs, "error", repr(e))
                        )
                        logger.warning("tool call %s raised %r", name, e)
                        parent_conn.send({"type": "error", "error": repr(e)})
                elif kind == "done":
                    result = ExecutionResult(
                        success=bool(msg.get("success")),
                        value=msg.get("value"),
                        stdout=msg.get("stdout", ""),
                        error=msg.get("error"),
                        tool_calls=attempts,
                    )
                    break
                else:
                    logger.error("unexpected child message: %r", msg)
                    break
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join()

        if result is None:
            return ExecutionResult(
                success=False,
                error="child terminated without result",
                tool_calls=attempts,
            )
        # SIGKILL by the kernel (e.g. RLIMIT_AS hit) shows as -9.
        if proc.exitcode is not None and proc.exitcode < 0 and result.success:
            result.success = False
            result.error = (
                f"child killed by signal {-proc.exitcode} (likely OOM/CPU limit)"
            )
        return result
