"""Shared interface for sandboxes.

Both tiers — `RestrictedSandbox` (in-process AST-restricted) and `DockerSandbox`
(container with seccomp/caps) — implement `Sandbox.execute(code)` and return an
`ExecutionResult`. The result records every attempted tool/registry call, not
only the successful ones, so probing behaviour is observable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


CallStatus = Literal["ok", "denied", "error"]


@dataclass
class CallAttempt:
    name: str
    args: tuple
    kwargs: dict
    status: CallStatus
    error: str | None = None


@dataclass
class ExecutionResult:
    success: bool
    value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    tool_calls: list[CallAttempt] = field(default_factory=list)


@dataclass
class LimitConfig:
    cpu_seconds: float = 5.0
    memory_bytes: int = 256 * 1024 * 1024
    wall_seconds: float = 10.0


class SandboxError(Exception):
    pass


class Sandbox(ABC):
    @abstractmethod
    def execute(self, code: str) -> ExecutionResult: ...
