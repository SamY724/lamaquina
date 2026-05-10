"""Two-tier sandbox package.

* `RestrictedSandbox` — Tier 1, RestrictedPython AST transform in a spawned
  child process with `setrlimit` CPU/memory bounds. Allowlist of host tools via
  `FunctionRegistry`. Prevents accidents, not adversarial escape.
* `DockerSandbox` — Tier 2, container with dropped caps, configurable seccomp
  (Docker default if no profile path given), optional gVisor runtime, no
  network by default. For untrusted code or multi-step agentic work.

Both implement `Sandbox.execute(code) -> ExecutionResult`.
"""

from .base import (
    CallAttempt,
    ExecutionResult,
    LimitConfig,
    Sandbox,
    SandboxError,
)
from .container import DockerSandbox, DockerSandboxConfig
from .restricted import RestrictedSandbox

__all__ = [
    "CallAttempt",
    "DockerSandbox",
    "DockerSandboxConfig",
    "ExecutionResult",
    "LimitConfig",
    "RestrictedSandbox",
    "Sandbox",
    "SandboxError",
]
