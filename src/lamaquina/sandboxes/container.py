"""Tier 2 — Docker (+ optional gVisor) sandbox.

Runs the LLM-supplied code inside a fresh container with:

  * memory cap and CPU-time bound (`mem_limit`, container wall timeout)
  * `cap_drop=["ALL"]` and `no-new-privileges` — kills `setuid` escalations
  * `pids_limit` — bounds fork bombs
  * `read_only` root filesystem; opt-in bind mounts must be declared
  * `network_mode="none"` by default
  * a custom seccomp profile (path → JSON contents) when supplied; otherwise
    Docker's built-in default seccomp profile (already an allowlist) is used
  * optional `runtime="runsc"` (gVisor) for an extra kernel intercept layer

The container is `--rm`'d after `wait()`. A short grace beyond `cpu_seconds` is
allowed before the container is killed and reported as a timeout.

Note: this tier does not bridge to the host `FunctionRegistry`. Inside the
container the LLM has Python's standard library; tool integration is the
caller's job (e.g. mount a script, expose an HTTP endpoint with a token).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import Mount

from .base import ExecutionResult, LimitConfig, Sandbox, SandboxError

logger = logging.getLogger("lamaquina.sandbox.docker")


@dataclass
class DockerSandboxConfig:
    image: str = "python:3.12-slim"
    network_mode: str = "none"
    runtime: str | None = None
    mounts: list[dict] = field(default_factory=list)
    read_only_root: bool = True
    pids_limit: int = 64
    seccomp_profile_path: str | None = None
    no_new_privileges: bool = True
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    env: dict[str, str] = field(default_factory=dict)
    user: str | None = "65534:65534"


class DockerSandbox(Sandbox):
    def __init__(
        self,
        config: DockerSandboxConfig | None = None,
        limits: LimitConfig | None = None,
    ):
        self.config = config or DockerSandboxConfig()
        self.limits = limits or LimitConfig()
        try:
            self._client = docker.from_env()
        except DockerException as e:
            raise SandboxError(f"could not connect to docker daemon: {e}") from e

    def execute(self, code: str) -> ExecutionResult:
        cfg = self.config

        security_opt: list[str] = []
        if cfg.seccomp_profile_path:
            try:
                profile = Path(cfg.seccomp_profile_path).read_text()
            except OSError as e:
                raise SandboxError(
                    f"could not read seccomp profile {cfg.seccomp_profile_path}: {e}"
                ) from e
            security_opt.append(f"seccomp={profile}")
        if cfg.no_new_privileges:
            security_opt.append("no-new-privileges:true")

        mounts = [
            Mount(
                target=m["target"],
                source=m["source"],
                type="bind",
                read_only=bool(m.get("read_only", True)),
            )
            for m in cfg.mounts
        ]

        run_kwargs: dict[str, Any] = dict(
            image=cfg.image,
            command=["python", "-I", "-c", code],
            mem_limit=self.limits.memory_bytes,
            memswap_limit=self.limits.memory_bytes,
            network_mode=cfg.network_mode,
            read_only=cfg.read_only_root,
            pids_limit=cfg.pids_limit,
            security_opt=security_opt or None,
            cap_drop=cfg.cap_drop,
            mounts=mounts or None,
            environment=cfg.env or None,
            user=cfg.user,
            detach=True,
            stdout=True,
            stderr=True,
        )
        if cfg.runtime:
            run_kwargs["runtime"] = cfg.runtime

        logger.info(
            "starting container image=%s network=%s runtime=%s mem=%d cpu=%.1fs",
            cfg.image,
            cfg.network_mode,
            cfg.runtime or "default",
            self.limits.memory_bytes,
            self.limits.cpu_seconds,
        )

        try:
            container = self._client.containers.run(**run_kwargs)
        except ImageNotFound:
            return ExecutionResult(success=False, error=f"image not found: {cfg.image}")
        except APIError as e:
            return ExecutionResult(success=False, error=f"docker api error: {e}")

        try:
            try:
                wait_result = container.wait(timeout=self.limits.cpu_seconds + 2.0)
            except Exception:
                logger.warning("container exceeded wall budget; killing")
                try:
                    container.kill()
                except Exception:
                    pass
                stdout = container.logs(stdout=True, stderr=False).decode(
                    "utf-8", errors="replace"
                )
                stderr = container.logs(stdout=False, stderr=True).decode(
                    "utf-8", errors="replace"
                )
                return ExecutionResult(
                    success=False,
                    error=f"timeout after {self.limits.cpu_seconds + 2.0:.1f}s",
                    stdout=stdout,
                    stderr=stderr,
                )

            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )
            exit_code = wait_result.get("StatusCode", -1)
            if exit_code == 0:
                return ExecutionResult(success=True, stdout=stdout, stderr=stderr)
            err = f"container exited with code {exit_code}"
            if exit_code == 137:
                err = "container killed (likely OOM or external SIGKILL)"
            return ExecutionResult(
                success=False, stdout=stdout, stderr=stderr, error=err
            )
        finally:
            try:
                container.remove(force=True)
            except Exception as e:
                logger.warning("failed to remove container: %s", e)
