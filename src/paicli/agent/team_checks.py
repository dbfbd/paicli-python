"""Execute Team's behavioral checks against an explicit integration workspace."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable
from itertools import groupby
from pathlib import Path

from paicli.agent.team_board import TeamCheck
from paicli.agent.team_workspace import TeamWorkspace
from paicli.config import PaiCliConfig
from paicli.harness.evidence import current_run
from paicli.harness.project_env import augment_env_for_venv, resolve_project_executable
from paicli.policy import CommandGuard

_CAPTURE_LIMIT = 64 * 1024
_PREVIEW_LIMIT = 1000
_ENV_KEYS = {
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
}
_SHELLS = {"cmd", "powershell", "pwsh", "bash", "sh", "zsh", "dash", "fish", "ksh"}


@dataclass(frozen=True, slots=True)
class TeamCheckResult:
    passed: bool
    detail: str
    evidence: dict | None = None
    status: str = "passed"


async def run_candidate_checks(
    checks: tuple[TeamCheck, ...],
    workspace: TeamWorkspace,
    *,
    config: PaiCliConfig,
    approval_callback=None,
) -> AsyncIterator[tuple[TeamCheck, TeamCheckResult]]:
    """串行检查复用单个隔离检查副本以减少 worktree 抖动；并行安全检查按并发上限并发执行。"""
    limit = max(1, config.team.max_active_workers)
    approvals = asyncio.Lock()

    async def approve(request):
        async with approvals:
            decision = approval_callback(request)
            return await decision if isawaitable(decision) else decision

    serial_approve = approval_callback
    parallel_approve = approve if approval_callback else None

    for independent, batch in groupby(checks, key=lambda check: check.parallel_safe):
        pending = tuple(batch)
        if not independent or len(pending) < 2 or limit == 1:
            # 同一串行批次共享一个检查 worktree；每条检查间只在产生脏文件时清理，避免反复新建/删除 worktree。
            with workspace.reusable_check_workspace() as (candidate, reset_if_dirty):
                for idx, check in enumerate(pending):
                    if idx > 0:
                        await asyncio.to_thread(reset_if_dirty)
                    result = await run_team_check(
                        check,
                        candidate,
                        config=config,
                        approval_callback=serial_approve,
                        revision=workspace.candidate_revision,
                        project_root=workspace.repository,
                    )
                    yield check, result
            continue

        semaphore = asyncio.Semaphore(limit)

        async def _run_parallel(check: TeamCheck) -> tuple[TeamCheck, TeamCheckResult]:
            async with semaphore:
                cwd = await asyncio.to_thread(workspace.open_check_workspace)
                try:
                    result = await run_team_check(
                        check,
                        cwd,
                        config=config,
                        approval_callback=parallel_approve,
                        revision=workspace.candidate_revision,
                        project_root=workspace.repository,
                    )
                finally:
                    await asyncio.to_thread(workspace.close_check_workspace, cwd)
                return check, result

        outcomes = await asyncio.gather(*(_run_parallel(check) for check in pending))
        for check, result in outcomes:
            yield check, result


async def run_team_check(
    check: TeamCheck,
    cwd: str | Path,
    *,
    config: PaiCliConfig,
    approval_callback=None,
    revision: str | None = None,
    project_root: str | Path | None = None,
) -> TeamCheckResult:
    """Check one process result with fixed argv and a bounded, recorded observation."""
    extra_roots = (project_root,) if project_root is not None else ()
    try:
        argv = _checked_argv(check.argv, Path(cwd), config, extra_roots=extra_roots)
        if config.policy.hitl_mode != "never":
            decision = (
                approval_callback(
                    {
                        "tool_name": "team_check",
                        "description": "Run a behavioral check against this Team working copy",
                        "danger_level": "high",
                        "input": {
                            "id": check.id,
                            "argv": list(check.argv),
                            "cwd": str(cwd),
                        },
                    }
                )
                if approval_callback
                else "deny"
            )
            if isawaitable(decision):
                decision = await decision
            if decision != "approve":
                return TeamCheckResult(
                    False, f"{check.id}: execution not approved", status="unavailable"
                )
        timeout = min(120.0, max(0.1, float(config.tools.timeout)))
        env = {
            key: value for key, value in os.environ.items() if key.upper() in _ENV_KEYS
        }
        env = augment_env_for_venv(env, cwd, extra_roots=extra_roots)
        env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **({} if os.name == "nt" else {"start_new_session": True}),
        )
    except (OSError, ValueError) as exc:
        return TeamCheckResult(False, f"{check.id}: {exc}", status="unavailable")

    out, err = bytearray(), bytearray()
    readers = [
        asyncio.create_task(_read_output(process.stdout, out)),
        asyncio.create_task(_read_output(process.stderr, err)),
    ]
    wait = asyncio.create_task(process.wait())
    truncated = [False, False]
    timed_out = False
    try:
        async with asyncio.timeout(timeout):
            # Keep the reader and waiter tasks available to drain after a timeout.
            await asyncio.gather(*(asyncio.shield(task) for task in [*readers, wait]))
        truncated = [task.result() for task in readers]
    except TimeoutError:
        timed_out = True
        await _stop_process(process)
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    finally:
        for task in [*readers, wait]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*readers, wait, return_exceptions=True)

    stdout, stderr = (data.decode("utf-8", errors="replace") for data in (out, err))
    stdout = stdout.replace("\r\n", "\n")
    contains = getattr(check, "stdout_contains", None)
    equals = getattr(check, "stdout_equals", None)
    expected = check.expected_exit
    if timed_out:
        status, reason = "unavailable", f"timed out after {timeout:g}s"
    elif process.returncode != expected:
        status, reason = "failed", f"expected exit {expected}, got {process.returncode}"
    elif truncated[0] and (contains is not None or equals is not None):
        status, reason = "unavailable", "stdout exceeded the capture limit"
    elif contains is not None and contains not in stdout:
        status, reason = "failed", f"stdout missing {contains!r}"
    elif equals is not None and stdout != equals.replace("\r\n", "\n"):
        status, reason = "failed", f"expected stdout {equals!r}"
    else:
        status, reason = "passed", f"matched expected exit {expected} and output"

    if truncated[0]:
        stdout += "\n[stdout truncated]"
    if truncated[1]:
        stderr += "\n[stderr truncated]"
    observation = f"stdout:\n{stdout}\nstderr:\n{stderr}"
    preview = observation
    if len(preview) > _PREVIEW_LIMIT:
        preview = "…\n" + (
            preview[:_PREVIEW_LIMIT]
            if status == "passed"
            else preview[-_PREVIEW_LIMIT:]
        )
    detail = f"{check.id}: {reason[:_PREVIEW_LIMIT]}\n{preview}"
    run = current_run()
    evidence = None
    if run:
        evidence = run.save_text(
            "team-check",
            f"{check.id} ({revision or 'uncommitted'}): {reason}\n{observation}",
        )
        run.record(
            "team.check",
            check_id=check.id,
            revision=revision,
            status=status,
            exit_code=process.returncode,
            output=evidence,
        )
    return TeamCheckResult(status == "passed", detail, evidence, status)


def _checked_argv(
    values: tuple[str, ...],
    cwd: Path,
    config: PaiCliConfig,
    *,
    extra_roots: tuple[str | Path, ...] = (),
) -> tuple[str, ...]:
    if not values or any(
        not isinstance(value, str) or "\0" in value for value in values
    ):
        raise ValueError("expected a non-empty argv of strings")
    program = values[0]
    if not program or program != program.strip():
        raise ValueError("argv[0] must name an executable")
    if config.policy.command_guard_enabled:
        CommandGuard(config.policy.command_blacklist).validate(shlex.join(values))
    name = Path(program).name.lower()
    executable = resolve_project_executable(program, cwd, extra_roots=extra_roots)
    if not executable:
        raise ValueError(f"executable not found: {program}")
    if (
        Path(name).stem in _SHELLS
        or Path(executable).suffix.lower() in {".bat", ".cmd"}
        or Path(executable).resolve().stem.lower() in _SHELLS
    ):
        raise ValueError(
            "use an executable and argument list rather than a shell or batch file"
        )
    return (executable, *values[1:])


async def _read_output(stream: asyncio.StreamReader | None, output: bytearray) -> bool:
    truncated = False
    if stream is None:
        return truncated
    while chunk := await stream.read(8192):
        remaining = _CAPTURE_LIMIT - len(output)
        output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return truncated


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """End the check before workspace cleanup even when the caller is cancelled."""
    with suppress(OSError, ProcessLookupError, TimeoutError):
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            system_root = os.environ.get("SystemRoot", "C:/Windows")
            killer = await asyncio.create_subprocess_exec(
                str(Path(system_root) / "System32" / "taskkill.exe"),
                "/pid",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await asyncio.wait_for(killer.wait(), timeout=2)
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    with suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=2)
