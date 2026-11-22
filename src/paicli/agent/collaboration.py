"""普通会话的按需委派。主 Agent 制定分工，协作执行器验证并交还代码。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from inspect import isawaitable
from pathlib import Path
from typing import Any

from paicli.agent.team import TeamCoordinator, _TeamBudget, _TeamRun
from paicli.agent.team_board import (
    TeamCheck,
    TeamPlan,
    normalize_team_path,
    parse_team_check,
    parse_team_plan,
)
from paicli.agent.team_workspace import TeamSource, TeamWorkspace
from paicli.harness.evidence import current_run
from paicli.harness.execution import RunStopped
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.builtins import get_builtin_tools
from paicli.tools.file_ops import resolve_path
from paicli.tools.registry import ToolRegistry

LEAD_INSTRUCTIONS = """
Lead the user's task from start to finish.
1. Fast orientation: On multi-file Python repos, call `code_map` early to see symbols, signatures,
   import edges, and candidate tests in one step instead of reading files serially.
2. Parallel read-only audit/research: When investigating or auditing 2–4 independent subsystems or
   filesets, prefer calling `explore` with 1 question per subsystem so they run concurrently without
   bloating your main context.
3. Parallel multi-module implementation (`delegate_work`):
   - Handle single-file fixes or tightly coupled 1–2 file edits directly yourself.
   - Whenever a task requires implementing or migrating **2 or more independent files/modules** with
     clear contracts (e.g., multiple parsers, adapters, channels, handlers, or validators), **prefer
     `delegate_work`** to build those modules in parallel Git worktrees while you handle the shared
     entrypoint/registry/dispatcher file in the foreground!
   - Put key shared data models, function signatures, and edge-case rules in `shared_context` so
     every Worker starts immediately without cold-reading the whole repository.
   - Give each Worker exact repository-relative writable `paths` (non-overlapping). `checks` is
     optional: omit it for automatic pytest/compile verification, or supply concise per-owner and
     integration checks.
   - Call `collect_work` once your own integration edits are ready; if checks fail, fix your files
     or let Workers repair on recheck, or adjust your own check via `correct_checks`.
   - Call `apply_work` after `collect_work` succeeds, then report verified results. If a group cannot
     succeed, discard it (`collect_work(discard=true)`) and finish cleanly yourself.
4. Discipline: Respect read-only requests (never write files). Ask clarifying questions only when
   truly blocked. Never claim tests passed or files changed unless actually executed and applied.
""".strip()

_STRING = {"type": "string"}
_STRINGS = {"type": "array", "items": _STRING}
_TASK = object_schema(
    {
        "id": _STRING,
        "goal": _STRING,
        "paths": _STRINGS,
        "needs": _STRINGS,
        "publishes": _STRING,
    },
    ["id", "goal", "paths"],
)
_CHECK = object_schema(
    {
        "id": _STRING,
        "argv": _STRINGS,
        "owner": {
            "type": "string",
            "description": "Worker task ID for local checks. Omit for lead-owned integration checks.",
        },
        "expected_exit": {"type": "integer"},
        "stdout_contains": _STRING,
        "stdout_equals": _STRING,
        "parallel_safe": {
            "type": "boolean",
            "description": "True only for independent checks with no shared external state.",
        },
    },
    ["id", "argv"],
)
DELEGATION_SCHEMA = object_schema(
    {
        "goal": _STRING,
        "shared_context": {
            "type": "string",
            "description": (
                "Concise shared contracts, data types, signatures, and test requirements injected "
                "identically to all Workers so they avoid cold re-reading and share KV prefix cache."
            ),
        },
        "tasks": {"type": "array", "items": _TASK},
        "checks": {"type": "array", "items": _CHECK},
        "lead_paths": _STRINGS,
    },
    ["goal", "tasks"],
)


class Collaboration:
    """一轮主会话可以按需启动、收取和应用多个先后执行的任务组。"""

    def __init__(
        self,
        *,
        llm_client,
        registry,
        config,
        cwd,
        approval_callback,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._approver = approval_callback
        self._approval_lock = asyncio.Lock()
        self.approval = self._approve if approval_callback is not None else None
        self._event_sink = event_sink
        self.coordinator = TeamCoordinator(
            llm_client=llm_client,
            tool_registry=registry,
            config=config,
            cwd=cwd,
            approval_callback=self.approval,
        )
        self.budget = _TeamBudget(config.team)
        self.workspace: TeamWorkspace | None = None
        self.job: asyncio.Task | None = None
        self.flow: _TeamRun | None = None
        self.ready = asyncio.Event()
        self.source: TeamSource | None = None
        self.integration_revision: str | None = None
        self.work_paths: tuple[str, ...] = ()
        self.lead_paths: tuple[str, ...] = ()
        self.lead_edits: set[Path] = set()
        self.collected = False
        self.applied = False
        self.corrected = False
        self.result: tuple[str, str] | None = None
        self.group = 0
        self.deadline = 0.0

    def _emit(self, event: dict[str, Any]) -> None:
        if self._event_sink is not None and event.get("type") == "team_status":
            self._event_sink(event)

    async def _approve(self, request):
        async with self._approval_lock:
            result = self._approver(request)
            return await result if isawaitable(result) else result

    def registry(self) -> ToolRegistry:
        registry = self.coordinator.tool_registry.copy()
        for builtin in get_builtin_tools():
            if builtin.name in {"edit_file", "write_file"}:
                tool = registry.get(builtin.name)
                if tool and tool.handler is builtin.handler:
                    registry.register(self._observe_edit(tool))
        registry.register_all(
            [
                Tool(
                    "delegate_work",
                    "Start scoped parallel Workers in isolated Git worktrees. Provide shared_context "
                    "with key contracts/types and exact Git-root-relative paths per task (empty paths "
                    "means read-only). `checks` is optional (auto-inferred from tests/ when omitted).",
                    DELEGATION_SCHEMA,
                    self.start,
                    is_concurrency_safe=False,
                    required_keys=["goal", "tasks"],
                ),
                Tool(
                    "collect_work",
                    "After your own edits, await the current group and validate the combined candidate."
                    " On failure, recheck after fixing your files. If a check you wrote contradicts"
                    " the requirements, give a replacement with the same ID and explain why."
                    " Set discard only to cancel or abandon this group and continue yourself.",
                    object_schema(
                        {
                            "discard": {"type": "boolean"},
                            "correct_checks": {"type": "array", "items": _CHECK},
                            "reason": _STRING,
                        }
                    ),
                    self.collect,
                    is_concurrency_safe=False,
                    timeout=None,
                ),
                Tool(
                    "apply_work",
                    "Apply the collected, verified Worker patch if the checked source state still matches.",
                    object_schema({}),
                    self.apply,
                    is_read_only=False,
                    is_concurrency_safe=False,
                    danger_level="medium",
                    requires_approval=True,
                ),
            ]
        )
        return registry

    def _observe_edit(self, tool: Tool) -> Tool:
        async def handler(data: dict, context: ToolContext) -> ToolResult:
            if data.get("dry_run") is True:
                return await tool.handler(data, context)
            target = resolve_path(
                context.cwd,
                str(data["path"]),
                context.config.policy.path_guard_enabled,
            )
            workspace = self.workspace
            if (
                workspace
                and not self.applied
                and target.is_relative_to(workspace.repository)
            ):
                path = target.relative_to(workspace.repository).as_posix().casefold()
                if any(path == name.casefold() for name in self.work_paths):
                    return ToolResult(
                        "此文件当前由 Worker 负责。请先收取并应用产物，或明确放弃委派后再改。",
                        is_error=True,
                    )
            result = await tool.handler(data, context)
            if not result.is_error:
                self.lead_edits.add(target)
            return result

        return replace(tool, handler=handler)

    def _lead_files(
        self, workspace: TeamWorkspace, declared: tuple[str, ...]
    ) -> tuple[str, ...]:
        files = set(declared)
        for target in self.lead_edits:
            if target.is_relative_to(workspace.repository):
                path = target.relative_to(workspace.repository).as_posix()
                try:
                    files.add(normalize_team_path(path))
                except ValueError:
                    continue
        return tuple(sorted(files))

    @staticmethod
    def _record(event: str, **fields: Any) -> None:
        if run := current_run():
            run.record(event, **fields)

    async def start(
        self, data: dict, _context: ToolContext, *, plan: TeamPlan | None = None
    ) -> ToolResult:
        if self.job is not None and not self.collected:
            return ToolResult(
                "已有协作组。先用 collect_work 收取或放弃。", is_error=True
            )

        goal = data["goal"]
        if not isinstance(goal, str) or not goal.strip():
            return ToolResult("请提供清楚的协作目标", is_error=True)
        if plan is None:
            plan = parse_team_plan(
                json.dumps(data, ensure_ascii=False),
                allow_empty_checks=True,
            )
        paths = tuple(path for task in plan.tasks for path in task.paths)
        if paths and not plan.checks:
            plan = replace(plan, checks=self._infer_default_checks(paths))
        shared_context = str(data.get("shared_context") or "").strip()[:8_000]
        if (
            paths
            and self.approval is None
            and self.coordinator.config.policy.hitl_mode != "never"
        ):
            return ToolResult(
                "当前入口未提供审批回调。请直接处理，或在交互式会话中审批协作检查和补丁。",
                is_error=True,
            )
        raw_lead = data.get("lead_paths", [])
        if not isinstance(raw_lead, list):
            return ToolResult("lead_paths 必须是文件路径列表", is_error=True)
        owned = {path.casefold() for path in paths}
        lead = tuple(dict.fromkeys(normalize_team_path(path) for path in raw_lead))
        if any(path.casefold() in owned for path in lead):
            return ToolResult("主 Agent 与 Worker 的写范围不可重叠", is_error=True)
        if self.job is not None:
            await self.close()

        workspace = TeamWorkspace(self.coordinator.cwd)
        workspace.__enter__()
        try:
            workspace.base = workspace.freeze_source(
                workspace.source_state(self._lead_files(workspace, lead), paths)
            )
        except BaseException:
            workspace.__exit__(None, None, None)
            raise

        self.workspace = workspace
        self.work_paths, self.lead_paths = paths, lead
        self.ready = asyncio.Event()
        self.source = None
        self.result = None
        self.integration_revision = None
        self.collected = self.applied = self.corrected = False
        self.group += 1
        self.deadline = asyncio.get_running_loop().time() + max(
            1.0, self.coordinator.config.team.timeout
        )
        flow = self.flow = _TeamRun(
            self.coordinator,
            goal.strip(),
            workspace,
            self.budget,
            self._emit,
            shared_context=shared_context,
        )
        self._record(
            "collaboration.started",
            group=self.group,
            tasks=[task.id for task in plan.tasks],
        )

        self.job = asyncio.create_task(
            self._bounded(
                lambda: flow.execute(
                    plan, integration_base=self._integration_base if paths else None
                )
            )
        )
        names = "、".join(task.id for task in plan.tasks)
        return ToolResult(
            f"协作组 {self.group} 已后台启动（{names}）。你可以继续处理自己的部分，"
            "再调用 collect_work 收取和检查组合结果。"
        )

    async def _integration_base(self) -> str:
        await self.ready.wait()
        assert self.integration_revision is not None
        return self.integration_revision

    async def _bounded(
        self,
        operation: Callable[[], Awaitable[tuple[str, str]]],
    ) -> tuple[str, str]:
        try:
            async with asyncio.timeout_at(self.deadline):
                return await operation()
        except TimeoutError:
            return "timed_out", "协作任务已超时"
        except RunStopped as exc:
            return exc.status, str(exc)
        except Exception as exc:
            return "failed", str(exc)

    async def collect(self, data: dict, _context: ToolContext) -> ToolResult:
        if self.job is None:
            return ToolResult("当前会话没有已委派的协作任务", is_error=True)
        if data.get("discard") is True:
            await self.close()
            self._record("collaboration.discarded", group=self.group)
            return ToolResult("协作组已放弃。请根据当前主工作区继续。")
        if self.collected:
            status, outcome = await self._recheck(data)
        else:
            if data.get("correct_checks"):
                return ToolResult("请先收取组合检查的失败结果", is_error=True)
            if self.work_paths and not self.ready.is_set() and not self.job.done():
                assert self.workspace is not None
                self.source = self.workspace.source_state(
                    self._lead_files(self.workspace, self.lead_paths), self.work_paths
                )
                self.integration_revision = self.workspace.freeze_source(self.source)
                self.ready.set()
            status, outcome = await asyncio.shield(self.job)
        self.result = status, outcome
        self.collected = True
        self._record("collaboration.collected", group=self.group, status=status)
        return ToolResult(outcome, is_error=status != "completed")

    async def _recheck(self, data: dict) -> tuple[str, str]:
        assert (
            self.workspace is not None
            and self.flow is not None
            and self.result is not None
        )
        corrections = data.get("correct_checks", [])
        if (
            self.applied
            or not self.work_paths
            or self.workspace.candidate_revision is None
        ):
            if corrections:
                raise ValueError("当前协作组没有可重新检查的候选版本")
            return self.result
        source = self.workspace.source_state(
            self._lead_files(self.workspace, self.lead_paths),
            self.work_paths,
        )
        if source == self.source and not corrections:
            return self.result
        revision = self.workspace.freeze_source(source)
        if corrections:
            self._correct_failed_checks(corrections, data.get("reason"))
        self.source, self.integration_revision = source, revision
        self.deadline = asyncio.get_running_loop().time() + max(
            1.0, self.coordinator.config.team.timeout
        )
        return await self._bounded(
            lambda: self.flow.verify(self._integration_base, repair_workers=True)
        )

    def _infer_default_checks(self, paths: tuple[str, ...]) -> tuple[TeamCheck, ...]:
        """当主 Agent 省略 checks 时自动推断默认验证，降低并行委派门槛。"""
        root = Path(self.coordinator.cwd).resolve()
        if (root / "tests").is_dir() or any(root.glob("test_*.py")):
            return (
                TeamCheck(
                    id="auto_pytest",
                    argv=("python", "-m", "pytest", "-q"),
                    expected_exit=0,
                ),
            )
        py_files = [p for p in paths if p.endswith(".py")]
        if py_files:
            return (
                TeamCheck(
                    id="auto_compile",
                    argv=("python", "-m", "py_compile", *py_files[:12]),
                    expected_exit=0,
                ),
            )
        return (
            TeamCheck(
                id="auto_noop",
                argv=("python", "-c", "print('ok')"),
                expected_exit=0,
            ),
        )

    def _correct_failed_checks(self, corrections, reason) -> None:
        assert self.flow is not None
        if self.corrected:
            raise ValueError("每组只允许按原需求更正一次检查")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1500:
            raise ValueError("请说明原检查和用户需求冲突的依据")
        if not isinstance(corrections, list) or len(corrections) > 20:
            raise ValueError("correct_checks 必须是至多 20 项检查的列表")
        board = self.flow.board
        original = {check.id: check for check in board.plan.checks}
        replacements = {}
        for data in corrections:
            check = parse_team_check(data)
            failure = self.flow.failed_checks.get(check.id)
            if (
                check.id in replacements
                or check.id not in original
                or failure is None
                or failure.status != "failed"
            ):
                raise ValueError("只能更正失败的原计划检查，每个检查编号限一次")
            check = replace(check, owner=original[check.id].owner)
            if check == original[check.id]:
                raise ValueError(f"检查 {check.id} 未修改")
            replacements[check.id] = check
        board.plan = replace(
            board.plan,
            checks=tuple(
                replacements.get(check.id, check) for check in board.plan.checks
            ),
        )
        self.corrected = True
        self._record(
            "collaboration.checks_corrected",
            group=self.group,
            checks=list(replacements),
            reason=reason.strip(),
        )

    async def apply(self, _data: dict, _context: ToolContext) -> ToolResult:
        if self.job is None:
            return ToolResult("当前会话没有已委派的协作任务", is_error=True)
        if self.applied:
            return ToolResult("这一组改动已经应用")
        if self.result is None or self.result[0] != "completed":
            return ToolResult("请先收取通过组合检查的结果", is_error=True)
        assert self.workspace is not None and self.flow is not None
        if not self.flow.patch:
            return ToolResult("这一组没有代码补丁")
        assert self.source is not None
        self.workspace.apply_checked(
            self.flow.patch,
            self.source,
            self._lead_files(self.workspace, self.lead_paths),
            self.work_paths,
        )
        self.applied = True
        self._record(
            "collaboration.applied", group=self.group, revision=self.flow.revision
        )
        return ToolResult("组合检查通过的 Worker 改动已应用到当前工作区。")

    def pending_instruction(self) -> str:
        if self.job is not None and not self.collected:
            return (
                "运行状态：还有已委派的任务等待收取。调用 collect_work 查看结果，"
                "或者明确放弃这组委派后再完成。"
            )
        return ""

    async def close(self) -> None:
        if self.job is not None:
            if not self.job.done():
                self.job.cancel()
            await asyncio.gather(self.job, return_exceptions=True)
            self.job = None
        if self.workspace is not None:
            workspace, self.workspace = self.workspace, None
            workspace.__exit__(None, None, None)
