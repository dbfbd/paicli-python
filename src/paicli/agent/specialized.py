"""普通会话按需调用的感知调研（code_map / explore）与测试驱动隔离修复（Debug）。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any

from paicli.agent.collaboration import Collaboration
from paicli.agent.debug import DebugCoordinator, is_application_source
from paicli.agent.explore import ExploreRunner, make_explore_tool
from paicli.agent.team_workspace import TeamWorkspace
from paicli.harness.code_map import build_code_map
from paicli.harness.evidence import current_run
from paicli.harness.project_env import resolve_project_python
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.builtins import get_builtin_tools
from paicli.tools.executor import ToolExecutor
from paicli.tools.registry import ToolRegistry
from paicli.types import Usage

_TEXT = {"type": "string"}
_PATHS = {"type": "array", "items": _TEXT}
_PYTEST_FAILURE = re.compile(r"^FAILED\s+(\S+\.py::[^\s]+)", re.MULTILINE)
_PYTEST_REFERENCE = re.compile(r"""([\w./\\-]+\.py::[^\s\`'"，。；,]+)""")
_PYTEST_FILE = re.compile(r"([\w./\\-]+\.py)(?![\w:/\\-])")


class SpecializedWorkflows:
    """为主会话装配常驻代码感知工具（code_map / explore）与受控 Debug 隔离验收入口。

    直接以异步方式调度子流程，并将进度事件实时透传回主会话。
    """

    def __init__(
        self,
        client,
        collaboration: Collaboration,
        *,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.client = client
        self.collaboration = collaboration
        self.event_sink = event_sink
        self.usage = Usage()
        self.calls = 0
        self.debug_started = False
        self.goal = ""
        self.requested_cases: list[str] = []
        self.regression_files: list[str] = []
        self.observed_cases: set[str] = set()
        self._explorer: ExploreRunner | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def register(self, registry: ToolRegistry, *, auto_plan: bool) -> None:
        if auto_plan:
            code_map_tool = self._code_map_tool()
            registry.register(code_map_tool)
            self._explorer = ExploreRunner(
                llm_client=self.client,
                base_registry=self.collaboration.coordinator.tool_registry,
                config=self.collaboration.coordinator.config,
                cwd=self.collaboration.coordinator.cwd,
                event_sink=self.event_sink,
            )
            registry.register(make_explore_tool(self._explorer, code_map_tool=code_map_tool))

        original = self.collaboration.coordinator.tool_registry
        for builtin in get_builtin_tools():
            base = original.get(builtin.name)
            tool = registry.get(builtin.name)
            if not base or not tool or base.handler is not builtin.handler:
                continue
            if tool.name in {"execute_command", "bash"}:
                registry.register(self._observe_pytest(tool, registry))
            elif self.requested_cases and tool.name in {"edit_file", "write_file"}:
                registry.register(self._verify_proposed_edit(tool))

    @property
    def total_usage(self) -> Usage:
        return self.usage + (self._explorer.usage if self._explorer else Usage())

    @property
    def total_calls(self) -> int:
        return self.calls + (self._explorer.calls if self._explorer else 0)

    def _code_map_tool(self) -> Tool:
        return Tool(
            name="code_map",
            description=(
                "Fast zero-LLM AST repository outline and static impact analyzer. "
                "Omit `paths` for a whole-project symbol & import map; pass `paths` (files or "
                "directories) to see public signatures, reverse callers/importers, and candidate tests."
            ),
            parameters=object_schema(
                {
                    "paths": {
                        "type": "array",
                        "items": _TEXT,
                        "description": "Optional repository-relative Python files or directories to focus on",
                    }
                }
            ),
            handler=self.code_map,
            is_read_only=True,
            is_concurrency_safe=True,
        )

    def _register_debug(self, registry: ToolRegistry) -> None:
        if registry.get("debug_failure") is None:
            registry.register(self._debug_tool())

    def _debug_tool(self) -> Tool:
        return Tool(
            "debug_failure",
            "Repair a reproducible pytest regression in an isolated Git worktree. "
            "Include the failing case and passing regression targets. Applies verified source only.",
            object_schema({"goal": _TEXT, "tests": _PATHS}, ["goal", "tests"]),
            self.run_debug,
            required_keys=["goal", "tests"],
            timeout=300,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        )

    def routing_context(self, message: str, cwd: str) -> str:
        root = Path(cwd).resolve()
        self.goal = message
        self.requested_cases = _local_test_references(root, _PYTEST_REFERENCE.findall(message))
        if not self.requested_cases:
            return ""
        self.regression_files = _local_test_references(
            root, [name for name in _PYTEST_FILE.findall(message) if _looks_like_test(Path(name))]
        )
        return (
            "\nFor a repair, first reproduce the user-named pytest case(s): "
            f"{json.dumps(self.requested_cases, ensure_ascii=False)}. "
            "After a confirmed failure, propose your source edit for isolated verification. "
            "If it returns an accepted receipt, report it for the covered targets. "
            "Only rerun them if later edits invalidate the receipt."
        )

    def _observe_pytest(self, tool: Tool, registry: ToolRegistry) -> Tool:
        async def handler(data: dict, context: ToolContext) -> ToolResult:
            result = await tool.handler(data, context)
            if (
                self.debug_started
                or self._pending_work()
                or not result.is_error
                or "pytest" not in str(data.get("command", ""))
                or "exit_code=1" not in result.content.partition("\n")[0]
            ):
                return result
            root = Path(context.cwd).resolve()
            targets = _local_test_references(root, _PYTEST_FAILURE.findall(result.content))
            if targets and not self.requested_cases:
                self._register_debug(registry)
            self.observed_cases.update(targets)
            if self.observed_cases.intersection(self.requested_cases):
                result.content += (
                    "\nThe named failure is confirmed. A proposed application-source edit "
                    "will run through isolated regression verification."
                )
            return result

        return replace(tool, handler=handler)

    def _verify_proposed_edit(self, tool: Tool) -> Tool:
        debug_tool = self._debug_tool()

        async def handler(data: dict, context: ToolContext) -> ToolResult:
            group = self.collaboration
            if (
                self.debug_started
                or self.observed_cases.isdisjoint(self.requested_cases)
                or data.get("dry_run")
                or self._pending_work()
            ):
                return await tool.handler(data, context)
            root = Path(context.cwd).resolve()
            path = (root / str(data["path"])).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or not is_application_source(path.relative_to(root))
                or not any((parent / ".git").exists() for parent in (root, *root.parents))
            ):
                return await tool.handler(data, context)

            tests = self.regression_files + [
                case
                for case in self.requested_cases
                if case.split("::", 1)[0] not in self.regression_files
            ]
            if len(tests) > 12:
                return await tool.handler(data, context)
            payload = {"goal": self.goal, "tests": tests}
            if await ToolExecutor.approval_decision(debug_tool, payload, context) != "approve":
                return ToolResult("隔离调试未获批准，未应用候选编辑。", is_error=True)
            proposal = (tool.name, {**data, "path": path.relative_to(root).as_posix()})
            result = await self.run_debug(payload, context, proposed_edit=proposal)
            if not result.is_error:
                group.lead_edits.add(path)
            return result

        return replace(tool, handler=handler, timeout=300)

    def _pending_work(self) -> bool:
        group = self.collaboration
        return group.job is not None and bool(group.work_paths) and not group.applied

    async def code_map(self, data: dict, context: ToolContext) -> ToolResult:
        raw_paths = data.get("paths")
        paths = [str(p) for p in raw_paths] if isinstance(raw_paths, list) else None
        _route_record("workflow.started", workflow="code_map", paths=paths or [])
        status = "failed"
        try:
            report = await asyncio.to_thread(build_code_map, context.cwd, paths=paths)
            status = "completed"
            return ToolResult(report["text"])
        finally:
            _route_record("workflow.finished", workflow="code_map", status=status)

    async def run_debug(
        self,
        data: dict,
        context: ToolContext,
        *,
        proposed_edit: tuple[str, dict] | None = None,
    ) -> ToolResult:
        if self._pending_work():
            return ToolResult(
                "当前还有未应用的写任务组。请先收取、应用或放弃该组再修复。",
                is_error=True,
            )
        return await self.debug(data, context, proposed_edit=proposed_edit)

    async def debug(
        self,
        data: dict,
        context: ToolContext,
        *,
        proposed_edit: tuple[str, dict] | None = None,
    ) -> ToolResult:
        goal = _goal(data)
        root = Path(context.cwd).resolve()
        targets = _python_files(root, data["tests"], "tests", allow_node_ids=True)
        selected = tuple(dict.fromkeys(target.split("::", 1)[0] for target in targets))
        group = self.collaboration
        _route_record(
            "workflow.started", workflow="debug", tests=targets, from_edit=proposed_edit is not None
        )
        self.debug_started = True
        status = "failed"
        detail: dict = {}
        try:
            with TeamWorkspace(context.cwd) as workspace:
                prefix = root.relative_to(workspace.repository)
                declared = selected + ((proposed_edit[1]["path"],) if proposed_edit else ())
                lead_files = group._lead_files(
                    workspace, tuple((prefix / name).as_posix() for name in declared)
                )
                source = workspace.source_state(lead_files, ())
                workspace.base = workspace.freeze_source(source)
                checkout = Path(workspace.create("debug"))
                python = resolve_project_python(root, extra_roots=(workspace.repository,))
                debug_cfg = context.config.debug
                has_knowledge_index = (root / ".paicli" / "knowledge.db").is_file()
                coordinator = DebugCoordinator(
                    llm_client=self.client,
                    tool_registry=group.coordinator.tool_registry,
                    config=context.config,
                    cwd=str(checkout / prefix),
                    targets=targets,
                    approval_callback=context.approval_callback,
                    max_attempts=debug_cfg.max_attempts,
                    timeout=debug_cfg.timeout,
                    max_models=debug_cfg.max_models,
                    max_tokens=debug_cfg.max_tokens,
                    max_tools=debug_cfg.max_tools,
                    use_knowledge=has_knowledge_index,
                    python_executable=python,
                    project_lock_root=str(root),
                    prefer_final_verify=True,
                    proposed_edit=proposed_edit,
                )
                done = None
                async with aclosing(coordinator.run(goal)) as stream:
                    try:
                        async for event in stream:
                            etype = event.get("type")
                            if etype == "done":
                                done = event
                            elif etype in {"debug_started", "tool_call", "tool_result"}:
                                self._emit({**event, "phase": "debug"})
                            elif etype == "text_delta" and str(event.get("text", "")).startswith(
                                "Debug："
                            ):
                                self._emit(
                                    {
                                        "type": "debug_status",
                                        "message": str(event["text"]).strip(),
                                    }
                                )
                    finally:
                        if hasattr(coordinator, "execution"):
                            self.usage += coordinator.execution.usage
                            self.calls += int(coordinator.execution.data["models"])
                if done is None:
                    return ToolResult(
                        "调试结束时未返回验收结果，主工作区未应用补丁。", is_error=True
                    )

                verification = done["verification"]
                detail = {
                    "debug_run_id": done.get("run_id"),
                    "verification": verification.get("status"),
                    "verification_report": verification.get("report_path"),
                    "fail_to_pass": _counts(verification.get("fail_to_pass", {})),
                    "pass_to_pass": _counts(verification.get("pass_to_pass", {})),
                }
                if done["status"] != "completed" or verification.get("status") != "accepted":
                    status = done["status"]
                    reason = done.get("error") or verification.get("reason", "")
                    return ToolResult(
                        f"调试未交付，主工作区未应用补丁。{reason}\n"
                        + json.dumps(detail, ensure_ascii=False),
                        is_error=True,
                    )

                paths = tuple(
                    Path(path).relative_to(checkout).as_posix() for path in coordinator.changes
                )
                patch = workspace.capture("debug", paths)
                if not patch.patch:
                    return ToolResult("测试通过，但未生成可应用的修复。", is_error=True)
                workspace.apply_checked(patch.patch, source, lead_files, ())
                status = "completed"
                detail["paths"] = patch.paths
                return ToolResult(
                    "经独立回归验收的 Debug 补丁已应用。已覆盖下列测试，无须重复执行。\n"
                    + json.dumps(targets, ensure_ascii=False)
                    + "\n"
                    + json.dumps(detail, ensure_ascii=False)
                )
        finally:
            _route_record("workflow.finished", workflow="debug", status=status, **detail)


def _goal(data: dict) -> str:
    raw = data.get("goal")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("需要明确目标和预期行为")
    return raw.strip()


def _python_files(
    root: Path, values, field: str, *, allow_node_ids=False
) -> tuple[str, ...]:
    if not isinstance(values, list) or not 1 <= len(values) <= 12:
        raise ValueError(f"{field} 需要 1 至 12 个现有 Python 文件")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value or value.startswith("-"):
            raise ValueError(f"{field} 必须使用项目内文件路径")
        filename, separator, node = value.partition("::") if allow_node_ids else (value, "", "")
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.suffix != ".py" or not path.is_file():
            raise ValueError(f"找不到项目内的 Python 文件：{value}")
        normalized.append(path.relative_to(root).as_posix() + separator + node)
    return tuple(dict.fromkeys(normalized))


def _counts(value: dict) -> dict:
    return {key: value[key] for key in ("passed", "total") if key in value}


def _local_test_references(root: Path, candidates: list[str]) -> list[str]:
    selected = []
    for case in dict.fromkeys(candidates):
        try:
            (test,) = _python_files(
                root, [case.rstrip(".。;；!?！？，")], "tests", allow_node_ids=True
            )
        except ValueError:
            continue
        if test not in selected:
            selected.append(test)
            if len(selected) == 8:
                break
    return selected


def _looks_like_test(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_") or path.name.endswith("_test.py")


def _route_record(event: str, **fields) -> None:
    if run := current_run():
        run.record(event, **fields)
