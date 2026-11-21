"""普通会话可按需调用的规划与测试驱动修复。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from paicli.agent.collaboration import DELEGATION_SCHEMA, Collaboration
from paicli.agent.debug import DebugCoordinator, is_application_source
from paicli.agent.team_board import TeamPlan, parse_team_plan
from paicli.agent.team_workspace import TeamWorkspace
from paicli.harness.evidence import current_run
from paicli.harness.impact import MAX_FILES, import_names, local_dependencies, plan_impact
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
_PLAN_PROBE_FILES = 12
_PLAN_PROBE_BYTES = 64 * 1024
_IGNORE = {
    ".git",
    ".paicli",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "build",
    "dist",
    ".tox",
}


class SpecializedWorkflows:
    def __init__(self, client, collaboration: Collaboration):
        self.client = client
        self.collaboration = collaboration
        self.usage = Usage()
        self.calls = 0
        self.debug_started = False
        self.goal = ""
        self.requested_cases: list[str] = []
        self.regression_files: list[str] = []
        self.observed_cases: set[str] = set()
        self.source_files: dict[Path, tuple[int, int]] = {}
        self.source_imports: dict[str, set[str]] = {}

    def register(self, registry: ToolRegistry, *, auto_plan: bool) -> None:
        if delegate := registry.get("delegate_work"):
            registry.register(replace(delegate, handler=self.delegate))
        original = self.collaboration.coordinator.tool_registry
        for builtin in get_builtin_tools():
            base = original.get(builtin.name)
            tool = registry.get(builtin.name)
            if not base or not tool or base.handler is not builtin.handler:
                continue
            if auto_plan and tool.name == "read_file":
                registry.register(self._observe_source(tool, registry))
            elif tool.name in {"execute_command", "bash"}:
                registry.register(self._observe_pytest(tool, registry))
            elif self.requested_cases and tool.name in {"edit_file", "write_file"}:
                registry.register(self._verify_proposed_edit(tool))

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
            timeout=240,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        )

    def _observe_source(self, tool: Tool, registry: ToolRegistry) -> Tool:
        async def handler(data: dict, context: ToolContext) -> ToolResult:
            result = await tool.handler(data, context)
            if result.is_error or not str(data["path"]).endswith(".py"):
                return result
            root = Path(context.cwd).resolve()
            path = (root / str(data["path"])).resolve()
            if not path.is_relative_to(root):
                return result
            relative = path.relative_to(root)
            if not is_application_source(relative):
                return result
            version = _source_version(path)
            if version is None or self.source_files.get(path) == version:
                return result
            self.source_files[path] = version
            if registry.get("plan_change") or len(self.source_files) > _PLAN_PROBE_FILES:
                return result
            name = relative.as_posix()
            self.source_imports[name] = await asyncio.to_thread(_probe_imports, path, name)
            if registry.get("plan_change") is None and _shared_dependencies(self.source_imports):
                registry.register(
                    Tool(
                        "plan_change",
                        "Before a coupled Python change, inspect affected callers and tests. "
                        "Pass the files that define the contract to change. "
                        "For an authorized implementation with a ready, independent split, "
                        "optionally include a Team draft to start if the scan finds no gaps.",
                        object_schema(
                            {"paths": _PATHS, "team": DELEGATION_SCHEMA}, ["paths"]
                        ),
                        self.plan,
                        required_keys=["paths"],
                        is_concurrency_safe=False,
                    )
                )
            return result

        return replace(tool, handler=handler)

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
            targets = _local_test_references(
                root, _PYTEST_FAILURE.findall(result.content)
            )
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
                case for case in self.requested_cases
                if case.split("::", 1)[0] not in self.regression_files
            ]
            if len(tests) > 12:
                return await tool.handler(data, context)
            # 测试列表来自本轮用户请求和真实失败记录，不沿用其他会话的目标。
            payload = {"goal": self.goal, "tests": tests}
            if await ToolExecutor.approval_decision(debug_tool, payload, context) != "approve":
                return ToolResult("隔离调试未获批准，未应用候选编辑。", is_error=True)
            proposal = (tool.name, {**data, "path": path.relative_to(root).as_posix()})
            result = await self.run_debug(payload, context, proposed_edit=proposal)
            if not result.is_error:
                group.lead_edits.add(path)
            return result

        return replace(tool, handler=handler, timeout=240)

    def _pending_work(self) -> bool:
        group = self.collaboration
        return group.job is not None and bool(group.work_paths) and not group.applied

    async def plan(self, data: dict, context: ToolContext) -> ToolResult:
        return await _route("plan", _RouteContext(self, data, context))

    async def delegate(self, data: dict, context: ToolContext) -> ToolResult:
        return await _route("team", _RouteContext(self, data, context))

    async def run_debug(
        self, data: dict, context: ToolContext, *, proposed_edit: tuple[str, dict] | None = None
    ) -> ToolResult:
        return await _route("debug", _RouteContext(self, data, context, proposed_edit))

    async def analyze(self, data: dict, context: ToolContext) -> _PlanReport:
        root = Path(context.cwd).resolve()
        paths = _python_files(root, data["paths"], "paths")
        _route_record("workflow.started", workflow="plan", paths=paths)
        status = "failed"
        try:
            known = frozenset(
                path.relative_to(root).as_posix()
                for path, version in self.source_files.items()
                if _source_version(path) == version
            )
            plan = await asyncio.to_thread(_plan_change, root, paths, known)
            status = "completed"
            return plan
        finally:
            _route_record("workflow.finished", workflow="plan", status=status)

    async def debug(
        self, data: dict, context: ToolContext, *, proposed_edit: tuple[str, dict] | None = None
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
                python = (
                    root
                    / ".venv"
                    / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                )
                coordinator = DebugCoordinator(
                    llm_client=self.client,
                    tool_registry=group.coordinator.tool_registry,
                    config=context.config,
                    cwd=str(checkout / prefix),
                    targets=targets,
                    approval_callback=context.approval_callback,
                    max_models=12,
                    max_tokens=180_000,
                    max_tools=36,
                    python_executable=str(python) if python.is_file() else None,
                    project_lock_root=str(root),
                    prefer_final_verify=True,
                    proposed_edit=proposed_edit,
                )
                done = None
                async with aclosing(coordinator.run(goal)) as stream:
                    try:
                        async for event in stream:
                            if event["type"] == "done":
                                done = event
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
                if (
                    done["status"] != "completed"
                    or verification.get("status") != "accepted"
                ):
                    status = done["status"]
                    reason = done.get("error") or verification.get("reason", "")
                    return ToolResult(
                        f"调试未交付，主工作区未应用补丁。{reason}\n"
                        + json.dumps(detail, ensure_ascii=False),
                        is_error=True,
                    )

                paths = tuple(
                    Path(path).relative_to(checkout).as_posix()
                    for path in coordinator.changes
                )
                patch = workspace.capture("debug", paths)
                if not patch.patch:
                    return ToolResult("测试通过，但未生成可应用的修复。", is_error=True)
                workspace.apply_checked(patch.patch, source, lead_files, ())
                status = "completed"
                detail["paths"] = patch.paths
                return ToolResult(
                    "经独立回归验收的 Debug 补丁已应用。已覆盖下列测试，无须重复执行。\n"
                    + json.dumps(targets, ensure_ascii=False) + "\n"
                    + json.dumps(detail, ensure_ascii=False)
                )
        finally:
            _route_record(
                "workflow.finished", workflow="debug", status=status, **detail
            )


@dataclass(frozen=True, slots=True)
class _PlanReport:
    text: str
    affected: frozenset[str]
    unread: tuple[str, ...]
    incomplete: bool


@dataclass(slots=True)
class _RouteContext:
    workflows: SpecializedWorkflows
    arguments: dict
    tools: ToolContext
    proposed_edit: tuple[str, dict] | None = None


class _RouteState(TypedDict, total=False):
    requested: Literal["plan", "team", "debug"]
    next_step: Literal["inspect", "team", "debug", "lead"]
    analysis: _PlanReport
    delegation: TeamPlan
    result: ToolResult


async def _select_route(state: _RouteState, runtime: Runtime[_RouteContext]) -> _RouteState:
    context = runtime.context
    workflows = context.workflows
    requested = state["requested"]
    if requested == "team":
        return {"next_step": "team"}
    if requested == "debug":
        if workflows._pending_work():
            return {
                "next_step": "lead",
                "result": ToolResult(
                    "当前还有未应用的写任务组。请先收取、应用或放弃该组再修复。",
                    is_error=True,
                ),
            }
        return {"next_step": "debug"}
    if "analysis" not in state:
        return {"next_step": "inspect"}

    analysis = state["analysis"]
    proposal = context.arguments.get("team")
    if proposal is None:
        return {"next_step": "lead", "result": ToolResult(analysis.text)}
    try:
        team = workflows.collaboration
        if workflows._pending_work() or (team.job is not None and not team.collected):
            raise ValueError("已有未收取或尚待应用的协作组")
        if analysis.incomplete or analysis.unread:
            raise ValueError("还有未确认的影响范围")
        if not (Path(context.tools.cwd) / ".git").exists():
            raise ValueError("合并调用需要在 Git 项目根目录执行")
        draft = parse_team_plan(json.dumps(proposal, ensure_ascii=False))
        if (
            team.coordinator.config.team.max_active_workers < 2
            or sum(not task.needs for task in draft.tasks) < 2
        ):
            raise ValueError("没有可同时启动的独立任务")
        if any(path not in analysis.affected for task in draft.tasks for path in task.paths):
            raise ValueError("有任务文件超出本次明确的影响范围")
    except ValueError as exc:
        _route_record("workflow.routed", workflow="plan", destination="lead", reason=str(exc))
        return {
            "next_step": "lead",
            "result": ToolResult(f"{analysis.text}\n协作组未启动：{exc}。由主 Agent 继续处理。"),
        }
    _route_record("workflow.routed", workflow="plan", destination="team")
    return {"next_step": "team", "delegation": draft}


async def _inspect_plan(_state: _RouteState, runtime: Runtime[_RouteContext]) -> _RouteState:
    context = runtime.context
    return {"analysis": await context.workflows.analyze(context.arguments, context.tools)}


async def _start_team(state: _RouteState, runtime: Runtime[_RouteContext]) -> _RouteState:
    context = runtime.context
    collaboration = context.workflows.collaboration
    if "delegation" not in state:
        return {"result": await collaboration.start(context.arguments, context.tools)}
    result = await collaboration.start(
        context.arguments["team"], context.tools, plan=state["delegation"]
    )
    return {
        "result": replace(result, content=f"{state['analysis'].text}\n{result.content}"),
    }


async def _run_debug(_state: _RouteState, runtime: Runtime[_RouteContext]) -> _RouteState:
    context = runtime.context
    result = await context.workflows.debug(
        context.arguments, context.tools, proposed_edit=context.proposed_edit
    )
    return {"result": result}


def _build_router():
    graph = StateGraph(_RouteState, context_schema=_RouteContext)
    graph.add_node("route", _select_route)
    graph.add_node("inspect", _inspect_plan)
    graph.add_node("team", _start_team)
    graph.add_node("debug", _run_debug)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["next_step"],
        {"inspect": "inspect", "team": "team", "debug": "debug", "lead": END},
    )
    graph.add_edge("inspect", "route")
    graph.add_edge("team", END)
    graph.add_edge("debug", END)
    return graph.compile()


_ROUTER = _build_router()


async def _route(
    requested: Literal["plan", "team", "debug"], context: _RouteContext
) -> ToolResult:
    state = await _ROUTER.ainvoke({"requested": requested}, context=context)
    return state["result"]


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
        filename, separator, node = (
            value.partition("::") if allow_node_ids else (value, "", "")
        )
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.suffix != ".py" or not path.is_file():
            raise ValueError(f"找不到项目内的 Python 文件：{value}")
        normalized.append(path.relative_to(root).as_posix() + separator + node)
    return tuple(dict.fromkeys(normalized))


def _probe_imports(path: Path, relative: str) -> set[str]:
    try:
        with path.open("rb") as stream:
            contents = stream.read(_PLAN_PROBE_BYTES + 1)
        if len(contents) <= _PLAN_PROBE_BYTES:
            return import_names(relative, contents.decode("utf-8-sig"))
    except (OSError, SyntaxError, ValueError, RecursionError):
        pass
    return set()


def _source_version(path: Path) -> tuple[int, int] | None:
    try:
        facts = path.stat()
        return facts.st_mtime_ns, facts.st_size
    except OSError:
        return None


def _shared_dependencies(imports: dict[str, set[str]]) -> bool:
    dependencies = local_dependencies(imports)
    # 共享模块或跨两层的调用链才值得交给 Plan，多读了几个文件本身不是理由。
    return any(
        len(targets | {source for source, edges in dependencies.items() if name in edges}) >= 2
        for name, targets in dependencies.items()
    )


def _plan_change(root: Path, paths: tuple[str, ...], known: frozenset[str]) -> _PlanReport:
    catalogue = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(
            name for name in dirs if name not in _IGNORE and not name.startswith(".")
        )
        catalogue.extend(
            (Path(directory) / name).relative_to(root).as_posix()
            for name in sorted(files)
            if name.endswith(".py")
        )
        if len(catalogue) > MAX_FILES:
            break
    impact = plan_impact(root, paths, [], catalogue, include_imports=True)
    relations = impact["affected_imports"]
    edges = [
        f"{name} imports {', '.join(dependencies)}"
        for name, dependencies in relations.items()
        if dependencies
    ][:20]
    unread = [name for name in relations if name not in known and not _looks_like_test(Path(name))]
    targets = impact["added_test_files"]
    text = (
        f"1. 核对起始文件的契约与兼容行为：{', '.join(paths)}。\n"
        "2. 按下列导入关系检查调用方，将需要协同的修改一起完成。\n"
        + ("\n".join(edges) or "未找到静态调用关系。")
        + "\n"
        + f"3. 检查实际业务行为，并运行相关用例：{', '.join(targets) or '由主 Agent 查找'}。\n"
        + f"{impact['scanned_files']} 个文件已扫描，静态导入仅作调查线索。\n"
        + (f"优先核对尚未读取的文件：{', '.join(unread[:10])}。\n" if unread else "")
        + (f"扫描限制：{'; '.join(impact['unknown'][:3])}" if impact['unknown'] else "")
    )
    return _PlanReport(
        text, frozenset(relations), tuple(unread), bool(impact["blocked"] or impact["unknown"])
    )


def _counts(value: dict) -> dict:
    return {key: value[key] for key in ("passed", "total") if key in value}


def _local_test_references(root: Path, candidates: list[str]) -> list[str]:
    selected = []
    for case in dict.fromkeys(candidates):
        try:
            test, = _python_files(root, [case.rstrip(".。;；!?！？，")], "tests", allow_node_ids=True)
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
