"""Plan 模式入口：生成任务依赖图，并复用 LangGraph 工作流执行。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from inspect import isawaitable
from typing import Any

from paicli.agent.orchestrator import AgentMessage, AgentMessageType, AgentRole, SubAgent
from paicli.agent.workflow import TaskOutcome, WorkerResult, WorkflowRunner
from paicli.config import PaiCliConfig
from paicli.harness.evidence import current_run, evidence_context, recorded_run
from paicli.harness.execution import RunStopped
from paicli.llm.base import LlmClient
from paicli.llm.pricing import calculate_costs
from paicli.plan import ExecutionPlan, Planner, PlanStatus, Task, TaskStatus
from paicli.plan.inspection import inspect_plan
from paicli.plan.planner import is_simple_goal
from paicli.prompt import PromptAssembler
from paicli.snapshot import SnapshotService
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, Usage

_REPORT_EVIDENCE_TOOLS = {
    "read_file", "grep", "grep_code", "search_code", "glob", "glob_files",
    "list_dir", "directory_tree", "get_file_info", "bash", "execute_command",
}
_HANDOFF_TOOLS = _REPORT_EVIDENCE_TOOLS | {"edit_file", "write_file"}


@dataclass(frozen=True, slots=True)
class _Observation:
    tool: str
    text: str
    artifact: dict | None
    path: str = ""


class PlanExecuteAgent:
    """将模型规划结果转换为共享工作流任务，汇总子任务状态和用量。"""

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback=None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
        max_concurrent_tasks: int | None = None,
        manage_snapshots: bool = True,
        plan_approval: Callable[[ExecutionPlan], Awaitable[bool] | bool] | None = None,
        staged_planning: bool = True,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.max_concurrent_tasks = (
            config.plan.max_concurrent_tasks
            if max_concurrent_tasks is None
            else max_concurrent_tasks
        )
        self.manage_snapshots = manage_snapshots
        self.plan_approval = plan_approval
        self.staged_planning = staged_planning
        self.history: list[Message] = []
        self.workflow: WorkflowRunner | None = None
        self._discovery = ""
        self._reports: dict[str, str] = {}
        self._handoffs: dict[str, str] = {}
        self._revision_notes: dict[str, str] = {}

    @recorded_run("plan")
    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """先查仓库并展示计划；有经核实的阻塞时，允许一次局部重规划。"""
        self.workflow = None
        self._discovery = ""
        self._reports = {}
        self._handoffs = {}
        self._revision_notes = {}
        snapshot = None
        usage = Usage()
        total_turns = 0
        revised = False
        status = "failed"
        final_text = ""
        try:
            yield {"type": "text_delta", "text": f"正在规划任务：{message}\n\n"}
            if self.staged_planning and not is_simple_goal(message):
                yield {"type": "plan_status", "phase": "inspection"}
                async for event in inspect_plan(
                    goal=message,
                    llm_client=self.llm_client,
                    tool_registry=self.tool_registry,
                    config=self.config,
                    cwd=self.cwd,
                    approval_callback=self.approval_callback,
                ):
                    if event["type"] == "plan_evidence":
                        self._discovery = event["text"]
                        continue
                    if event["type"] == "usage":
                        usage += Usage.from_mapping(event.get("usage"))
                    yield event

            yield {"type": "plan_status", "phase": "planning"}
            plan: ExecutionPlan | None = None
            options = {"evidence": self._discovery} if self._discovery else {}
            async for event in self.planner.stream_plan(message, **options):
                if event.get("type") == "plan_created":
                    plan = event["plan"]
                    continue
                if event.get("type") == "usage":
                    usage += Usage.from_mapping(event.get("usage"))
                yield event
            if plan is None:
                raise ValueError("planner did not produce an execution plan")
            prior: dict[str, TaskOutcome] = {}
            while True:
                yield {"type": "text_delta", "text": _describe_plan(plan)}
                if self.plan_approval is not None:
                    yield {"type": "plan_status", "phase": "approval"}
                    decision = self.plan_approval(plan)
                    if isawaitable(decision):
                        decision = await decision
                    if decision is not True:
                        status = "cancelled"
                        break
                if self.manage_snapshots and snapshot is None:
                    snapshot = SnapshotService(self.cwd)
                    with suppress(Exception):
                        snapshot.create("pre-turn")

                yield {"type": "text_delta", "text": "开始执行计划……\n\n"}
                yield {"type": "plan_status", "phase": "execution"}
                plan.mark_started()
                self._reports.clear()

                async def worker(task, dependencies, read_only, feedback, emit):
                    return await self._run_task(plan, task, dependencies, read_only, feedback, emit)

                self.workflow = WorkflowRunner(
                    tasks=plan.all_tasks(),
                    worker=worker,
                    max_concurrency=self.max_concurrent_tasks,
                    prior_results=prior,
                )
                async for event in self.workflow.run():
                    yield event
                    if event.get("type") == "workflow_task_finished":
                        label = "已完成" if event["status"] == "completed" else "未完成"
                        content = event.get("error") or event.get("text") or event["status"]
                        progress = f"{label} [{event['task_id']}]：{_preview(str(content))}\n\n"
                        final_text += progress
                        yield {"type": "text_delta", "text": progress}

                outcomes = self.workflow.results
                for task_id, outcome in outcomes.items():
                    if task_id not in prior:
                        usage += outcome.usage
                        total_turns += outcome.turns
                _apply_outcomes(plan, outcomes)
                status = self.workflow.status
                if status != "replan_required":
                    break
                status = "failed"
                if revised or not self._reports:
                    note = (
                        "本次运行已重规划一次，仍有任务前提无法满足，暂停执行。\n\n"
                        if revised
                        else "未收到可核对的任务观察，暂停执行。\n\n"
                    )
                    final_text += note
                    yield {"type": "text_delta", "text": note}
                    break

                yield {"type": "plan_status", "phase": "replanning"}
                candidate = None
                reason = ""
                try:
                    async for event in self.planner.stream_revision(
                        plan, reports=self._reports, evidence=self._discovery
                    ):
                        if event["type"] == "plan_revision":
                            candidate, reason = event["plan"], event["reason"]
                            continue
                        if event["type"] == "usage":
                            usage += Usage.from_mapping(event.get("usage"))
                        yield event
                    if candidate is not None:
                        prior = _revision_priors(plan, candidate, outcomes, self._reports)
                except Exception as exc:  # noqa: BLE001 - 保留已经执行的任务结果。
                    candidate, reason = None, str(exc)
                if candidate is None:
                    note = "未继续重规划" + (f"：{reason}" if reason else "") + "。\n\n"
                    final_text += note
                    yield {"type": "text_delta", "text": note}
                    break

                revised = True
                plan = candidate
                self._revision_notes = dict(self._reports)
                _apply_outcomes(plan, prior)
                note = f"已调整受影响的任务，保留 {len(prior)} 项既有结果：{reason}\n\n"
                final_text += note
                yield {"type": "text_delta", "text": note}

            if status == "completed":
                plan.mark_completed()
            elif status == "cancelled":
                plan.status = PlanStatus.CANCELLED
            else:
                plan.mark_failed()
            result_text = (
                "已取消：未执行未经确认的计划。\n"
                if status == "cancelled"
                else _build_plan_result(plan)
            )
            final_text += result_text
            yield {"type": "text_delta", "text": result_text}
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "error": exc}
            return
        finally:
            if snapshot:
                with suppress(Exception):
                    snapshot.create("post-turn")

        done: dict[str, Any] = {
            "type": "done",
            "status": status,
            "total_turns": total_turns,
            "total_tokens": usage.total_tokens,
            "usage": usage.to_dict(),
            "messages": self.history,
            "task_results": {
                key: asdict(value)
                for key, value in (self.workflow.results if self.workflow else {}).items()
            },
        }
        costs = calculate_costs(self.llm_client, usage)
        if costs:
            done["cost"] = costs
        yield done

    async def _run_task(
        self,
        plan: ExecutionPlan,
        task: Task,
        dependencies: dict[str, TaskOutcome],
        read_only: bool,
        feedback: str,
        emit: Callable[[dict[str, Any]], None],
    ) -> WorkerResult:
        """根据依赖结果组装任务上下文，为每个任务创建独立工作 Agent 并返回统一结果。"""
        registry = self.tool_registry.read_only() if read_only else self.tool_registry
        context = _task_context(plan, task, dependencies)
        for dep_id in dependencies:
            if handoff := self._handoffs.get(dep_id):
                context += f"\n\n[{dep_id}] 工具实际返回（资料）：\n{handoff}"
        if self._discovery:
            context += f"\n\n规划阶段观察（有疑问时核对最新源码）：\n{self._discovery}"
        if previous_observation := self._revision_notes.get(task.id):
            context += (
                "\n\n本任务已因以下观察修订，请从当前工作区继续。"
                "先核对已产生的修改，不重复已完成且可能有副作用的操作。\n"
                + previous_observation
            )
        if feedback:
            context += f"\n\n本次重试反馈：\n{feedback}"
        observations: list[_Observation] = []
        requested_paths: dict[str, str] = {}
        if self.staged_planning:
            registry = self._work_registry(registry, task, observations)

        def forward(event: dict[str, Any]) -> None:
            if (
                self.staged_planning
                and event["type"] == "tool_call"
                and event.get("name") == "read_file"
                and isinstance(path := event.get("input", {}).get("path"), str)
            ):
                requested_paths[event["tool_call_id"]] = path[:180]
            if self.staged_planning and event["type"] == "tool_result":
                name = event.get("name")
                text = str(event.get("result") or "")
                path = requested_paths.pop(event.get("tool_call_id"), "")
                if name in _HANDOFF_TOOLS and text and (
                    not event.get("is_error")
                    or (
                        name in {"bash", "execute_command"}
                        and text.startswith("Command failed: shell=")
                        and "; exit_code=" in text.partition("\n")[0]
                    )
                ):
                    observations.append(_Observation(name, text, event.get("evidence"), path))
            emit(event)

        agent = SubAgent(
            name=f"plan-{task.id}",
            role=AgentRole.WORKER,
            llm_client=self.llm_client,
            tool_registry=registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            event_callback=forward,
            system_prompt=self._task_system_prompt(plan, task, registry, read_only),
            max_turns=self.max_task_turns,
        )
        result = await agent.execute(AgentMessage.task("plan", context))
        if self.staged_planning and result.status == "completed" and observations:
            latest: dict[str, _Observation] = {}
            for observation in observations:
                latest.pop(observation.tool, None)
                latest[observation.tool] = observation
            self._handoffs[task.id] = "\n".join(
                f"{observation.tool}: {_observation_excerpt(observation.text, observation.artifact)}"
                for observation in list(latest.values())[-2:]
            )
        if result.status == "replan_required" and task.id in self._reports:
            return WorkerResult(
                result.content,
                result.usage,
                result.turns,
                "replan_required",
                self._reports[task.id],
            )
        error = result.content if result.type == AgentMessageType.ERROR else ""
        return WorkerResult(
            result.content,
            result.usage,
            result.turns,
            result.status,
            error or (result.status if result.status != "completed" else ""),
        )

    def _work_registry(
        self,
        base: ToolRegistry,
        task: Task,
        observations: list[_Observation],
    ) -> ToolRegistry:
        registry = ToolRegistry()
        for name in base.list_names():
            if tool := base.get(name):
                registry.register(tool)

        async def report(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return ToolResult("Describe which task assumption is contradicted.", is_error=True)
            selected = _select_observation(observations, payload)
            if isinstance(selected, ToolResult):
                return selected
            observation_id, observation = selected
            report_text = (
                f"{reason.strip()[:650]}\n"
                f"{observation.tool}: {_observation_excerpt(observation.text, observation.artifact)}"
            )
            if observation.path:
                report_text += f"\n已读取文件：{observation.path}"
            writes = {item.tool: item for item in observations if item.tool in {"write_file", "edit_file"}}
            for name, item in writes.items():
                report_text += f"\n已执行的编辑 {name}: {_observation_excerpt(item.text, item.artifact)}"
            self._reports[task.id] = report_text
            if run := current_run():
                run.record(
                    "plan.issue",
                    task_id=task.id,
                    source=observation.tool,
                    observation_id=observation_id,
                    evidence=observation.artifact,
                )
            raise RunStopped("replan_required", "Sent the observed problem to the planner.")

        registry.register(
            Tool(
                name="report_plan_issue",
                description=(
                    "Report an observed contradiction that requires changing a planned task, "
                    "its inputs, deliverable or dependencies. Do not use for ordinary errors "
                    "you can correct within the current task, missing permission or a timeout. "
                    "Give the observed tool name, or select an observation ID from a prior response."
                ),
                parameters=object_schema(
                    {
                        "reason": {"type": "string"},
                        "observed_tool": {"type": "string"},
                        "observation_id": {
                            "type": "integer",
                            "description": "Number of the exact observation listed by this tool.",
                        },
                    },
                    ["reason"],
                ),
                required_keys=["reason"],
                handler=report,
                is_concurrency_safe=False,
            )
        )
        return registry

    def _task_system_prompt(
        self, plan: ExecutionPlan, task: Task, registry: ToolRegistry, read_only: bool
    ) -> str:
        """按任务和只读边界组装稳定执行提示，工具注册表同时落实权限约束。"""
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_static()
        task_type = getattr(task.type, "value", task.type)
        boundary = "本任务仅允许只读调查，不得尝试写入文件或执行命令。" if read_only else ""
        escalation = (
            "\n原任务仍可完成时，在当前任务内修正。"
            "工具结果证明计划的前提不成立，并且会改变任务或依赖时，调用 report_plan_issue。"
            if self.staged_planning
            else ""
        )
        return (
            base
            + "\n\n你正在执行 Plan-and-Execute DAG 中的一个任务。\n"
            + f"任务 id：{task.id}\n任务类型：{task_type}\n"
            + "请具体完成任务，并在需要时使用工具。"
            + boundary
            + escalation
            + _task_language_instruction(plan.goal)
        )


def _observation_excerpt(text: str, artifact: dict | None) -> str:
    preview = text if len(text) <= 750 else text[:750] + "\n[预览已截断]"
    if artifact and artifact.get("run_id") and artifact.get("id"):
        preview += f"\n证据 {artifact['run_id']}:{artifact['id']}"
    return preview


def _select_observation(
    observations: list[_Observation], payload: dict[str, Any]
) -> tuple[int, _Observation] | ToolResult:
    available = {
        number: item
        for number, item in enumerate(observations, 1)
        if item.tool in _REPORT_EVIDENCE_TOOLS
    }
    if not available:
        return ToolResult(
            "First use a source or command tool to observe the issue, then cite its name.",
            is_error=True,
        )
    tool = payload.get("observed_tool")
    number = payload.get("observation_id")
    if type(number) is int and number in available:
        item = available[number]
        if tool is None or item.tool == tool:
            return number, item
    candidates = [(key, item) for key, item in available.items() if item.tool == tool]
    if number is None and len(candidates) == 1:
        return candidates[0]
    choices = "\n".join(
        f"- {key} ({item.tool}): {item.path + ': ' if item.path else ''}{' '.join(item.text.split())[:120]}"
        for key, item in (candidates or list(available.items()))[-6:]
    )
    return ToolResult(
        "Choose the observation_id of the tool output that contradicts your task assumption:\n"
        + choices,
        is_error=True,
    )


def _describe_plan(plan: ExecutionPlan) -> str:
    lines = [plan.summarize().rstrip(), ""]
    for task_id in plan.execution_order():
        task = plan.tasks[task_id]
        after = f"（等待：{', '.join(task.dependencies)}）" if task.dependencies else ""
        lines.append(f"- [{task.id}] {task.description}{after}")
        if task.expected_output:
            lines.append(f"  交付：{task.expected_output}")
        if task.verification:
            lines.append(f"  检查：{task.verification}")
    return "\n".join(lines) + "\n\n"


def _task_definition(task: Task) -> tuple:
    return (
        task.description,
        task.type,
        frozenset(task.dependencies),
        task.expected_output,
        task.verification,
    )


def _revision_priors(
    original: ExecutionPlan,
    revision: ExecutionPlan,
    outcomes: dict[str, TaskOutcome],
    reports: dict[str, str],
) -> dict[str, TaskOutcome]:
    """局部替换受阻任务及下游，并保留其他已执行的结果与副作用。"""
    if original.goal != revision.goal or original.tasks.keys() - revision.tasks.keys():
        raise ValueError("重规划必须保留用户目标和原任务编号")
    affected = set(reports)
    for task_id in original.execution_order():
        if affected.intersection(original.tasks[task_id].dependencies):
            affected.add(task_id)
    changed = {
        task_id
        for task_id, task in original.tasks.items()
        if _task_definition(task) != _task_definition(revision.tasks[task_id])
    }
    if not reports.keys() <= changed:
        raise ValueError("重规划未调整报告问题的任务")
    if changed - affected:
        raise ValueError("重规划改动了与问题无关的任务")
    additions = revision.tasks.keys() - original.tasks.keys()
    if unrelated := _unconnected_additions(revision, affected, additions):
        raise ValueError(f"重规划添加了与受影响任务无关的新任务：{', '.join(sorted(unrelated))}")
    if any(outcome.status == "completed" for task_id, outcome in outcomes.items() if task_id in affected):
        raise ValueError("仍需核对受影响的已执行任务，无法自动重跑")
    return {task_id: outcome for task_id, outcome in outcomes.items() if task_id not in affected}


def _unconnected_additions(
    revision: ExecutionPlan, affected: set[str], additions: set[str]
) -> set[str]:
    if not additions:
        return set()
    # 新任务可以是修订任务的前置或下游，连接不能借道无关的旧任务。
    neighbors: dict[str, list[str]] = {task_id: [] for task_id in affected | additions}
    for task_id in neighbors:
        for dependency in revision.tasks[task_id].dependencies:
            if dependency in neighbors:
                neighbors[task_id].append(dependency)
                neighbors[dependency].append(task_id)
    reachable = set(affected)
    queue = list(affected)
    while queue:
        for neighbor in neighbors[queue.pop()]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return additions - reachable


def _apply_outcomes(plan: ExecutionPlan, outcomes: dict[str, TaskOutcome]) -> None:
    """把工作流终态同步回计划模型，保留失败、阻塞和预算耗尽等结果。"""
    for task in plan.all_tasks():
        outcome = outcomes.get(task.id)
        if outcome is None:
            continue
        if outcome.status == "completed":
            task.mark_completed(outcome.text)
        elif outcome.status == "blocked":
            task.mark_skipped()
        else:
            task.mark_failed(outcome.error or outcome.status)
        task.result = outcome.text
        task.error = outcome.error


def _task_context(plan: ExecutionPlan, task: Task, dependencies: dict[str, TaskOutcome]) -> str:
    lines = [
        f"目标：{plan.goal}",
        f"当前任务 [{task.id}]：{task.description}",
    ]
    if task.expected_output:
        lines.append(f"预期交付：{task.expected_output}")
    if task.verification:
        lines.append(f"完成前检查：{task.verification}")
    lines.extend(["", "已完成的依赖任务结果："])
    for dep_id in task.dependencies:
        outcome = dependencies.get(dep_id)
        if outcome and outcome.status == "completed":
            dep = plan.get_task(dep_id)
            description = dep.description if dep else dep_id
            lines.append(
                f"- [{dep_id}] {description}: {evidence_context(outcome.text, outcome.evidence)}"
            )
    return "\n".join(lines)


def _build_plan_result(plan: ExecutionPlan) -> str:
    status_labels = {
        TaskStatus.PENDING: "待执行",
        TaskStatus.RUNNING: "执行中",
        TaskStatus.COMPLETED: "已完成",
        TaskStatus.FAILED: "失败",
        TaskStatus.SKIPPED: "已跳过",
    }
    header = "计划执行完成。" if plan.is_all_completed() else "计划未完成。"
    lines = [header, "", "任务摘要："]
    intermediate_ids = {dep for task in plan.all_tasks() for dep in task.dependencies}
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {status_labels[task.status]}：{task.description}")
        content = task.result or task.error
        if content:
            result = _preview(content) if task.id in intermediate_ids else content
            lines.append(f"  结果：{result}")
    return "\n".join(lines) + "\n"


def _task_language_instruction(goal: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in goal):
        return "\n用户目标包含中文；所有进度说明、分析和最终结果都必须使用中文。"
    return "\nUse the same language as the user's goal for all progress and results."


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
