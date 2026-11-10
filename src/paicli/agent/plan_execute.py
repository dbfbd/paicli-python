"""Plan 模式入口：生成任务依赖图，并复用 LangGraph 工作流执行。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import asdict
from typing import Any

from paicli.agent.orchestrator import AgentMessage, AgentMessageType, AgentRole, SubAgent
from paicli.agent.workflow import TaskOutcome, WorkerResult, WorkflowRunner
from paicli.config import PaiCliConfig
from paicli.harness.evidence import evidence_context, recorded_run
from paicli.llm.base import LlmClient
from paicli.llm.pricing import calculate_costs
from paicli.plan import ExecutionPlan, Planner, Task, TaskStatus
from paicli.prompt import PromptAssembler
from paicli.snapshot import SnapshotService
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, Usage


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
        max_concurrent_tasks: int = 2,
        manage_snapshots: bool = True,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.max_concurrent_tasks = max_concurrent_tasks
        self.manage_snapshots = manage_snapshots
        self.history: list[Message] = []
        self.workflow: WorkflowRunner | None = None

    @recorded_run("plan")
    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """生成计划、执行任务图并收集终态；规划和工作任务的用量分别累计后汇总。"""
        snapshot = SnapshotService(self.cwd) if self.manage_snapshots else None
        if snapshot:
            with suppress(Exception):
                snapshot.create("pre-turn")
        planner_usage = Usage()
        final_text = ""
        try:
            yield {"type": "text_delta", "text": f"正在规划任务：{message}\n\n"}
            yield {"type": "plan_status", "phase": "planning"}
            plan: ExecutionPlan | None = None
            async for event in self.planner.stream_plan(message):
                if event.get("type") == "plan_created":
                    plan = event["plan"]
                    continue
                if event.get("type") == "usage":
                    planner_usage = planner_usage + Usage.from_mapping(event.get("usage") or {})
                yield event
            if plan is None:
                raise ValueError("planner did not produce an execution plan")
            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
            yield {"type": "text_delta", "text": "开始执行计划……\n\n"}
            yield {"type": "plan_status", "phase": "execution"}
            plan.mark_started()

            async def worker(task, dependencies, read_only, feedback, emit):
                return await self._run_task(plan, task, dependencies, read_only, feedback, emit)

            self.workflow = WorkflowRunner(
                tasks=plan.all_tasks(),
                worker=worker,
                max_concurrency=self.max_concurrent_tasks,
            )
            async for event in self.workflow.run():
                yield event
                if event.get("type") == "workflow_task_finished":
                    label = "已完成" if event["status"] == "completed" else "未完成"
                    content = event.get("error") or event.get("text") or event["status"]
                    progress = f"{label} [{event['task_id']}]：{_preview(str(content))}\n\n"
                    final_text += progress
                    yield {"type": "text_delta", "text": progress}

            _apply_outcomes(plan, self.workflow.results)
            if self.workflow.status == "completed":
                plan.mark_completed()
            else:
                plan.mark_failed()
            result_text = _build_plan_result(plan)
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

        total_usage = planner_usage + self.workflow.total_usage
        done: dict[str, Any] = {
            "type": "done",
            "status": self.workflow.status,
            "total_turns": self.workflow.total_turns,
            "total_tokens": total_usage.total_tokens,
            "usage": total_usage.to_dict(),
            "messages": self.history,
            "task_results": {key: asdict(value) for key, value in self.workflow.results.items()},
        }
        costs = calculate_costs(self.llm_client, total_usage)
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
        if feedback:
            context += f"\n\n本次重试反馈：\n{feedback}"
        agent = SubAgent(
            name=f"plan-{task.id}",
            role=AgentRole.WORKER,
            llm_client=self.llm_client,
            tool_registry=registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            event_callback=emit,
            system_prompt=self._task_system_prompt(plan, task, registry, read_only),
            max_turns=self.max_task_turns,
        )
        result = await agent.execute(AgentMessage.task("plan", context))
        error = result.content if result.type == AgentMessageType.ERROR else ""
        return WorkerResult(
            result.content,
            result.usage,
            result.turns,
            result.status,
            error or (result.status if result.status != "completed" else ""),
        )

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
        return (
            base
            + "\n\n你正在执行 Plan-and-Execute DAG 中的一个任务。\n"
            + f"任务 id：{task.id}\n任务类型：{task_type}\n"
            + "请具体完成任务，并在需要时使用工具。"
            + boundary
            + _task_language_instruction(plan.goal)
        )


def _apply_outcomes(plan: ExecutionPlan, outcomes: dict[str, TaskOutcome]) -> None:
    """把工作流终态同步回计划模型，保留失败、阻塞和预算耗尽等结果。"""
    for task in plan.all_tasks():
        outcome = outcomes[task.id]
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
        "",
        "已完成的依赖任务结果：",
    ]
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
