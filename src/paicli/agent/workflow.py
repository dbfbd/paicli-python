"""Plan 与 Team 共用的 LangGraph 调度，以及单任务执行、审核和终态归并。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from graphlib import TopologicalSorter
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from paicli.harness.evidence import current_run, task_scope
from paicli.plan.models import Task, TaskType
from paicli.plan.validation import validate_plan_nodes
from paicli.types import Usage


@dataclass(slots=True)
class WorkerResult:
    """工作任务的返回值，包含文字、状态、轮数、用量和证据。"""

    text: str
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    status: str = "completed"
    error: str = ""
    evidence: dict | None = None


@dataclass(slots=True)
class ReviewResult:
    """审核回调的返回值，包含明确通过标记、修复反馈及审核开销。"""

    approved: bool
    feedback: str = ""
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    error: str = ""


@dataclass(slots=True)
class TaskOutcome:
    """任务最终交付状态，供依赖调度和外层汇总使用。"""

    task_id: str
    text: str = ""
    status: str = "completed"
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    attempts: int = 0
    error: str = ""
    evidence: dict | None = None


EventSink = Callable[[dict[str, Any]], None]
Worker = Callable[[Task, dict[str, TaskOutcome], bool, str, EventSink], Awaitable[WorkerResult]]
Reviewer = Callable[[Task, WorkerResult, EventSink], Awaitable[ReviewResult]]


def _merge_results(left: dict, right: dict) -> dict:
    # 各并行分支只发布自身终态，避免在归并时重复累计计数。
    """只合并各分支发布的任务终态，避免并行归并时重复累计用量。"""
    return {**left, **right}


class WorkflowState(TypedDict):
    results: Annotated[dict[str, TaskOutcome], _merge_results]


class AttemptState(TypedDict, total=False):
    task: Task
    dependencies: dict[str, TaskOutcome]
    read_only: bool
    attempt: int
    feedback: str
    result: WorkerResult
    review: ReviewResult
    usage: Usage
    turns: int
    outcome: TaskOutcome


class WorkflowRunner:
    """对只读任务实施有界并行，对写入或命令任务实施独占调度。

    此对象维护一次运行内的图状态；进程重启恢复由专用 Debug 路径另行提供。
    执行异常直接形成失败结果，避免在可能已产生写入副作用时自动重试。
    """

    def __init__(
        self,
        *,
        tasks: list[Task],
        worker: Worker,
        reviewer: Reviewer | None = None,
        max_concurrency: int = 2,
        max_retries: int = 0,
    ):
        validate_plan_nodes(
            {"tasks": [{"id": task.id, "dependencies": task.dependencies} for task in tasks]},
            primary_key="tasks",
            id_prefix="task",
        )
        by_id = {task.id: task for task in tasks}
        order = TopologicalSorter({task.id: task.dependencies for task in tasks}).static_order()
        self.tasks = [by_id[task_id] for task_id in order]
        self.worker = worker
        self.reviewer = reviewer
        self.max_concurrency = max(1, max_concurrency)
        self.max_retries = max(0, max_retries)
        self.results: dict[str, TaskOutcome] = {}
        self.task_graph = self._build_task_graph()

        graph = StateGraph(WorkflowState)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("execute_task", self._execute_task)
        graph.add_edge(START, "dispatch")
        graph.add_conditional_edges("dispatch", self._route, ["execute_task", END])
        graph.add_edge("execute_task", "dispatch")
        self.graph = graph.compile()

    @property
    def total_usage(self) -> Usage:
        return sum((result.usage for result in self.results.values()), Usage())

    @property
    def total_turns(self) -> int:
        return sum(result.turns for result in self.results.values())

    @property
    def status(self) -> str:
        statuses = {result.status for result in self.results.values()}
        for status in ("budget_exhausted", "failed", "blocked"):
            if status in statuses:
                return status
        return "completed" if len(self.results) == len(self.tasks) else "failed"

    async def run(self) -> AsyncIterator[dict[str, Any]]:
        """运行任务图并持续转发事件，只从所属图命名空间更新最终结果。"""
        self.results = {}
        root_namespace = None
        async for namespace, mode, data in self.graph.astream(
            {"results": {}},
            config={"recursion_limit": 2 * len(self.tasks) + 4},
            stream_mode=["custom", "values"],
            subgraphs=True,
        ):
            if mode == "custom":
                yield data
            elif mode == "values" and "results" in data:
                # 嵌套执行器继承上层命名空间，所属根空间不一定是空元组。
                if root_namespace is None:
                    root_namespace = namespace
                if namespace == root_namespace:
                    self.results = dict(data["results"])

    def _dispatch(self, state: WorkflowState) -> dict:
        """将依赖已失败的下游任务标为 blocked，防止继续执行失去前提的任务。"""
        results = dict(state["results"])
        blocked = {}
        for task in self.tasks:
            if task.id in results:
                continue
            failed_deps = [
                dep
                for dep in task.dependencies
                if dep in results and results[dep].status != "completed"
            ]
            if failed_deps:
                reason = f"dependency failed: {', '.join(failed_deps)}"
                outcome = TaskOutcome(task.id, status="blocked", error=reason)
                results[task.id] = blocked[task.id] = outcome
                self._sink(task)({"type": "workflow_task_finished", **asdict(outcome)})
        return {"results": blocked}

    def _route(self, state: WorkflowState) -> list[Send] | str:
        """选择依赖已齐备的任务；只读任务按并发上限分发，修改任务独占一个批次。"""
        results = state["results"]
        ready = [
            task
            for task in self.tasks
            if task.id not in results and all(dep in results for dep in task.dependencies)
        ]
        if not ready:
            return END
        # 同一轮运行中，写入或命令任务与其他读写任务互斥。
        if self._read_only(ready[0]):
            selected = [task for task in ready if self._read_only(task)][: self.max_concurrency]
        else:
            selected = ready[:1]
        return [
            Send(
                "execute_task",
                {
                    "task": task,
                    "dependencies": {dep: results[dep] for dep in task.dependencies},
                    "read_only": self._read_only(task),
                    "attempt": 0,
                    "feedback": "",
                    "usage": Usage(),
                    "turns": 0,
                },
            )
            for task in selected
        ]

    @staticmethod
    def _read_only(task: Task) -> bool:
        return task.type in {TaskType.FILE_READ, TaskType.ANALYSIS}

    async def _execute_task(self, state: AttemptState) -> dict:
        """执行单任务子图，并向外层只发布该任务的一份最终结果。"""
        final = await self.task_graph.ainvoke(
            state, config={"recursion_limit": 3 * (self.max_retries + 1) + 4}
        )
        outcome = final["outcome"]
        get_stream_writer()(
            {"type": "workflow_task_finished", "phase": "execution", **asdict(outcome)}
        )
        return {"results": {outcome.task_id: outcome}}

    def _build_task_graph(self):
        """构建 work、review、finish 子图，仅让审核拒绝触发有上限的局部重试。"""
        graph = StateGraph(AttemptState)
        graph.add_node("work", self._work)
        graph.add_node("review", self._review)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "work")
        graph.add_conditional_edges(
            "work", self._after_work, {"review": "review", "finish": "finish"}
        )
        graph.add_conditional_edges(
            "review", self._after_review, {"retry": "work", "finish": "finish"}
        )
        graph.add_edge("finish", END)
        return graph.compile()

    def _sink(self, task: Task) -> EventSink:
        writer = get_stream_writer()
        return lambda event: writer(
            {
                **event,
                "phase": "execution",
                "task_id": task.id,
                "task_description": task.description,
            }
        )

    async def _work(self, state: AttemptState) -> dict:
        """调用工作回调并累计本次开销，将执行异常转换为失败结果。"""
        task = state["task"]
        attempt = state["attempt"] + 1
        emit = self._sink(task)
        emit({"type": "plan_task_started", "attempt": attempt, "read_only": state["read_only"]})
        try:
            with task_scope(task.id, attempt):
                result = await self.worker(
                    task, state["dependencies"], state["read_only"], state["feedback"], emit
                )
                if run := current_run():
                    result.evidence = run.save_text("task-result", result.text)
        except Exception as exc:  # noqa: BLE001 - 将异常转换为明确任务终态。
            result = WorkerResult("", status="failed", error=str(exc))
        if result.status not in {"completed", "failed", "budget_exhausted"}:
            result = WorkerResult(
                result.text, result.usage, result.turns, "failed", "invalid worker status"
            )
        return {
            "attempt": attempt,
            "result": result,
            "review": ReviewResult(False),
            "usage": state["usage"] + result.usage,
            "turns": state["turns"] + result.turns,
        }

    def _after_work(self, state: AttemptState) -> str:
        if state["result"].status != "completed" or self.reviewer is None:
            return "finish"
        return "review"

    async def _review(self, state: AttemptState) -> dict:
        """调用审核回调并追加审核用量，将反馈交给下一次局部尝试。"""
        emit = self._sink(state["task"])
        try:
            assert self.reviewer is not None
            with task_scope(state["task"].id, state["attempt"]):
                review = await self.reviewer(state["task"], state["result"], emit)
        except Exception as exc:  # noqa: BLE001
            review = ReviewResult(False, f"review error: {exc}", error=str(exc))
        emit({"type": "workflow_review", "approved": review.approved, "attempt": state["attempt"]})
        return {
            "review": review,
            "feedback": review.feedback,
            "usage": state["usage"] + review.usage,
            "turns": state["turns"] + review.turns,
        }

    def _after_review(self, state: AttemptState) -> str:
        """审核通过、审核异常或达到重试上限时结束，其余未通过结果可返回工作节点。"""
        review = state["review"]
        if review.approved is True or review.error or state["attempt"] > self.max_retries:
            return "finish"
        return "retry"

    def _finish(self, state: AttemptState) -> dict:
        """把工作结果与审核决定合成终态，要求审核的任务必须明确获准才能完成。"""
        result = state["result"]
        status, error = result.status, result.error
        if status == "completed" and self.reviewer and state["review"].approved is not True:
            review = state["review"]
            status = "failed"
            error = (
                f"review failed: {review.error}"
                if review.error
                else f"review rejected: {review.feedback}"
            )
        return {
            "outcome": TaskOutcome(
                task_id=state["task"].id,
                text=result.text,
                status=status,
                usage=state["usage"],
                turns=state["turns"],
                attempts=state["attempt"],
                error=error,
                evidence=result.evidence,
            )
        }
