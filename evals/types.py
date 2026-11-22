"""评测任务定义、遥测数据与报告结构。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

SuiteName = Literal["A", "B", "C", "D"]
EvalMode = Literal["minimal_react", "paicli_react", "paicli_full", "paicli_team"]


@dataclass(frozen=True, slots=True)
class EvalTask:
    """单个自包含本地 Git 仓库评测任务。"""

    id: str
    suite: SuiteName
    title: str
    description: str
    prompt: str
    files: dict[str, str]
    protected_paths: tuple[str, ...] = ()
    public_test_targets: tuple[str, ...] = ()
    hidden_tests: dict[str, str] = field(default_factory=dict)
    expected_initial_f2p: tuple[str, ...] = ()
    expected_initial_p2p: tuple[str, ...] = ()
    read_only: bool = False
    expected_routes: tuple[str, ...] = ()
    forbidden_routes: tuple[str, ...] = ()
    knowledge_paths: tuple[str, ...] = ()
    custom_grader: Callable[[str, dict[str, str]], tuple[bool, str, dict[str, Any]]] | None = None
    max_turns: int = 24
    timeout_sec: float = 240.0


@dataclass(slots=True)
class GradeResult:
    """确定性评分结果（公开测试、隐藏测试、防篡改、自定义检查）。"""

    passed: bool
    public_passed: bool
    hidden_passed: bool
    tampered: bool
    p2p_regressions: int = 0
    public_summary: str = ""
    hidden_summary: str = ""
    tamper_details: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunTelemetry:
    """从执行过程与 EvidenceRun events.jsonl 提取的遥测指标。"""

    wall_clock_sec: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    route_taken: str = "react"
    routes_seen: list[str] = field(default_factory=list)
    misrouted: bool = False
    first_try_integration_pass: bool | None = None
    worker_count: int = 0
    run_id: str | None = None
    evidence_dir: str | None = None
    error: str = ""


@dataclass(slots=True)
class TaskRunRecord:
    """单个任务在某个评测模式下的完整执行记录。"""

    task_id: str
    suite: SuiteName
    title: str
    mode: EvalMode
    grade: GradeResult
    telemetry: RunTelemetry
    response_excerpt: str = ""
    changed_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalSuiteReport:
    """整轮评测汇总结果，用于持久化 JSON 与多组评测 A/B 对比。"""

    label: str
    created_at: str
    git_commit: str
    provider: str
    model: str
    modes: list[str]
    suites: list[str]
    records: list[TaskRunRecord] = field(default_factory=list)
