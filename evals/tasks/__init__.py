"""评测任务注册表。"""

from __future__ import annotations

from evals.tasks.suite_a_direct import SUITE_A_TASKS
from evals.tasks.suite_b_impact import SUITE_B_TASKS
from evals.tasks.suite_c_team import SUITE_C_TASKS
from evals.tasks.suite_d_debug import SUITE_D_TASKS
from evals.types import EvalTask

ALL_TASKS: tuple[EvalTask, ...] = (
    *SUITE_A_TASKS,
    *SUITE_B_TASKS,
    *SUITE_C_TASKS,
    *SUITE_D_TASKS,
)

TASKS_BY_ID: dict[str, EvalTask] = {task.id: task for task in ALL_TASKS}


def select_tasks(
    *,
    suites: list[str] | None = None,
    task_ids: list[str] | None = None,
) -> list[EvalTask]:
    if task_ids:
        unknown = [tid for tid in task_ids if tid not in TASKS_BY_ID]
        if unknown:
            raise ValueError(f"Unknown eval task IDs: {', '.join(unknown)}")
        return [TASKS_BY_ID[tid] for tid in task_ids]
    if suites:
        wanted = {s.upper() for s in suites}
        return [task for task in ALL_TASKS if task.suite in wanted]
    return list(ALL_TASKS)
