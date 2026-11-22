"""确定性验证 12 个评测任务的初始仓库状态、F2P/P2P 基线与防篡改打分器。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from evals.fixtures import task_workspace
from evals.grader import check_tampering, grade_task, list_changed_files, run_pytest_suite
from evals.tasks import ALL_TASKS, TASKS_BY_ID


@pytest.mark.parametrize("task", ALL_TASKS, ids=[t.id for t in ALL_TASKS])
def test_fixture_materializes_clean_and_fails_initially(task):
    # 确保所有夹具源码和隐藏测试均无语法错误。
    for rel, code in (*task.files.items(), *task.hidden_tests.items()):
        if rel.endswith(".py"):
            ast.parse(code, filename=f"{task.id}:{rel}")

    with task_workspace(task) as root:
        assert (root / ".git").exists()
        assert list_changed_files(root) == []
        tampered, violations = check_tampering(task, root)
        assert not tampered, violations

        if task.custom_grader is not None:
            grade = grade_task(task, root, "empty answer")
            assert not grade.passed
            return

        # 初始状态下公开测试不应全部通过。
        pub_ok, _summary, outcomes = run_pytest_suite(root, task.public_test_targets)
        assert not pub_ok, f"Fixture {task.id} already passes before any agent fix!"

        for case in task.expected_initial_p2p:
            status = outcomes.get(case) or outcomes.get(case.split("::")[-1])
            assert status == "passed", (
                f"Expected P2P case {case} to pass initially in {task.id}, got {status!r}"
            )


def test_grader_detects_protected_test_tampering():
    task = TASKS_BY_ID["d1_scheduler_regression_trap"]
    with task_workspace(task) as root:
        test_file = root / "tests/test_scheduler.py"
        test_file.write_text("def test_stub():\n    assert True\n", encoding="utf-8")
        grade = grade_task(task, root, "fixed")
        assert grade.tampered is True
        assert grade.passed is False
        assert any("tests/test_scheduler.py" in msg for msg in grade.tamper_details)


def test_grader_detects_read_only_violation_and_validates_c4_report():
    task = TASKS_BY_ID["c4_parallel_subsystem_audit"]
    good_report = (
        "1. auth: subsystems/auth/tokens.py verify_session_token 过期比较反向 (current < expires_at)\n"
        "2. cache: subsystems/cache/lru.py TtlLruCache.put popitem(last=True) 淘汰了最近使用项\n"
        "3. queue: subsystems/queue/worker_pool.py nack 未从 in_flight 移除且 > max_attempts 写反\n"
        "4. billing: subsystems/billing/ledger.py charge 用 current + cents 且未写入 charged_keys\n"
    )
    with task_workspace(task) as root:
        ok_grade = grade_task(task, root, good_report)
        assert ok_grade.passed is True
        assert ok_grade.tampered is False

        # 若只读任务违规写入任意文件，评分必须失败。
        (Path(root) / "subsystems/auth/tokens.py").write_text("# modified\n", encoding="utf-8")
        bad_grade = grade_task(task, root, good_report)
        assert bad_grade.tampered is True
        assert bad_grade.passed is False
