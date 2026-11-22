"""确定性评测打分器：公开测试、隐藏测试、防测试篡改校验与结构化报告校验。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.types import EvalTask, GradeResult


def _git_output(cwd: Path, *args: str) -> str:
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )
    return result.stdout.strip()


def list_changed_files(cwd: Path) -> list[str]:
    """列出相对初始提交所有修改或新增的文件（排除内部忽略目录）。"""
    tracked = _git_output(cwd, "diff", "--name-only", "HEAD").splitlines()
    untracked = _git_output(cwd, "ls-files", "--others", "--exclude-standard").splitlines()
    combined = sorted({p.replace("\\", "/") for p in (*tracked, *untracked) if p.strip()})
    return combined


def check_tampering(task: EvalTask, cwd: Path) -> tuple[bool, list[str]]:
    """校验受保护的测试/规范文件是否被篡改，或只读任务是否违规写入文件。"""
    changed = list_changed_files(cwd)
    if task.read_only:
        return (bool(changed), [f"read-only violation: {path}" for path in changed])

    violations: list[str] = []
    protected_norm = {p.replace("\\", "/").casefold() for p in task.protected_paths}
    for path in changed:
        if path.casefold() in protected_norm:
            violations.append(f"modified protected path: {path}")
    return (bool(violations), violations)


def run_pytest_suite(
    cwd: Path,
    targets: tuple[str, ...] | list[str],
    *,
    timeout: float = 60.0,
) -> tuple[bool, str, dict[str, str]]:
    """运行指定 pytest 目标并解析 JUnit 报告，返回 (是否全通过, 摘要, {case: status})。"""
    if not targets:
        return True, "no pytest targets", {}

    with TemporaryDirectory(prefix="paicli-grade-") as tmp:
        junit_path = Path(tmp) / "junit.xml"
        basetemp = Path(tmp) / "pytest-tmp"
        env = os.environ.copy()
        env.pop("PYTEST_ADDOPTS", None)
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(cwd)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--color=no",
            "-q",
            f"--rootdir={cwd}",
            f"--basetemp={basetemp}",
            "-o",
            "junit_family=xunit1",
            f"--junitxml={junit_path}",
            *targets,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {timeout:.0f}s", {}

        outcomes: dict[str, str] = {}
        if junit_path.exists():
            try:
                root = ET.parse(junit_path).getroot()
                for case in root.iter("testcase"):
                    name = case.get("name", "")
                    file_attr = (case.get("file") or "").replace("\\", "/")
                    case_id = f"{file_attr}::{name}" if file_attr else name
                    status = "passed"
                    for tag, label in (
                        ("failure", "failed"),
                        ("error", "error"),
                        ("skipped", "skipped"),
                    ):
                        if case.find(tag) is not None:
                            status = label
                            break
                    outcomes[case_id] = status
                    outcomes[name] = status
            except ET.ParseError:
                pass

        passed = proc.returncode == 0 and bool(outcomes)
        output = (proc.stdout + "\n" + proc.stderr).strip()
        summary = output[-800:] if len(output) > 800 else output
        return passed, summary, outcomes


def grade_task(task: EvalTask, cwd: Path, response_text: str) -> GradeResult:
    """对 Agent 执行完的工作区和回复做完整确定性评分。"""
    tampered, tamper_details = check_tampering(task, cwd)

    if task.custom_grader is not None:
        workspace_files = {
            path.relative_to(cwd).as_posix(): path.read_text(encoding="utf-8", errors="replace")
            for path in cwd.rglob("*.py")
            if ".git" not in path.parts and "__pycache__" not in path.parts
        }
        custom_ok, custom_summary, extra = task.custom_grader(response_text, workspace_files)
        return GradeResult(
            passed=custom_ok and not tampered,
            public_passed=custom_ok,
            hidden_passed=True,
            tampered=tampered,
            public_summary=custom_summary,
            tamper_details=tamper_details,
            extra=extra,
        )

    pub_ok, pub_summary, pub_outcomes = run_pytest_suite(cwd, task.public_test_targets)

    p2p_regressions = 0
    for case in task.expected_initial_p2p:
        status = pub_outcomes.get(case) or pub_outcomes.get(case.split("::")[-1])
        if status != "passed":
            p2p_regressions += 1

    hid_ok = True
    hid_summary = "no hidden tests"
    if task.hidden_tests:
        hidden_dir = cwd / ".eval_hidden"
        try:
            hidden_dir.mkdir(parents=True, exist_ok=True)
            hidden_targets: list[str] = []
            for rel_name, content in task.hidden_tests.items():
                target = hidden_dir / rel_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")
                hidden_targets.append(target.relative_to(cwd).as_posix())
            hid_ok, hid_summary, _ = run_pytest_suite(cwd, hidden_targets)
        finally:
            shutil.rmtree(hidden_dir, ignore_errors=True)

    return GradeResult(
        passed=pub_ok and hid_ok and not tampered and p2p_regressions == 0,
        public_passed=pub_ok,
        hidden_passed=hid_ok,
        tampered=tampered,
        p2p_regressions=p2p_regressions,
        public_summary=pub_summary,
        hidden_summary=hid_summary,
        tamper_details=tamper_details,
    )
