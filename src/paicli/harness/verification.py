"""固定 pytest 验收契约，通过独立进程和真实结果判定修复是否交付。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import signal
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

from paicli.harness.evidence import EvidenceRun, load_run

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".paicli", ".pytest_cache"}
_CONFIG_NAMES = {"pytest.ini", ".pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
_LOCK_NAMES = {"uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock", "requirements.txt"}
_ENV_POLICY = {
    "clear": ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH"],
    "bytecode": "fresh external cache prefix; no bytecode writes",
    "plugins": "installed plugin autoload policy is preserved and fingerprinted",
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path):
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in _SKIP_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Project file points outside the workspace: {path}")
            yield path


def _targets(root: Path, values: list[str]) -> list[str]:
    """规范化项目内的 pytest 目标，拒绝越界路径及不允许的目标参数。"""
    if not values:
        raise ValueError("Provide at least one pytest test file or directory")
    normalized = []
    for value in values:
        if not value or value.startswith("-") or "\x00" in value:
            raise ValueError("Targets must be paths or pytest node selections, not options")
        file_name, separator, node = value.partition("::")
        path = (root / file_name).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f"Target must exist inside the workspace: {file_name}")
        if path.is_file() and path.suffix != ".py":
            raise ValueError("Only Python pytest files and directories are supported")
        if path.is_dir() and separator:
            raise ValueError("A pytest node selection must identify a Python file")
        relative = path.relative_to(root).as_posix() or "."
        normalized.append(relative + (f"::{node}" if separator else ""))
    return list(dict.fromkeys(normalized))


def _fingerprints(
    root: Path, targets: list[str], collected_sources: dict | None = None
) -> tuple[dict, dict]:
    """记录测试、配置等受保护文件与应用源码指纹，供前后验收核对。"""
    selected = [(root / target.split("::", 1)[0]).resolve() for target in targets]
    protected, code = {}, {}
    for path in _files(root):
        name = path.name
        relative = path.relative_to(root).as_posix()
        config = name in _CONFIG_NAMES or name == "conftest.py"
        dependency = name in _LOCK_NAMES or (
            name.startswith("requirements") and path.suffix == ".txt"
        )
        # 整项目测试目标只保护测试模块，应用源码仍需允许修复。
        test = (
            relative in (collected_sources or {})
            or (
                path.suffix == ".py"
                and (
                    name.startswith("test_")
                    or name.endswith("_test.py")
                    or "tests" in path.relative_to(root).parts
                )
            )
            or any(
                path == target
                or (
                    target.is_dir()
                    and target != root
                    and path.is_relative_to(target)
                    and path.suffix == ".py"
                )
                or (
                    target == root
                    and path.suffix == ".py"
                    and (
                        name.startswith("test_")
                        or name.endswith("_test.py")
                        or "tests" in path.relative_to(root).parts
                    )
                )
                for target in selected
            )
        )
        if config or dependency or test or path.suffix == ".py":
            digest = _digest(path)
            code[relative] = digest
            if config or dependency or test:
                protected[relative] = digest
    return protected, code


def _environment(run: EvidenceRun, label: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in _ENV_POLICY["clear"]:
        env.pop(key, None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(run.directory / f"{label}-bytecode")
    return env


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        command = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32/taskkill.exe"
        killer = await asyncio.create_subprocess_exec(
            str(command),
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(killer.wait(), 5)
        except TimeoutError:
            killer.kill()
            await killer.wait()
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await process.wait()


async def _interpreter(run: EvidenceRun, executable: str, env: dict) -> dict:
    script = (
        "import sys,json,hashlib,importlib.metadata as m;"
        "packages=sorted((d.metadata.get('Name',''),d.version) for d in m.distributions());"
        "print(json.dumps({'version':sys.version,'prefix':sys.prefix,"
        "'packages_sha256':hashlib.sha256(json.dumps(packages).encode()).hexdigest()}))"
    )
    process = await asyncio.create_subprocess_exec(
        executable,
        "-c",
        script,
        cwd=run.directory,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(process.communicate(), 15)
    except (TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ValueError("Could not inspect the selected Python interpreter")
    identity = json.loads(stdout)
    identity["executable_sha256"] = _digest(Path(executable))
    stable_env = {
        key: value
        for key, value in env.items()
        if key not in {"PYTHONPYCACHEPREFIX", "PWD", "OLDPWD", "_"}
    }
    identity["environment_sha256"] = hashlib.sha256(
        json.dumps(stable_env, sort_keys=True).encode("utf-8")
    ).hexdigest()
    identity["plugin_autoload_policy"] = os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "")
    return identity


def _junit(path: Path) -> dict:
    if not path.exists() or path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("JUnit report is missing or exceeds 20 MB")
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Invalid JUnit report root")
    cases = []
    for case in root.iter("testcase"):
        identity = json.dumps([case.get("classname", ""), case.get("name", "")], ensure_ascii=False)
        status = "passed"
        for tag, value in [("failure", "failed"), ("error", "error"), ("skipped", "skipped")]:
            if case.find(tag) is not None:
                status = value
                break
        cases.append(
            {"testcase_id": identity, "status": status, "source_file": case.get("file", "")}
        )
    counts = Counter(case["status"] for case in cases)
    return {
        "cases": cases,
        "counts": {key: counts[key] for key in ["passed", "failed", "error", "skipped"]},
        "total": len(cases),
    }


async def _execute(run: EvidenceRun, contract: dict, label: str) -> dict:
    """在固定解释器和环境策略下运行 pytest，保存输出与 JUnit，并检查验收条件完整性。"""
    root = Path(contract["cwd"])
    before, code_before = _fingerprints(
        root, contract["targets"], contract.get("collected_sources")
    )
    env = _environment(run, label)
    interpreter = await _interpreter(run, contract["python"], env)
    if before != contract["protected"] or interpreter != contract["interpreter"]:
        return {
            "status": "invalidated",
            "reason": "Protected tests/config/dependencies or interpreter environment changed",
        }
    stdout_path = run.directory / f"{label}.stdout.txt"
    stderr_path = run.directory / f"{label}.stderr.txt"
    junit_path = run.directory / f"{label}.junit.xml"
    argv = [
        contract["python"],
        "-m",
        "pytest",
        "--color=no",
        f"--rootdir={root}",
        "-o",
        "junit_family=xunit1",
        f"--junitxml={junit_path}",
        *contract["targets"],
    ]
    run.record("verification.started", argv=argv, cwd=str(root), timeout=contract["timeout"])
    start = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
        )
        try:
            await asyncio.wait_for(process.wait(), contract["timeout"])
        except TimeoutError:
            timed_out = True
            await _stop(process)
        except asyncio.CancelledError:
            await _stop(process)
            raise
    artifacts = {
        "stdout": run.adopt_file("verification_stdout", stdout_path),
        "stderr": run.adopt_file("verification_stderr", stderr_path),
    }
    if junit_path.exists():
        artifacts["junit"] = run.adopt_file("verification_junit", junit_path)
    after, code_after = _fingerprints(root, contract["targets"], contract.get("collected_sources"))
    report: dict[str, Any] = {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - start, 3),
        "artifacts": artifacts,
        "code_snapshot": run.save_text(
            "verification_code_snapshot",
            json.dumps(code_before, ensure_ascii=False, sort_keys=True, indent=2),
        ),
    }
    if after != before or code_after != code_before:
        return {
            **report,
            "status": "invalidated",
            "reason": "Project code or acceptance files changed while pytest was running",
        }
    if timed_out:
        return {**report, "status": "unverifiable", "reason": "pytest timed out"}
    try:
        report.update(_junit(junit_path))
        sources = {}
        for case in report["cases"]:
            source = (root / case["source_file"]).resolve()
            if not case["source_file"] or not source.is_relative_to(root):
                raise ValueError("Collected test source cannot be bound to this workspace")
            relative = source.relative_to(root).as_posix()
            if source.suffix != ".py" or relative not in code_before:
                raise ValueError("Collected test source is missing from the execution fingerprint")
            sources[relative] = code_before[relative]
        report["collected_sources"] = sources
    except (OSError, ET.ParseError, ValueError) as exc:
        return {**report, "status": "unverifiable", "reason": str(exc)}
    return report


def _publish(run: EvidenceRun, report: dict) -> dict:
    report = {"run_id": run.run_id, **report}
    reference = run.save_text(
        "verification_report", json.dumps(report, ensure_ascii=False, indent=2)
    )
    report["report_path"] = str(run.directory / reference["path"])
    report["report_artifact"] = reference["id"]
    run.record("verification.finished", status=report["status"], report_artifact=reference["id"])
    run.finish(report["status"])
    return report


async def create_baseline(
    cwd: Path,
    targets: list[str],
    python_executable: str | None = None,
    timeout: float = 60,
    *,
    required_failures: list[str] | None = None,
) -> dict:
    """固定测试目标、解释器和受保护文件，实际复现断言失败并保存验收契约。

    收集错误、空测试或跳过结果不能建立有效失败基线；可指定本次必须修复的失败项。
    """
    root = Path(cwd).resolve()
    if not root.is_dir() or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError(
            "Use an existing project directory and a timeout between 0 and 3600 seconds"
        )
    selected = _targets(root, targets)
    candidate = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = Path(
        python_executable or (str(candidate) if candidate.exists() else sys.executable)
    ).absolute()
    if not python.is_file():
        raise ValueError("The selected Python interpreter does not exist")
    run = EvidenceRun.create(str(root), "verification")
    protected, _code = _fingerprints(root, selected)
    contract = {
        "version": 1,
        "cwd": str(root),
        "python": str(python),
        "targets": selected,
        "timeout": timeout,
        "protected": protected,
        "environment_policy": _ENV_POLICY,
        "interpreter": await _interpreter(run, str(python), _environment(run, "probe")),
    }
    report = await _execute(run, contract, "baseline")
    if "status" not in report:
        counts = report["counts"]
        if report["exit_code"] == 1 and counts["failed"] > 0 and counts["error"] == 0:
            report.update(status="reproduced", reason="Existing test failure reproduced")
        elif report["exit_code"] == 0 and counts["passed"] > 0:
            report.update(
                status="not_reproduced", reason="Selected tests already pass; no failure baseline"
            )
        else:
            report.update(
                status="unverifiable",
                reason=(
                    "Expected an assertion failure, not empty/skipped tests, "
                    "collection, setup, usage or execution errors"
                ),
            )
    if required_failures is not None:
        if not required_failures:
            raise ValueError("Required failure identities cannot be empty")
        failed = {
            case["testcase_id"] for case in report.get("cases", []) if case["status"] == "failed"
        }
        contract["required_failures"] = list(dict.fromkeys(required_failures))
        if report["status"] == "reproduced" and not set(required_failures) <= failed:
            report.update(status="invalidated", reason="Required failure identities changed")
    contract["baseline"] = report
    contract["collected_sources"] = report.get("collected_sources", {})
    contract["protected"].update(contract["collected_sources"])
    path = run.directory / "verification-contract.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    run.adopt_file("verification_contract", path)
    return _publish(run, {"phase": "baseline", "contract_path": str(path), **report})


async def verify_run(run_id: str) -> dict:
    """按原契约独立复测，核对测试集合和源码保护，再判定失败转通过及原通过项保持。

    测试被改写、移除或跳过会使验收失效；明确限定修复目标时，其他原有失败单独报告。
    """
    run = load_run(run_id)
    path = run.directory / "verification-contract.json"
    if not path.is_file():
        raise ValueError("This run does not contain a verification contract")
    references = [
        item
        for item in run.manifest["artifacts"].values()
        if item["path"] == path.name and item["kind"] == "verification_contract"
    ]
    if len(references) != 1 or _digest(path) != references[0]["sha256"]:
        raise ValueError("Verification contract integrity check failed")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("version") != 1:
        raise ValueError("Unsupported verification contract version")
    baseline = contract["baseline"]
    if baseline["status"] != "reproduced":
        return _publish(
            run,
            {
                "phase": "check",
                "status": "unverifiable",
                "reason": "The original run did not establish a reproducible failing baseline",
            },
        )
    report = await _execute(run, contract, f"check-{uuid.uuid4().hex[:12]}")
    if "status" not in report:
        observed = {case["testcase_id"]: case["status"] for case in report["cases"]}
        for label, status in [("fail_to_pass", "failed"), ("pass_to_pass", "passed")]:
            cases = [case["testcase_id"] for case in baseline["cases"] if case["status"] == status]
            if label == "fail_to_pass" and "required_failures" in contract:
                cases = contract["required_failures"]
            report[label] = {
                "total": len(cases),
                "passed": sum(observed.get(case) == "passed" for case in cases),
                "failed": [case for case in cases if observed.get(case) != "passed"],
            }
        old_ids = Counter(case["testcase_id"] for case in baseline["cases"])
        new_ids = Counter(case["testcase_id"] for case in report["cases"])
        old_skips = Counter(
            case["testcase_id"] for case in baseline["cases"] if case["status"] == "skipped"
        )
        new_skips = Counter(
            case["testcase_id"] for case in report["cases"] if case["status"] == "skipped"
        )
        if (
            old_ids != new_ids
            or new_skips != old_skips
            or report["collected_sources"] != contract["collected_sources"]
        ):
            report.update(
                status="invalidated", reason="Test collection or skip/xfail outcomes changed"
            )
        elif (
            report["exit_code"] in {0, 1}
            and report["total"] > 0
            and report["counts"]["error"] == 0
            and not report["fail_to_pass"]["failed"]
            and not report["pass_to_pass"]["failed"]
        ):
            remaining = [
                case
                for case in baseline["cases"]
                if case["status"] == "failed"
                and case["testcase_id"]
                not in contract.get(
                    "required_failures",
                    [c["testcase_id"] for c in baseline["cases"] if c["status"] == "failed"],
                )
                and observed.get(case["testcase_id"]) != "passed"
            ]
            report.update(
                status="accepted",
                existing_failures=remaining,
                reason="Required failures fixed and originally passing acceptance preserved",
            )
        else:
            report.update(
                status="rejected", reason="The original acceptance tests did not all pass"
            )
    return _publish(run, {"phase": "check", **report})
