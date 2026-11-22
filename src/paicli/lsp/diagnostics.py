"""代码写后即时静态诊断：语法检查、Diff-Scoped Ruff 诊断与 AST 公开接口变动提示。"""

from __future__ import annotations

import ast
import json
import subprocess
from collections import Counter
from pathlib import Path
from traceback import format_exception_only

from paicli.harness.code_map import static_callers_for_file
from paicli.harness.project_env import resolve_project_executable

_RUFF_SELECT = "E9,F63,F7,F82,F401"


def diagnose_file(path: str | Path) -> list[str]:
    """保留兼容入口：对 Python 文件做即时字节码编译语法检查。"""
    file_path = Path(path)
    if file_path.suffix != ".py":
        return []
    try:
        compile(file_path.read_bytes(), str(file_path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return ["".join(format_exception_only(type(exc), exc)).rstrip()]
    return []


def _ruff_diagnostics(
    ruff_bin: str,
    cwd: Path,
    file_path: Path,
    *,
    source_override: str | None = None,
) -> list[tuple[str, str, int]]:
    """调用 ruff 检查严重代码错误，返回 [(code, message, line), ...]。"""
    cmd = [
        ruff_bin,
        "check",
        "--no-cache",
        "--select",
        _RUFF_SELECT,
        "--output-format",
        "json",
    ]
    if source_override is not None:
        cmd.extend(["--stdin-filename", str(file_path), "-"])
    else:
        cmd.append(str(file_path))

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=source_override,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    raw = (proc.stdout or "").strip()
    if not raw or not raw.startswith("["):
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    findings: list[tuple[str, str, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        msg = str(item.get("message") or "")
        loc = item.get("location") or {}
        line = int(loc.get("row") or 0)
        if code and msg:
            findings.append((code, msg, line))
    return findings


def _public_signatures(source: str) -> dict[str, str]:
    """提取模块顶层公开函数/类的接口签名，用于识别破坏性公共契约变更。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    sigs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            sigs[node.name] = ast.dump(node.args)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = {
                child.name: ast.dump(child.args)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not child.name.startswith("_") or child.name == "__init__")
            }
            sigs[node.name] = json.dumps(methods, sort_keys=True)
    return sigs


def diagnose_python_edit(
    cwd: str | Path,
    path: str | Path,
    *,
    before_text: str | None = None,
) -> list[str]:
    """写后综合静态反馈：语法错 + Diff-Scoped Ruff 新增严重告警 + 公共签名变动调用方提示。"""
    file_path = Path(path).resolve()
    if file_path.suffix != ".py" or not file_path.is_file():
        return []

    # 1. 即时语法检查（~1ms）
    syntax_errors = diagnose_file(file_path)
    if syntax_errors:
        return syntax_errors

    notes: list[str] = []
    root = Path(cwd).resolve()

    # 2. Diff-Scoped Ruff 检查（仅报本次编辑新引入的严重问题，不刷屏历史遗留告警）
    ruff_bin = resolve_project_executable("ruff", root)
    if ruff_bin:
        after_issues = _ruff_diagnostics(ruff_bin, root, file_path)
        if after_issues:
            if before_text is not None:
                before_issues = _ruff_diagnostics(
                    ruff_bin, root, file_path, source_override=before_text
                )
                before_counts = Counter((code, msg) for code, msg, _ in before_issues)
                new_issues: list[tuple[str, str, int]] = []
                for code, msg, line in after_issues:
                    key = (code, msg)
                    if before_counts[key] > 0:
                        before_counts[key] -= 1
                    else:
                        new_issues.append((code, msg, line))
                after_issues = new_issues
            for code, msg, line in after_issues[:8]:
                notes.append(f"L{line} [{code}]: {msg}")

    # 3. AST 公开签名变更 -> 自动提醒静态调用方
    if before_text is not None and file_path.is_relative_to(root):
        try:
            after_text = file_path.read_text(encoding="utf-8")
        except OSError:
            after_text = ""
        if after_text:
            old_sigs = _public_signatures(before_text)
            new_sigs = _public_signatures(after_text)
            changed = sorted(
                name for name in old_sigs if name not in new_sigs or new_sigs[name] != old_sigs[name]
            )
            if changed:
                rel = file_path.relative_to(root).as_posix()
                callers = static_callers_for_file(root, rel)
                if callers:
                    notes.append(
                        f"[Contract] Public surface changed ({', '.join(changed[:5])}); "
                        f"static importers to verify: {', '.join(callers[:8])}"
                    )

    return notes
