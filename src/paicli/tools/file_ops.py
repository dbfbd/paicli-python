"""可复用的文件操作实现，覆盖读取、写入、精确替换、遍历与内容搜索。

工具定义和参数适配位于 builtins.py；本模块集中处理路径、文件内容和操作结果。
"""

from __future__ import annotations

import glob as glob_module
import re
from dataclasses import dataclass
from pathlib import Path

from paicli.lsp import diagnose_file
from paicli.policy import PathGuard

# ---------------------------------------------------------------------------
# 文件操作返回类型
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileOpResult:
    """文件操作统一返回值，包含文本、错误标记和可选诊断信息。"""

    content: str
    is_error: bool = False
    display_summary: str | None = None


# ---------------------------------------------------------------------------
# 路径和文件过滤
# ---------------------------------------------------------------------------

SKIP_DIRS = frozenset({".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"})
MAX_FILE_SIZE = 1_000_000  # 文件大小上限：一百万字节。


def resolve_path(cwd: str, value: str, path_guard_enabled: bool = True) -> Path:
    """相对工作目录解析目标路径，并按配置执行工作区路径检查。"""
    if path_guard_enabled:
        return PathGuard(cwd).validate(value)
    path = Path(value)
    return path if path.is_absolute() else Path(cwd).resolve() / path


def skip_file(path: Path) -> bool:
    """判断文件是否因目录排除、大小或二进制类型而应跳过。"""
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    try:
        return path.stat().st_size > MAX_FILE_SIZE
    except OSError:
        return True


def _line_numbered(content: str, offset: int = 1) -> str:
    """为文本添加从 1 开始的行号，offset 用于保留原文件中的位置。"""
    lines = content.splitlines()
    return "\n".join(f"{idx + offset}: {line}" for idx, line in enumerate(lines))


# ---------------------------------------------------------------------------
# 读取文件
# ---------------------------------------------------------------------------


def read_file(
    cwd: str,
    path: str,
    *,
    offset: int = 1,
    limit: int = 500,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """读取带行号的文本片段；offset 从 1 开始，limit 限制返回行数。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.is_file():
        return FileOpResult(f"Not a file: {resolved}", is_error=True)

    try:
        raw = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FileOpResult(f"Failed to read {resolved}: {exc}", is_error=True)

    offset = max(offset, 1)
    lines = raw.splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{idx + offset}: {line}" for idx, line in enumerate(selected))
    rel = _relative_to(resolved, cwd)
    return FileOpResult(numbered, display_summary=f"Read {rel}")


def read_file_raw(
    cwd: str,
    path: str,
    *,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """读取不带行号的原始文本，供需要完整内容的调用方使用。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.is_file():
        return FileOpResult(f"Not a file: {resolved}", is_error=True)
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FileOpResult(f"Failed to read {resolved}: {exc}", is_error=True)
    rel = _relative_to(resolved, cwd)
    return FileOpResult(content, display_summary=f"Read {rel}")


# ---------------------------------------------------------------------------
# 写入文件
# ---------------------------------------------------------------------------


def write_file(
    cwd: str,
    path: str,
    content: str,
    *,
    append: bool = False,
    path_guard_enabled: bool = True,
    run_diagnostics: bool = True,
) -> FileOpResult:
    """写入或追加文本，可对 Python 文件执行语法诊断并把诊断结果附在返回值中。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    encoded = content.encode("utf-8")
    if len(encoded) > 5 * 1024 * 1024:
        return FileOpResult("write_file rejected: content exceeds 5 MB", is_error=True)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    try:
        with resolved.open(mode, encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return FileOpResult(f"Failed to write {resolved}: {exc}", is_error=True)

    rel = _relative_to(resolved, cwd)
    suffix = ""
    if run_diagnostics and resolved.suffix == ".py":
        diagnostics = diagnose_file(resolved)
        if diagnostics:
            suffix = "\n\nDiagnostics:\n" + "\n".join(diagnostics)
    return FileOpResult(f"Wrote {rel}{suffix}", display_summary=f"Wrote {rel}")


# ---------------------------------------------------------------------------
# 在原始文本中精确替换匹配片段。
# ---------------------------------------------------------------------------


def edit_file(
    cwd: str,
    path: str,
    old_text: str,
    new_text: str,
    *,
    path_guard_enabled: bool = True,
    dry_run: bool = False,
) -> FileOpResult:
    """在原始文本中精确替换首个匹配片段；dry_run 只生成差异，不写文件。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.is_file():
        return FileOpResult(f"Not a file: {resolved}", is_error=True)

    try:
        original = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return FileOpResult(f"Failed to read {resolved}: {exc}", is_error=True)

    if old_text not in original:
        return FileOpResult(
            "edit_file failed: `old_text` not found in the file. "
            "Provide the exact existing text to replace.",
            is_error=True,
        )

    new_content = original.replace(old_text, new_text, 1)
    if new_content == original:
        return FileOpResult("edit_file: no changes made (old_text == new_text).")

    # 生成紧凑的修改概览。
    diff_lines = _build_diff_summary(old_text, new_text)

    if dry_run:
        return FileOpResult(
            f"[DRY RUN] Would edit {_relative_to(resolved, cwd)}\n" + diff_lines,
            display_summary=f"Dry-run edit {_relative_to(resolved, cwd)}",
        )

    try:
        with resolved.open("w", encoding="utf-8") as handle:
            handle.write(new_content)
    except OSError as exc:
        return FileOpResult(f"Failed to write {resolved}: {exc}", is_error=True)

    rel = _relative_to(resolved, cwd)
    suffix = ""
    if resolved.suffix == ".py":
        diagnostics = diagnose_file(resolved)
        if diagnostics:
            suffix = "\n\nDiagnostics:\n" + "\n".join(diagnostics)

    return FileOpResult(
        f"Edited {rel}\n" + diff_lines + suffix,
        display_summary=f"Edited {rel}",
    )


def _build_diff_summary(old_text: str, new_text: str) -> str:
    """生成紧凑的修改概览，供工具结果快速展示。"""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    parts: list[str] = []
    if len(old_lines) <= 5 and len(new_lines) <= 5:
        for line in old_lines:
            parts.append(f"-{line}")
        for line in new_lines:
            parts.append(f"+{line}")
    else:
        parts.append(f"--- removed {len(old_lines)} line(s)")
        parts.append(f"+++ added {len(new_lines)} line(s)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 列出目录
# ---------------------------------------------------------------------------


def list_directory(
    cwd: str,
    path: str,
    *,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """列出目录直属内容，并用末尾斜杠标记子目录。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.is_dir():
        return FileOpResult(f"Not a directory: {resolved}", is_error=True)

    rows: list[str] = []
    try:
        children = sorted(
            resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        )
    except OSError as exc:
        return FileOpResult(f"Failed to list {resolved}: {exc}", is_error=True)

    for child in children:
        marker = "/" if child.is_dir() else ""
        rows.append(f"{child.name}{marker}")
    return FileOpResult("\n".join(rows) or "(empty directory)")


# ---------------------------------------------------------------------------
# 通配查找
# ---------------------------------------------------------------------------


def glob_files(
    cwd: str,
    pattern: str,
    *,
    limit: int = 100,
) -> FileOpResult:
    """按相对工作目录的通配模式查找文件。"""
    root = Path(cwd).resolve()
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return FileOpResult("glob pattern must stay inside workspace", is_error=True)

    matches = glob_module.glob(str(root / pattern), recursive=True)
    rels: list[str] = []
    for match in sorted(matches):
        mp = Path(match).resolve()
        try:
            rels.append(str(mp.relative_to(root)))
        except ValueError:
            continue
        if len(rels) >= limit:
            break
    return FileOpResult("\n".join(rels) or "(no matches)")


# ---------------------------------------------------------------------------
# 带深度限制的递归目录树
# ---------------------------------------------------------------------------


def directory_tree(
    cwd: str,
    path: str = ".",
    *,
    max_depth: int = 3,
    path_guard_enabled: bool = True,
    exclude_patterns: tuple[str, ...] | None = None,
) -> FileOpResult:
    """递归生成目录树，并用 max_depth 限制深度和输出规模。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.is_dir():
        return FileOpResult(f"Not a directory: {resolved}", is_error=True)

    exclude = set(exclude_patterns) if exclude_patterns else SKIP_DIRS

    lines: list[str] = [f"{resolved.name}/"]
    _walk_tree(resolved, resolved, "", max_depth, exclude, lines)
    return FileOpResult("\n".join(lines))


def _walk_tree(
    root: Path,
    current: Path,
    prefix: str,
    max_depth: int,
    exclude: set[str],
    lines: list[str],
) -> None:
    if max_depth <= 0:
        return
    try:
        entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError:
        return
    for i, child in enumerate(entries):
        if child.name in exclude:
            continue
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        marker = "/" if child.is_dir() else ""
        lines.append(f"{prefix}{connector}{child.name}{marker}")
        if child.is_dir():
            extension = "    " if is_last else "│   "
            _walk_tree(root, child, prefix + extension, max_depth - 1, exclude, lines)


# ---------------------------------------------------------------------------
# 文件内容搜索
# ---------------------------------------------------------------------------


def grep(
    cwd: str,
    pattern: str,
    *,
    path: str = ".",
    limit: int = 100,
    use_regex: bool = True,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """在指定路径内搜索文本；use_regex 控制正则匹配或普通子串匹配。"""
    root = Path(cwd).resolve()
    start = resolve_path(cwd, path, path_guard_enabled)

    try:
        compiled = re.compile(pattern) if use_regex else None
    except re.error as exc:
        return FileOpResult(f"invalid regex: {exc}", is_error=True)

    matches: list[str] = []
    files = [start] if start.is_file() else [p for p in start.rglob("*") if p.is_file()]

    for file_path in files:
        if skip_file(file_path):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            found = bool(compiled.search(line)) if compiled else pattern in line
            if found:
                matches.append(f"{file_path.relative_to(root)}:{line_number}: {line.strip()}")
                if len(matches) >= limit:
                    return FileOpResult("\n".join(matches))

    return FileOpResult("\n".join(matches) or "(no matches)")


# ---------------------------------------------------------------------------
# 文件元信息
# ---------------------------------------------------------------------------


def get_file_info(
    cwd: str,
    path: str,
    *,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """读取文件或目录的类型、大小及修改时间等元信息。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if not resolved.exists():
        return FileOpResult(f"Path does not exist: {resolved}", is_error=True)

    try:
        stat = resolved.stat()
    except OSError as exc:
        return FileOpResult(f"Failed to stat {resolved}: {exc}", is_error=True)

    kind = "directory" if resolved.is_dir() else "file"
    size = stat.st_size
    mtime = stat.st_mtime
    rel = _relative_to(resolved, cwd)

    info = (
        f"Path: {rel}\n"
        f"Type: {kind}\n"
        f"Size: {_human_size(size)}\n"
        f"Modified: {mtime:.0f}\n"
        f"Permissions: {oct(stat.st_mode & 0o777)}"
    )
    return FileOpResult(info)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _relative_to(path: Path, cwd: str) -> Path:
    """优先返回相对工作目录的路径，无法转换时返回绝对路径。"""
    try:
        return path.relative_to(Path(cwd).resolve())
    except ValueError:
        return path


def _human_size(size: int) -> str:
    """将字节数格式化为便于阅读的大小。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}" if isinstance(size, float) else f"{size}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
