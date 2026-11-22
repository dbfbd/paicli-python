"""统一解析目标项目的 Python 解释器、工具可执行文件与子进程运行环境。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_VENV_NAMES = (".venv", "venv")


def _venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _venv_python(venv_dir: Path) -> Path:
    return _venv_bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")


def find_project_venv(
    cwd: str | Path,
    *,
    extra_roots: tuple[str | Path, ...] = (),
) -> Path | None:
    """按 cwd、父目录（至 Git 根）以及补充项目根目录查找本地虚拟环境。"""
    candidates: list[Path] = []
    start = Path(cwd).resolve()
    candidates.append(start)
    for parent in start.parents:
        candidates.append(parent)
        if (parent / ".git").exists():
            break
    for extra in extra_roots:
        if extra:
            resolved = Path(extra).resolve()
            if resolved not in candidates:
                candidates.append(resolved)

    for directory in candidates:
        for name in _VENV_NAMES:
            venv_dir = directory / name
            if _venv_python(venv_dir).is_file():
                return venv_dir
    return None


def resolve_project_python(
    cwd: str | Path,
    *,
    explicit: str | Path | None = None,
    extra_roots: tuple[str | Path, ...] = (),
) -> str:
    """优先返回显式指定或目标项目 `.venv` 中的 Python，未命中时回落到当前解释器。"""
    if explicit:
        candidate = Path(explicit).expanduser().absolute()
        if candidate.is_file():
            return str(candidate)
    venv_dir = find_project_venv(cwd, extra_roots=extra_roots)
    if venv_dir is not None:
        return str(_venv_python(venv_dir))
    return sys.executable


def resolve_project_executable(
    program: str,
    cwd: str | Path,
    *,
    extra_roots: tuple[str | Path, ...] = (),
) -> str | None:
    """优先从目标项目 `.venv` 解析 CLI 可执行文件（如 pytest / ruff），再回落 PATH。"""
    lowered = program.lower()
    if lowered in {"python", "python3", "python.exe"}:
        return resolve_project_python(cwd, extra_roots=extra_roots)

    if "/" in program or "\\" in program:
        return shutil.which(str(Path(cwd) / program))

    venv_dir = find_project_venv(cwd, extra_roots=extra_roots)
    if venv_dir is not None:
        bin_dir = _venv_bin_dir(venv_dir)
        found = shutil.which(program, path=str(bin_dir))
        if found:
            return found
    return shutil.which(program)


def augment_env_for_venv(
    env: dict[str, str],
    cwd: str | Path,
    *,
    extra_roots: tuple[str | Path, ...] = (),
) -> dict[str, str]:
    """若目标项目存在虚拟环境，将其 Scripts/bin 前置到 PATH 并设置 VIRTUAL_ENV。"""
    venv_dir = find_project_venv(cwd, extra_roots=extra_roots)
    if venv_dir is None:
        return env
    result = dict(env)
    bin_str = str(_venv_bin_dir(venv_dir))
    existing_path = result.get("PATH", "")
    result["PATH"] = f"{bin_str}{os.pathsep}{existing_path}" if existing_path else bin_str
    result["VIRTUAL_ENV"] = str(venv_dir)
    return result
