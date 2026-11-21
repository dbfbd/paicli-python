from __future__ import annotations

from traceback import format_exception_only
from pathlib import Path


def diagnose_file(path: str | Path) -> list[str]:
    file_path = Path(path)
    if file_path.suffix != ".py":
        return []
    try:
        compile(file_path.read_bytes(), str(file_path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return ["".join(format_exception_only(type(exc), exc)).rstrip()]
    return []
