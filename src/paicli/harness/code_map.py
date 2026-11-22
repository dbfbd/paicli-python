"""零 LLM 成本的 AST 符号骨架图与静态影响范围分析。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from paicli.harness.impact import MAX_FILES, import_names, local_dependencies, plan_impact

_IGNORE_DIRS = {
    ".git",
    ".paicli",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    ".tox",
    ".eval_hidden",
}
_MAX_FILE_BYTES = 128 * 1024


def _iter_python_files(root: Path, *, max_files: int = MAX_FILES) -> list[str]:
    catalogue: list[str] = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(
            name for name in dirs if name not in _IGNORE_DIRS and not name.startswith(".")
        )
        for name in sorted(files):
            if name.endswith(".py"):
                catalogue.append((Path(directory) / name).relative_to(root).as_posix())
                if len(catalogue) >= max_files:
                    return catalogue
    return catalogue


def _format_args(args: ast.arguments) -> str:
    parts: list[str] = []
    posonly = list(args.posonlyargs)
    regular = list(args.args)
    all_pos = posonly + regular
    defaults_offset = len(all_pos) - len(args.defaults)
    for idx, arg in enumerate(all_pos):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if idx >= defaults_offset:
            piece += "=..."
        parts.append(piece)
        if posonly and idx == len(posonly) - 1:
            parts.append("/")
    if args.vararg is not None:
        var = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            var += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(var)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += "=..."
        parts.append(piece)
    if args.kwarg is not None:
        kw = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            kw += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(kw)
    return ", ".join(parts)


def extract_file_outline(path: Path, relative: str) -> dict[str, Any]:
    """提取单个 Python 文件的公开类、函数签名、方法与首行文档摘要。"""
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return {"path": relative, "symbols": [], "error": "file too large"}
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=relative)
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        return {"path": relative, "symbols": [], "error": type(exc).__name__}

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not (
                node.name.startswith("__") and node.name.endswith("__")
            ):
                continue
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sig = f"{prefix} {node.name}({_format_args(node.args)})"
            if node.returns is not None:
                sig += f" -> {ast.unparse(node.returns)}"
            doc = (ast.get_docstring(node) or "").splitlines()
            doc_hint = f"  # {doc[0].strip()[:80]}" if doc and doc[0].strip() else ""
            symbols.append(f"L{node.lineno}: {sig}{doc_hint}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            header = f"L{node.lineno}: class {node.name}({bases})" if bases else f"L{node.lineno}: class {node.name}"
            methods: list[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") and child.name != "__init__":
                        continue
                    mprefix = "async " if isinstance(child, ast.AsyncFunctionDef) else ""
                    msig = f"{mprefix}{child.name}({_format_args(child.args)})"
                    if child.returns is not None:
                        msig += f" -> {ast.unparse(child.returns)}"
                    methods.append(msig)
            if methods:
                symbols.append(f"{header}: {'; '.join(methods[:8])}")
            else:
                symbols.append(header)
    return {
        "path": relative,
        "lines": len(source.splitlines()),
        "symbols": symbols,
        "imports": import_names(relative, source),
    }


def _looks_like_test(relative: str) -> bool:
    p = Path(relative)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def build_code_map(
    root: str | Path,
    *,
    paths: list[str] | None = None,
    max_chars: int = 10_000,
) -> dict[str, Any]:
    """构建项目符号地图；指定 paths 时额外计算正/反向静态依赖、调用方与候选测试。"""
    base = Path(root).resolve()
    catalogue = _iter_python_files(base)
    if not catalogue:
        return {
            "text": "未发现项目内 Python 文件。",
            "scanned_files": 0,
            "affected": [],
            "candidate_tests": [],
        }

    focus_files: list[str] = []
    if paths:
        for raw in paths:
            if not isinstance(raw, str) or not raw.strip():
                continue
            target = (base / raw.strip()).resolve()
            if not target.is_relative_to(base) or not target.exists():
                continue
            if target.is_file() and target.suffix == ".py":
                focus_files.append(target.relative_to(base).as_posix())
            elif target.is_dir():
                prefix = target.relative_to(base).as_posix().rstrip("/")
                matched = [
                    name
                    for name in catalogue
                    if (not prefix or prefix == ".") or name == prefix or name.startswith(prefix + "/")
                ]
                focus_files.extend(matched)
        focus_files = list(dict.fromkeys(focus_files))

    # 全仓模式 vs 聚焦影响分析模式
    outlines: dict[str, dict[str, Any]] = {}
    imports_map: dict[str, set[str]] = {}
    for rel in catalogue:
        info = extract_file_outline(base / rel, rel)
        outlines[rel] = info
        imports_map[rel] = info.get("imports", set())

    deps = local_dependencies(imports_map)
    reverse_deps: dict[str, list[str]] = {name: [] for name in catalogue}
    for src, targets in deps.items():
        for dst in sorted(targets):
            reverse_deps.setdefault(dst, []).append(src)

    lines: list[str] = []
    affected_list: list[str] = []
    candidate_tests: list[str] = []

    if focus_files:
        impact = plan_impact(base, focus_files, [], catalogue, include_imports=True)
        affected_imports: dict[str, list[str]] = impact["affected_imports"]
        affected_list = sorted(affected_imports.keys())
        candidate_tests = list(impact["added_test_files"])

        lines.append(
            f"# Code Map & Impact Analysis ({len(focus_files)} focus / {len(catalogue)} scanned Python files)"
        )
        lines.append(f"Focus files: {', '.join(focus_files[:12])}")

        edges = [
            f"  - {name} -> imports {', '.join(imported)}"
            for name, imported in affected_imports.items()
            if imported
        ]
        lines.append("\n## Affected Reverse-Import Chain")
        lines.extend(edges[:30] or ["  (no local static import edges found)"])

        lines.append(
            f"\n## Candidate Tests ({len(candidate_tests)}): "
            + (", ".join(candidate_tests) if candidate_tests else "(none matched statically)")
        )

        # 优先展示聚焦文件与直接受影响的非测试源码符号骨架
        display_files = list(
            dict.fromkeys(
                focus_files
                + [f for f in affected_list if not _looks_like_test(f)]
                + [f for f in candidate_tests]
            )
        )
        lines.append("\n## Symbol Outlines (Affected Scope)")
    else:
        lines.append(f"# Repository Code Map ({len(catalogue)} Python files)")
        # 优先展示应用源码，再展示测试文件
        app_files = [f for f in catalogue if not _looks_like_test(f)]
        test_files = [f for f in catalogue if _looks_like_test(f)]
        display_files = app_files + test_files
        candidate_tests = test_files[:12]

    used = sum(len(line) + 1 for line in lines)
    omitted = 0
    for rel in display_files:
        info = outlines.get(rel, {})
        syms = info.get("symbols") or []
        importers = reverse_deps.get(rel, [])
        imported = sorted(deps.get(rel, set()))
        block_lines = [f"- {rel} ({info.get('lines', '?')}L)"]
        if imported:
            block_lines.append(f"    imports: {', '.join(imported[:8])}")
        if importers:
            block_lines.append(f"    imported_by: {', '.join(importers[:8])}")
        for sym in syms[:12]:
            block_lines.append(f"    {sym}")
        if len(syms) > 12:
            block_lines.append(f"    ... (+{len(syms) - 12} symbols)")
        block = "\n".join(block_lines)
        if used + len(block) + 1 > max_chars:
            omitted += 1
            continue
        lines.append(block)
        used += len(block) + 1

    if omitted:
        lines.append(f"... ({omitted} additional files omitted; pass narrower `paths` to zoom in)")

    return {
        "text": "\n".join(lines),
        "scanned_files": len(catalogue),
        "affected": affected_list or catalogue,
        "candidate_tests": candidate_tests,
    }


def static_callers_for_file(root: str | Path, target_relative: str) -> list[str]:
    """快速查找全仓中静态导入 target_relative 的直接调用方（供写后 Harness 提示使用）。"""
    base = Path(root).resolve()
    catalogue = _iter_python_files(base)
    if target_relative not in catalogue:
        catalogue.append(target_relative)
    imports_map: dict[str, set[str]] = {}
    for rel in catalogue:
        p = base / rel
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                continue
            imports_map[rel] = import_names(rel, p.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError, ValueError, RecursionError):
            continue
    deps = local_dependencies(imports_map)
    return sorted(src for src, targets in deps.items() if target_relative in targets)
