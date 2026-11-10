"""按静态导入关系筛选有界回归候选；结果仅代表依赖关联，不代表执行覆盖。"""

from __future__ import annotations

import ast
from pathlib import Path

MAX_FILES, MAX_FILE_BYTES, MAX_BYTES, MAX_TESTS = 256, 256 * 1024, 4 * 1024 * 1024, 8


def _module(name):
    parts = list(Path(name).with_suffix("").parts)
    if parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(name, text):
    module, result = _module(name), set()
    for node in ast.walk(ast.parse(text, filename=name)):
        if isinstance(node, ast.Import):
            result.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = module if Path(name).stem == "__init__" else module.rpartition(".")[0]
                prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*prefix, *([base] if base else [])])
            result.update([base, *(f"{base}.{a.name}".strip(".") for a in node.names)])
    return result


def _test(name):
    p = Path(name)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def plan_impact(root, repair_files, targets, catalogue, test_sources=()):
    """在扫描和测试数量上限内反向追踪导入依赖，补充相关测试并报告未分析部分。"""
    root = Path(root).resolve()
    imports, unknown, total = {}, [], 0
    names = sorted(set(catalogue) | set(repair_files), key=lambda n: (n not in repair_files, n))
    for name in names[:MAX_FILES]:
        path = (root / name).resolve()
        try:
            if not path.is_relative_to(root) or path.suffix != ".py":
                raise ValueError("outside Python scope")
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or total + size > MAX_BYTES:
                raise ValueError("source byte limit")
            raw = path.read_bytes()
            if len(raw) != size:
                raise ValueError("source changed while reading")
            total += size
            imports[name] = _imports(name, raw.decode("utf-8-sig"))
        except (OSError, ValueError, SyntaxError, RecursionError) as exc:
            unknown.append(f"{name}: {type(exc).__name__}")
    if len(names) > MAX_FILES:
        unknown.append(f"Only {MAX_FILES}/{len(names)} Python files scanned")
    modules = {_module(name): name for name in imports}
    dependencies = {
        name: {
            modules[prefix]
            for dep in refs
            for prefix in [
                next(
                    (
                        dep[:i]
                        for i in range(len(dep), 0, -1)
                        if dep[:i] in modules and (i == len(dep) or dep[i] == ".")
                    ),
                    "",
                )
            ]
            if prefix
        }
        for name, refs in imports.items()
    }
    affected = set(repair_files)
    while True:
        related = affected | {name for name, deps in dependencies.items() if deps & affected}
        if related == affected:
            break
        affected = related
    mandatory = list(dict.fromkeys(t.split("::", 1)[0] for t in targets))
    candidates = [
        name
        for name in affected
        if (_test(name) or name in test_sources)
        and not any((root / name).is_relative_to(root / target) for target in mandatory)
    ]
    candidates.sort(key=lambda n: (-len(dependencies.get(n, set()) & set(repair_files)), n))
    selected, deferred = candidates[:MAX_TESTS], candidates[MAX_TESTS:]
    if deferred:
        unknown.append(f"{len(deferred)} optional test files deferred by the regression budget")
    return {
        "repair_files": list(repair_files),
        "targets": list(dict.fromkeys(mandatory + selected)),
        "mandatory_targets": mandatory,
        "added_test_files": selected,
        "deferred_test_files": deferred,
        "scanned_files": len(imports),
        "unknown": unknown[:32],
        "blocked": "Repair source cannot be analyzed"
        if any(n not in imports for n in repair_files)
        else "",
        "note": "Import dependencies are candidates only; "
        "dynamic imports and execution coverage are not inferred",
    }
