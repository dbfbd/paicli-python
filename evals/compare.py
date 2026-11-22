"""评测对比脚本：计算两次评测之间的 Pass@1、耗时加速比、Token 与缓存命中率差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_records(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["task_id"], r["mode"]): r for r in report.get("records", [])}


def compare_reports(base_path: Path, new_path: Path) -> str:
    base = _load(base_path)
    new = _load(new_path)
    base_idx = _index_records(base)
    new_idx = _index_records(new)
    common_keys = sorted(set(base_idx) & set(new_idx))

    lines = [
        f"# Eval Delta: `{base['label']}` (`{base['git_commit']}`) -> `{new['label']}` (`{new['git_commit']}`)",
        "",
        "| Task | Mode | Pass (Base -> New) | Wall (s) | Speedup | Tokens | Token Delta | Route (Base -> New) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    base_pass = new_pass = 0
    base_wall = new_wall = 0.0
    base_tok = new_tok = 0

    for task_id, mode in common_keys:
        b = base_idx[(task_id, mode)]
        n = new_idx[(task_id, mode)]
        bp = bool(b["grade"]["passed"])
        np = bool(n["grade"]["passed"])
        bw = float(b["telemetry"]["wall_clock_sec"])
        nw = float(n["telemetry"]["wall_clock_sec"])
        bt = int(b["telemetry"]["total_tokens"])
        nt = int(n["telemetry"]["total_tokens"])

        base_pass += int(bp)
        new_pass += int(np)
        base_wall += bw
        new_wall += nw
        base_tok += bt
        new_tok += nt

        speedup = f"{bw / nw:.2f}x" if nw > 0 else "-"
        tok_delta = f"{((nt - bt) / bt * 100):+.1f}%" if bt > 0 else "-"
        lines.append(
            f"| `{task_id}` | `{mode}` | {'PASS' if bp else 'FAIL'} -> {'PASS' if np else 'FAIL'} | "
            f"{bw:.1f} -> {nw:.1f} | {speedup} | {bt:,} -> {nt:,} | {tok_delta} | "
            f"`{b['telemetry']['route_taken']}` -> `{n['telemetry']['route_taken']}` |"
        )

    if common_keys:
        total = len(common_keys)
        overall_speedup = f"{base_wall / new_wall:.2f}x" if new_wall > 0 else "-"
        overall_tok = f"{((new_tok - base_tok) / base_tok * 100):+.1f}%" if base_tok > 0 else "-"
        lines.extend(
            [
                "",
                f"**Overall ({total} matched runs)**: "
                f"Pass {base_pass}/{total} -> {new_pass}/{total} | "
                f"Total Wall {base_wall:.1f}s -> {new_wall:.1f}s ({overall_speedup}) | "
                f"Total Tokens {base_tok:,} -> {new_tok:,} ({overall_tok})",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PaiCLI eval result JSON files")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(compare_reports(args.baseline, args.candidate))


if __name__ == "__main__":
    main()
