"""自动化消融评测运行器：支持 minimal_react / paicli_react / paicli_full / paicli_team。"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from evals.fixtures import task_workspace
from evals.grader import grade_task, list_changed_files
from evals.minimal_react import run_minimal_react
from evals.tasks import select_tasks
from evals.types import EvalMode, EvalSuiteReport, EvalTask, RunTelemetry, TaskRunRecord
from paicli.agent import Agent
from paicli.agent.query import query as _raw_query
from paicli.config import PaiCliConfig, load_config
from paicli.harness.evidence import EvidenceRun
from paicli.llm import create_llm_client
from paicli.llm.pricing import calculate_costs
from paicli.prompt import PromptAssembler
from paicli.tools import ToolRegistry, get_builtin_tools
from paicli.types import Usage

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            timeout=10,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _extract_cost_usd(cost: dict[str, Any] | None) -> float:
    if not isinstance(cost, dict):
        return 0.0
    usd = cost.get("usd")
    if isinstance(usd, dict):
        return float(usd.get("total") or 0.0)
    return 0.0


def _parse_evidence_telemetry(evidence_dir: str | None) -> dict[str, Any]:
    """解析 EvidenceRun 落盘的 events.jsonl，还原真实路由路径、工具次数与首次集成通过情况。"""
    info: dict[str, Any] = {
        "routes_seen": [],
        "tool_calls": 0,
        "worker_count": 0,
        "first_try_integration_pass": None,
    }
    if not evidence_dir:
        return info
    events_path = Path(evidence_dir) / "events.jsonl"
    if not events_path.is_file():
        return info

    routes: list[str] = []
    workers: set[str] = set()
    tool_calls = 0
    saw_team_check = False
    saw_team_repair = False
    first_round_failed = False

    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(event.get("type") or "")
        if kind == "tool.started":
            tool_calls += 1
            tname = str(event.get("tool_name") or "")
            if tname in {"code_map", "impact_analysis"} and "code_map" not in routes:
                routes.append("code_map")
            elif tname == "explore" and "explore" not in routes:
                routes.append("explore")
            elif tname == "plan_change" and "plan" not in routes:
                routes.append("plan")
            elif tname == "delegate_work" and "collaboration" not in routes:
                routes.append("collaboration")
            elif tname == "debug_failure" and "debug" not in routes:
                routes.append("debug")
        elif kind == "workflow.started":
            wf = str(event.get("workflow") or "")
            if wf and wf not in routes:
                routes.append(wf)
        elif kind == "explore.started":
            if "explore" not in routes:
                routes.append("explore")
            for idx, topic in enumerate(event.get("topics") or [], 1):
                workers.add(f"explore:{idx}:{topic}")
        elif kind == "collaboration.started":
            if "collaboration" not in routes:
                routes.append("collaboration")
            for tid in event.get("tasks") or []:
                workers.add(str(tid))
        elif kind == "team.planned":
            if "team" not in routes:
                routes.append("team")
            for tid in event.get("tasks") or []:
                workers.add(str(tid))
        elif kind == "team.worker.started":
            tid = str(event.get("task_id") or "")
            if tid:
                workers.add(tid)
        elif kind == "team.repair":
            saw_team_repair = True
        elif kind == "team.checked":
            saw_team_check = True
            if not event.get("passed") and not saw_team_repair:
                first_round_failed = True
        elif kind == "debug.baseline":
            if "debug" not in routes:
                routes.append("debug")

    info["routes_seen"] = routes or ["react"]
    info["tool_calls"] = tool_calls
    info["worker_count"] = len(workers)
    if saw_team_check:
        info["first_try_integration_pass"] = not first_round_failed and not saw_team_repair
    return info


def _primary_route(routes_seen: list[str], mode: EvalMode) -> str:
    if mode == "paicli_team":
        return "team"
    if mode in {"minimal_react", "paicli_react"}:
        return "react"
    for priority in ("debug", "team", "collaboration", "explore", "plan", "code_map"):
        if priority in routes_seen:
            return priority
    return "react"


def _eval_config(base_config: PaiCliConfig, *, cwd: str, mode: EvalMode) -> PaiCliConfig:
    """构造评测专用配置：禁用外部 MCP 与长期记忆污染，确保每次任务完全隔离可复现。"""
    return replace(
        base_config,
        features=replace(
            base_config.features,
            mcp=False,
            memory=False,
            skill=False,
            audit_log=False,
        ),
        policy=replace(
            base_config.policy,
            hitl_mode="auto",
            path_guard_enabled=True,
        ),
        plan=replace(
            base_config.plan,
            auto_route=(mode == "paicli_full"),
        ),
    )


async def _run_single_paicli_react(
    *,
    llm_client,
    registry: ToolRegistry,
    config: PaiCliConfig,
    cwd: str,
    prompt: str,
    max_turns: int,
) -> dict[str, Any]:
    """纯 PaiCLI ReAct（关闭 Collaboration/Specialized 路由，仅保留核心 query + context + builtins）。"""
    run = EvidenceRun.create(cwd, "react")
    sys_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=registry.list_names(),
        model=llm_client.model_name,
        provider=llm_client.provider_name,
    ).build_static()
    text = ""
    turns = 0
    tool_calls = 0
    usage = Usage()
    cost: dict[str, Any] = {}
    status = "completed"
    async for event in _raw_query(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=sys_prompt,
        user_message=prompt,
        history=[],
        cwd=cwd,
        config=config,
        approval_callback=lambda _req: "approve",
        max_turns=max_turns,
    ):
        kind = event.get("type")
        if kind == "text_delta":
            text += str(event.get("text") or "")
        elif kind == "tool_call":
            tool_calls += 1
        elif kind == "error":
            status = "failed"
            run.finish(status)
            raise event["error"]
        elif kind == "done":
            status = str(event.get("status") or "completed")
            turns = int(event.get("total_turns") or 0)
            usage = Usage.from_mapping(event.get("usage") or {})
            cost = dict(event.get("cost") or calculate_costs(llm_client, usage) or {})
    run.finish(status)
    return {
        "text": text,
        "turns": turns,
        "tool_calls": tool_calls,
        "usage": usage,
        "cost": cost,
        "run_id": run.run_id,
        "evidence_dir": str(run.directory),
    }


async def run_task_in_mode(
    task: EvalTask,
    mode: EvalMode,
    base_config: PaiCliConfig,
) -> TaskRunRecord:
    """在独立临时 Git 仓库中运行单个评测任务并打分。"""
    telemetry = RunTelemetry(route_taken="react")
    response_text = ""

    with task_workspace(task) as repo_dir:
        cwd = str(repo_dir)
        cfg = _eval_config(base_config, cwd=cwd, mode=mode)
        client = create_llm_client(cfg.llm)
        registry = ToolRegistry()
        registry.register_all(get_builtin_tools())

        started = perf_counter()
        try:
            async with asyncio.timeout(task.timeout_sec):
                if mode == "minimal_react":
                    out = await run_minimal_react(
                        llm_client=client,
                        config=cfg,
                        cwd=cwd,
                        prompt=task.prompt,
                        max_turns=task.max_turns,
                    )
                    response_text = out["text"]
                    usage: Usage = out["usage"]
                    telemetry.turns = out["turns"]
                    telemetry.tool_calls = out["tool_calls"]
                    telemetry.input_tokens = usage.input_tokens
                    telemetry.output_tokens = usage.output_tokens
                    telemetry.cache_hit_tokens = usage.cache_hit_tokens
                    telemetry.total_tokens = usage.total_tokens
                    telemetry.cost_usd = _extract_cost_usd(out["cost"])
                    telemetry.routes_seen = ["react"]
                    telemetry.route_taken = "react"

                elif mode == "paicli_react":
                    out = await _run_single_paicli_react(
                        llm_client=client,
                        registry=registry,
                        config=cfg,
                        cwd=cwd,
                        prompt=task.prompt,
                        max_turns=task.max_turns,
                    )
                    response_text = out["text"]
                    usage = out["usage"]
                    telemetry.turns = out["turns"]
                    telemetry.tool_calls = out["tool_calls"]
                    telemetry.input_tokens = usage.input_tokens
                    telemetry.output_tokens = usage.output_tokens
                    telemetry.cache_hit_tokens = usage.cache_hit_tokens
                    telemetry.total_tokens = usage.total_tokens
                    telemetry.cost_usd = _extract_cost_usd(out["cost"])
                    telemetry.run_id = out["run_id"]
                    telemetry.evidence_dir = out["evidence_dir"]
                    telemetry.routes_seen = ["react"]
                    telemetry.route_taken = "react"

                elif mode in {"paicli_full", "paicli_team"}:
                    agent_mode = "team" if mode == "paicli_team" else "react"
                    agent = Agent(
                        llm_client=client,
                        tool_registry=registry,
                        config=cfg,
                        cwd=cwd,
                        approval_callback=lambda _req: "approve",
                        mode=agent_mode,
                        max_turns=task.max_turns,
                    )
                    text_parts: list[str] = []
                    done_event: dict[str, Any] = {}
                    observed_tool_calls = 0
                    async for event in agent.run(task.prompt):
                        etype = event.get("type")
                        if etype == "text_delta":
                            text_parts.append(str(event.get("text") or ""))
                        elif etype == "tool_call":
                            observed_tool_calls += 1
                        elif etype == "error":
                            raise event["error"]
                        elif etype == "done":
                            done_event = event

                    response_text = "".join(text_parts)
                    # 显式 team 模式仅生成补丁文件而不自动应用到主工作区；
                    # 若 team 验收通过并产出补丁，评测器模拟用户确认后应用到 repo_dir 以统一测试。
                    if mode == "paicli_team" and done_event.get("status") == "completed":
                        _apply_team_patch_if_present(done_event.get("evidence_dir"), repo_dir)

                    usage = Usage.from_mapping(done_event.get("usage") or {})
                    telemetry.turns = int(done_event.get("total_turns") or 0)
                    telemetry.input_tokens = usage.input_tokens
                    telemetry.output_tokens = usage.output_tokens
                    telemetry.cache_hit_tokens = usage.cache_hit_tokens
                    telemetry.total_tokens = usage.total_tokens
                    telemetry.cost_usd = _extract_cost_usd(done_event.get("cost"))
                    telemetry.run_id = done_event.get("run_id")
                    telemetry.evidence_dir = done_event.get("evidence_dir")
                    if done_event.get("status", "completed") != "completed":
                        telemetry.error = f"status={done_event.get('status')}: {done_event.get('reason', '')}"

                    ev = _parse_evidence_telemetry(telemetry.evidence_dir)
                    telemetry.routes_seen = ev["routes_seen"]
                    telemetry.route_taken = _primary_route(ev["routes_seen"], mode)
                    telemetry.tool_calls = max(observed_tool_calls, ev["tool_calls"])
                    telemetry.worker_count = ev["worker_count"]
                    telemetry.first_try_integration_pass = ev["first_try_integration_pass"]
        except TimeoutError:
            telemetry.error = f"timed_out after {task.timeout_sec:.0f}s"
        except Exception as exc:  # noqa: BLE001
            telemetry.error = f"{type(exc).__name__}: {exc}"
        finally:
            telemetry.wall_clock_sec = round(perf_counter() - started, 2)

        if mode == "paicli_full" and task.forbidden_routes:
            telemetry.misrouted = any(r in task.forbidden_routes for r in telemetry.routes_seen)

        changed = list_changed_files(repo_dir)
        grade = grade_task(task, repo_dir, response_text)
        return TaskRunRecord(
            task_id=task.id,
            suite=task.suite,
            title=task.title,
            mode=mode,
            grade=grade,
            telemetry=telemetry,
            response_excerpt=response_text[:600],
            changed_files=changed,
        )


def _apply_team_patch_if_present(evidence_dir: str | None, repo_dir: Path) -> None:
    if not evidence_dir:
        return
    patch_file = Path(evidence_dir) / "team.patch"
    if not patch_file.is_file():
        return
    patch_bytes = patch_file.read_bytes()
    if not patch_bytes:
        return
    subprocess.run(
        ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
        cwd=repo_dir,
        input=patch_bytes,
        capture_output=True,
        check=False,
        timeout=30,
    )


def format_markdown_report(report: EvalSuiteReport) -> str:
    """将评测结果格式化为便于审阅的 Markdown 评分卡。"""
    lines = [
        f"# PaiCLI Eval Scorecard — `{report.label}`",
        "",
        f"- **Time**: {report.created_at}",
        f"- **Git Commit**: `{report.git_commit}`",
        f"- **Model**: `{report.provider}/{report.model}`",
        f"- **Suites**: {', '.join(report.suites)} | **Modes**: {', '.join(report.modes)}",
        "",
        "## 1. 按模式汇总（Pareto Summary）",
        "",
        "| Mode | Passed | Pass@1 | Avg Wall (s) | Total Tokens | Cache Hit % | Cost ($) | Misrouted | Tampered |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in report.modes:
        recs = [r for r in report.records if r.mode == mode]
        if not recs:
            continue
        passed = sum(1 for r in recs if r.grade.passed)
        total = len(recs)
        avg_wall = sum(r.telemetry.wall_clock_sec for r in recs) / total
        tot_tok = sum(r.telemetry.total_tokens for r in recs)
        in_tok = sum(r.telemetry.input_tokens for r in recs)
        hit_tok = sum(r.telemetry.cache_hit_tokens for r in recs)
        hit_pct = (100.0 * hit_tok / in_tok) if in_tok else 0.0
        tot_cost = sum(r.telemetry.cost_usd for r in recs)
        misrouted = sum(1 for r in recs if r.telemetry.misrouted)
        tampered = sum(1 for r in recs if r.grade.tampered)
        lines.append(
            f"| `{mode}` | {passed}/{total} | {100.0 * passed / total:.1f}% | {avg_wall:.2f} | "
            f"{tot_tok:,} | {hit_pct:.1f}% | ${tot_cost:.4f} | {misrouted} | {tampered} |"
        )

    lines.extend(
        [
            "",
            "## 2. 分任务明细",
            "",
            "| Suite | Task | Mode | Pass | Wall(s) | Turns | Tools | Tokens | Route | Workers | 1st-Try Int | Note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rec in report.records:
        mark = "PASS" if rec.grade.passed else "FAIL"
        first_try = (
            "-"
            if rec.telemetry.first_try_integration_pass is None
            else ("Y" if rec.telemetry.first_try_integration_pass else "N")
        )
        note_parts = []
        if rec.grade.tampered:
            note_parts.append("TAMPERED")
        if rec.grade.p2p_regressions:
            note_parts.append(f"p2p_reg={rec.grade.p2p_regressions}")
        if rec.telemetry.misrouted:
            note_parts.append("MISROUTED")
        if rec.telemetry.error:
            note_parts.append(rec.telemetry.error[:60])
        elif not rec.grade.passed:
            note_parts.append(
                "public_fail" if not rec.grade.public_passed else "hidden_fail"
            )
        lines.append(
            f"| {rec.suite} | `{rec.task_id}` | `{rec.mode}` | **{mark}** | "
            f"{rec.telemetry.wall_clock_sec:.1f} | {rec.telemetry.turns} | {rec.telemetry.tool_calls} | "
            f"{rec.telemetry.total_tokens:,} | `{rec.telemetry.route_taken}` | "
            f"{rec.telemetry.worker_count} | {first_try} | {'; '.join(note_parts) or 'ok'} |"
        )
    return "\n".join(lines) + "\n"


async def run_eval(
    *,
    label: str,
    suites: list[str] | None = None,
    task_ids: list[str] | None = None,
    modes: list[EvalMode],
    output_dir: Path = RESULTS_DIR,
) -> EvalSuiteReport:
    config = load_config(project_root=Path(__file__).resolve().parent.parent)
    tasks = select_tasks(suites=suites, task_ids=task_ids)
    report = EvalSuiteReport(
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git_head_sha(),
        provider=config.llm.provider,
        model=config.llm.model,
        modes=list(modes),
        suites=sorted({t.suite for t in tasks}),
    )

    for task in tasks:
        for mode in modes:
            print(f"[eval:{label}] running {task.id} ({task.suite}) in mode={mode} ...", flush=True)
            rec = await run_task_in_mode(task, mode, config)
            report.records.append(rec)
            status = "PASS" if rec.grade.passed else "FAIL"
            print(
                f"  -> {status} | wall={rec.telemetry.wall_clock_sec:.1f}s | "
                f"tokens={rec.telemetry.total_tokens:,} | route={rec.telemetry.route_taken} | "
                f"workers={rec.telemetry.worker_count}"
                + (f" | err={rec.telemetry.error}" if rec.telemetry.error else ""),
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    md_path = output_dir / f"{label}.md"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(format_markdown_report(report), encoding="utf-8")
    print(f"\nSaved report:\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PaiCLI benchmark and ablation runner")
    parser.add_argument("--label", default="run", help="Output label under evals/results/")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["A", "B", "C", "D"],
        help="Suites to run (default: all)",
    )
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["minimal_react", "paicli_full"],
        choices=["minimal_react", "paicli_react", "paicli_full", "paicli_team"],
        help="Ablation modes to compare",
    )
    args = parser.parse_args()
    asyncio.run(
        run_eval(
            label=args.label,
            suites=args.suites,
            task_ids=args.tasks,
            modes=args.modes,
        )
    )


if __name__ == "__main__":
    main()
