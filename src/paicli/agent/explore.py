"""按需并行只读调研子 Agent（Explore）：隔离上下文并行调查多个子系统并返回带行号证据的简报。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from paicli.agent.query import query
from paicli.config import PaiCliConfig
from paicli.harness.evidence import current_run, task_scope
from paicli.llm.base import LlmClient
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.registry import ToolRegistry
from paicli.types import Usage

_EXPLORE_READ_TOOLS = {
    "read_file",
    "grep",
    "glob",
    "list_dir",
    "directory_tree",
    "get_file_info",
    "search_code",
    "read_evidence",
    "code_map",
}

_EXPLORE_SYSTEM_PROMPT = """You are a focused read-only Codebase Explorer subagent in PaiCLI.
Your job is to investigate ONE assigned question quickly and accurately using read-only tools only.
Guidelines:
- Stay focused on the assigned question and focus_paths; follow direct dependencies only when needed.
- Prefer `code_map`, `grep`, and targeted `read_file` calls. Never propose or claim file edits.
- Finish within a few turns as soon as you have verified facts.
- Return a concise, structured report citing exact `path:line`, symbol names, observed logic/contracts,
  any concrete defects found, and explicit uncertainties if something was not verified.
- Match the language of the assigned question.
"""


class ExploreRunner:
    """并行调度 1–4 个独立只读子 Agent，将总用量与事件汇总回主会话。"""

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        base_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.base_registry = base_registry
        self.config = replace(
            config,
            features=replace(config.features, memory=False, skill=False),
            policy=replace(config.policy, path_guard_enabled=True),
        )
        self.cwd = cwd
        self.event_sink = event_sink
        self.usage = Usage()
        self.calls = 0

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def _read_only_registry(self, code_map_tool: Tool | None = None) -> ToolRegistry:
        reg = ToolRegistry()
        for name in self.base_registry.list_names():
            tool = self.base_registry.get(name)
            if (
                tool is not None
                and name in _EXPLORE_READ_TOOLS
                and tool.is_read_only
                and tool.is_concurrency_safe
            ):
                reg.register(tool)
        if code_map_tool is not None and reg.get("code_map") is None:
            reg.register(code_map_tool)
        return reg

    async def run_questions(
        self,
        data: dict[str, Any],
        _context: ToolContext,
        *,
        code_map_tool: Tool | None = None,
    ) -> ToolResult:
        raw_items = data.get("questions")
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 4:
            return ToolResult("explore 需要 1 至 4 条独立的调研问题。", is_error=True)

        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items, 1):
            if not isinstance(item, dict):
                return ToolResult("每条 explore 调研项必须是对象。", is_error=True)
            question = str(item.get("question") or "").strip()
            if not question:
                return ToolResult(f"第 {idx} 条 explore 缺少 question。", is_error=True)
            topic = str(item.get("topic") or f"explore-{idx}").strip()[:48]
            focus = item.get("focus_paths") or []
            if not isinstance(focus, list):
                focus = []
            normalized.append(
                {
                    "id": f"explore-{idx}",
                    "topic": topic,
                    "question": question,
                    "focus_paths": [str(p) for p in focus if isinstance(p, str) and p.strip()][:12],
                }
            )

        registry = self._read_only_registry(code_map_tool=code_map_tool)
        sem = asyncio.Semaphore(max(1, min(4, self.config.team.max_active_workers)))

        if run := current_run():
            run.record(
                "explore.started",
                count=len(normalized),
                topics=[item["topic"] for item in normalized],
            )
        self._emit(
            {
                "type": "explore_status",
                "phase": "started",
                "count": len(normalized),
                "topics": [item["topic"] for item in normalized],
            }
        )

        async def _run_one(item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                with task_scope(item["id"], 1):
                    focus_text = (
                        f"Focus paths: {', '.join(item['focus_paths'])}\n"
                        if item["focus_paths"]
                        else ""
                    )
                    prompt = (
                        f"Topic: {item['topic']}\n"
                        f"{focus_text}"
                        f"Question: {item['question']}\n"
                        "Return exact file paths, line numbers, symbol names, and concrete findings."
                    )
                    parts: list[str] = []
                    sub_usage = Usage()
                    sub_turns = 0
                    status = "completed"
                    try:
                        async with asyncio.timeout(min(90.0, max(20.0, self.config.tools.timeout))):
                            async for ev in query(
                                llm_client=self.llm_client,
                                tool_registry=registry,
                                system_prompt=_EXPLORE_SYSTEM_PROMPT,
                                user_message=prompt,
                                history=[],
                                cwd=self.cwd,
                                config=self.config,
                                approval_callback=None,
                                max_turns=6,
                            ):
                                kind = ev.get("type")
                                if kind == "text_delta":
                                    parts.append(str(ev.get("text") or ""))
                                elif kind == "done":
                                    sub_usage = Usage.from_mapping(ev.get("usage") or {})
                                    sub_turns = int(ev.get("total_turns") or 0)
                                    status = str(ev.get("status") or "completed")
                                elif kind == "error":
                                    status = "failed"
                                    parts.append(f"[Explorer error: {ev.get('error')}]")
                                    break
                    except TimeoutError:
                        status = "timed_out"
                        parts.append("[Explorer timed out; partial findings retained]")

                    self.usage = self.usage + sub_usage
                    self.calls += sub_turns
                    self._emit(
                        {
                            "type": "explore_status",
                            "phase": "finished",
                            "topic": item["topic"],
                            "status": status,
                            "turns": sub_turns,
                        }
                    )
                    return {
                        "topic": item["topic"],
                        "status": status,
                        "findings": "".join(parts).strip()[:4000],
                    }

        results = await asyncio.gather(*(_run_one(item) for item in normalized))
        if run := current_run():
            run.record(
                "explore.finished",
                count=len(results),
                statuses=[r["status"] for r in results],
            )
        return ToolResult(json.dumps({"reports": results}, ensure_ascii=False, indent=2))


def make_explore_tool(runner: ExploreRunner, *, code_map_tool: Tool | None = None) -> Tool:
    schema = object_schema(
        {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "description": "1 to 4 independent read-only investigation tasks to execute in parallel.",
                "items": object_schema(
                    {
                        "topic": {"type": "string", "description": "Short label (e.g. 'auth', 'cache')"},
                        "question": {"type": "string", "description": "Concrete question to investigate"},
                        "focus_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional repository-relative directories or files to inspect first",
                        },
                    },
                    ["question"],
                ),
            }
        },
        ["questions"],
    )

    async def handler(data: dict[str, Any], context: ToolContext) -> ToolResult:
        return await runner.run_questions(data, context, code_map_tool=code_map_tool)

    return Tool(
        name="explore",
        description=(
            "Run 1 to 4 independent read-only Explorer subagents in parallel to inspect separate "
            "subsystems, call chains, or defect hypotheses without bloating the main context."
        ),
        parameters=schema,
        handler=handler,
        is_read_only=True,
        is_concurrency_safe=False,
        timeout=120.0,
        required_keys=["questions"],
    )
