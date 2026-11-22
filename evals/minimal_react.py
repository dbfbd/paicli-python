"""极简轻量级单 Agent ReAct 对照基线（类似 mini-swe-agent）。

使用与 PaiCLI 完全相同的 LlmClient 和基础文件/命令工具，但零快照、零动态路由、
零 Team/Debug/Harness 额外开销，用作消融实验的下界速度与能力参照。
"""

from __future__ import annotations

import json
import platform
from typing import Any

from paicli.config import PaiCliConfig
from paicli.llm.base import LlmClient
from paicli.llm.pricing import calculate_costs
from paicli.tools.base import ToolContext
from paicli.tools.builtins import get_builtin_tools
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, Usage

_BASIC_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "grep",
    "execute_command",
}

_SYSTEM_PROMPT = (
    "You are a concise, direct Python coding agent running in a local repository. "
    "Use the provided tools to inspect files, make scoped edits, and run pytest to verify your changes. "
    "Do not modify protected test files unless explicitly asked. "
    "Reply concisely when done."
)


def _merge_delta(states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    idx = int(delta.get("index") or 0)
    slot = states.setdefault(
        idx,
        {
            "id": delta.get("id") or f"tool_{idx}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        slot["id"] = delta["id"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


async def run_minimal_react(
    *,
    llm_client: LlmClient,
    config: PaiCliConfig,
    cwd: str,
    prompt: str,
    max_turns: int = 24,
) -> dict[str, Any]:
    """运行纯单循环 ReAct 基线并返回响应文本与用量统计。"""
    registry = ToolRegistry()
    for tool in get_builtin_tools():
        if tool.name in _BASIC_TOOLS:
            registry.register(tool)

    tool_defs = registry.definitions()
    shell_hint = (
        "Execution OS: Windows (cmd.exe). Prefer file tools for reading; use execute_command for pytest."
        if platform.system() == "Windows"
        else "Execution OS: POSIX (/bin/sh). Prefer file tools for reading; use execute_command for pytest."
    )
    sys_prompt = f"{_SYSTEM_PROMPT}\n{shell_hint}\nWorking directory: {cwd}"
    ctx = ToolContext(cwd=cwd, config=config, approval_callback=lambda _req: "approve")

    messages: list[Message] = [Message(role="user", content=prompt)]
    total_usage = Usage()
    turns = 0
    tool_calls_count = 0
    final_text = ""

    while turns < max_turns:
        turns += 1
        text = ""
        reasoning = ""
        stop_reason = "end_turn"
        tool_states: dict[int, dict[str, Any]] = {}

        async for event in llm_client.chat(messages, tool_defs, system_prompt=sys_prompt):
            kind = event.get("type")
            if kind == "text_delta":
                text += str(event.get("text") or "")
            elif kind == "thinking_delta":
                reasoning += str(event.get("thinking") or "")
            elif kind == "tool_call_delta":
                _merge_delta(tool_states, event["tool_call"])
            elif kind == "message_end":
                stop_reason = str(event.get("stop_reason") or "end_turn")
            elif kind == "usage":
                total_usage = total_usage + Usage.from_mapping(event.get("usage") or {})
            elif kind == "error":
                raise event["error"]

        calls = [
            tool_states[idx]
            for idx in sorted(tool_states)
            if tool_states[idx]["function"]["name"]
        ]
        messages.append(
            Message(
                role="assistant",
                content=text,
                tool_calls=[] if stop_reason == "max_tokens" else calls,
                reasoning_content=reasoning or None,
            )
        )
        if text.strip():
            final_text = text

        if stop_reason == "max_tokens" or (stop_reason != "tool_use" and not calls):
            break

        for call in calls:
            tool_calls_count += 1
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"] or "{}"
            tool = registry.get(name)
            if tool is None:
                output = f'Tool "{name}" not found.'
            else:
                try:
                    parsed = json.loads(raw_args)
                    res = await tool.execute(parsed, ctx)
                    output = res.content
                except Exception as exc:  # noqa: BLE001
                    output = f'Tool "{name}" error: {exc}'
            if len(output) > 12_000:
                output = output[:12_000] + "\n...[truncated]"
            messages.append(
                Message(role="tool", content=output, tool_call_id=str(call.get("id") or ""))
            )

    costs = calculate_costs(llm_client, total_usage) or {}
    return {
        "text": final_text,
        "turns": turns,
        "tool_calls": tool_calls_count,
        "usage": total_usage,
        "cost": costs,
    }
