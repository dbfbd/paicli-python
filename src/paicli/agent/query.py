"""共享执行循环：组装模型输入、收集工具请求、执行并回填结果。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from paicli.config import PaiCliConfig
from paicli.context import ContextBudget, ContextWindowManager
from paicli.harness.contracts import active_policy
from paicli.harness.execution import RunStopped
from paicli.image import parse_image_references
from paicli.llm.base import LlmClient
from paicli.llm.pricing import calculate_costs
from paicli.prompt import PromptAssembler
from paicli.skill import SkillRegistry
from paicli.tools.base import ToolContext
from paicli.tools.executor import ToolExecutor
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, Usage


async def query(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: PaiCliConfig,
    approval_callback=None,
    skill_context_buffer=None,
    max_turns: int = 20,
) -> AsyncIterator[dict[str, Any]]:
    """在限定轮数内交替调用模型和工具，向外产出统一事件。

    messages 保存当前上下文；工具结果通过 tool_call_id 对应回原请求。
    遇到截断、预算耗尽或取消时停止派发；专员提交完成后交还外层编排。
    此处完成表示当前执行循环结束，Debug 的验收由外层另行判定。
    """
    original_user_message = user_message
    user_message = _prepend_skill_candidates(user_message, cwd, config)
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    messages = [
        *(history or []),
        Message(role="user", content=parse_image_references(user_message, cwd)),
    ]
    # 模型只接收工具说明；对应的本地处理函数留在注册表中。
    tool_definitions = tool_registry.definitions()
    executor = ToolExecutor(tool_registry)
    context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
    )
    dynamic_prompt = PromptAssembler(
        config=config,
        cwd=cwd,
        tool_names=tool_registry.list_names(),
        model=llm_client.model_name,
        provider=llm_client.provider_name,
    ).build_dynamic(original_user_message)
    effective_system_prompt = f"{system_prompt}\n\n{dynamic_prompt}".strip()
    context_manager = ContextWindowManager(
        ContextBudget(
            context_window=llm_client.max_context_window,
            max_output_tokens=config.llm.max_tokens,
            compression_threshold=config.memory.compression_threshold,
            compression_target=config.memory.compression_target,
            reserve_tokens=config.memory.compression_reserve_tokens,
        ),
        max_history_messages=config.memory.max_conversation_history,
        min_recent_messages=config.memory.min_recent_messages,
        summary_max_chars=config.memory.summary_max_chars,
    )

    total_usage = Usage()
    turn = 0
    completed = False
    execution_status = ""
    stop_reason = "turn_limit"

    # 一个任务可能包含多次模型调用；每轮把工具观察继续放回消息历史。
    while turn < max_turns:
        turn += 1
        text = ""
        reasoning = ""
        stop_reason = "end_turn"
        turn_usage = Usage()
        tool_states: dict[int, dict[str, Any]] = {}

        if config.features.context_compression:
            compression = context_manager.prepare(
                messages,
                system_prompt=effective_system_prompt,
                tool_definitions=tool_definitions,
            )
            messages = compression.messages
            if compression.compressed:
                yield {
                    "type": "context_compressed",
                    "before_tokens": compression.estimated_tokens_before,
                    "after_tokens": compression.estimated_tokens_after,
                    "summarized_messages": compression.summarized_messages,
                }

        try:
            async for event in llm_client.chat(
                messages,
                tool_definitions,
                system_prompt=effective_system_prompt,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    delta = str(event.get("text") or "")
                    text += delta
                    yield {"type": "text_delta", "text": delta}
                elif event_type == "thinking_delta":
                    delta = str(event.get("thinking") or "")
                    reasoning += delta
                    yield {"type": "thinking_delta", "thinking": delta}
                elif event_type == "tool_call_delta":
                    _merge_tool_delta(tool_states, event["tool_call"])
                elif event_type == "message_end":
                    stop_reason = str(event.get("stop_reason") or "end_turn")
                elif event_type == "usage":
                    usage = Usage.from_mapping(event.get("usage") or {})
                    turn_usage = turn_usage + usage
                    yield {"type": "usage", "usage": usage.to_dict()}
                elif event_type == "error":
                    yield {"type": "error", "error": event["error"]}
                    return
        except RunStopped as exc:
            execution_status, stop_reason = exc.status, "execution_stopped"
            total_usage += turn_usage
            if text or reasoning:
                messages.append(
                    Message(role="assistant", content=text, reasoning_content=reasoning or None)
                )
            yield {"type": "execution_stopped", "status": exc.status, "reason": str(exc)}
            break

        total_usage = total_usage + turn_usage
        # 等完整响应收齐再处理工具；输出截断时不执行可能残缺的参数。
        tool_calls = _finalize_tool_calls(tool_states)
        messages.append(
            Message(
                role="assistant",
                content=text,
                tool_calls=[] if stop_reason == "max_tokens" else tool_calls,
                reasoning_content=reasoning or None,
            )
        )
        yield {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}

        if stop_reason == "max_tokens":
            break
        if stop_reason != "tool_use" and not tool_calls:
            completed = True
            break

        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            yield {
                "type": "tool_call",
                "name": name,
                "input": _tool_input(call),
                "tool_call_id": str(call.get("id") or ""),
            }

        try:
            tool_results = await executor.execute_all(tool_calls, context)
        except RunStopped as exc:
            execution_status, stop_reason = exc.status, "execution_stopped"
            messages.extend(
                Message(
                    role="tool",
                    tool_call_id=str(call.get("id") or ""),
                    content="Batch stopped; earlier calls may have completed. "
                    "Inspect current files before continuing.",
                )
                for call in tool_calls
            )
            yield {"type": "execution_stopped", "status": exc.status, "reason": str(exc)}
            break
        loaded_skill_context = _drain_skill_context(skill_context_buffer)
        load_skill_ids = [
            str(call.get("id") or "")
            for call in tool_calls
            if str(call.get("function", {}).get("name") or "") == "load_skill"
        ]
        injection_target = load_skill_ids[-1] if load_skill_ids else ""
        injected = False
        # 用调用编号配对工具结果，下一次模型请求便能看到真实执行反馈。
        for result in tool_results:
            yield {
                "type": "tool_result",
                "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
                "result": result.content,
                "is_error": result.is_error,
                "tool_call_id": result.tool_use_id,
                "evidence": result.evidence,
            }
            model_content = result.content
            if loaded_skill_context and result.tool_use_id == injection_target:
                model_content = f"{model_content}\n\n{loaded_skill_context}"
                injected = True
            messages.append(
                Message(
                    role="tool",
                    content=model_content,
                    tool_call_id=result.tool_use_id,
                )
            )
        if loaded_skill_context and not injected:
            messages.append(
                Message(
                    role="user",
                    content=(
                        f"{loaded_skill_context}\n\n"
                        "Use these loaded instructions to continue the current request."
                    ),
                )
            )

        if (policy := active_policy()) and policy.submission is not None:
            completed = True
            break
        if policy and policy.checkpoint():
            completed = True  # 结束当前专员执行，真实验收仍由外层负责。
            break
        if policy and policy.halt_reason:
            break
        if policy:
            tool_definitions = [
                definition
                for definition in tool_definitions
                if policy.allows(definition.get("function", {}).get("name", ""))
            ]

    done_event: dict[str, Any] = {
        "type": "done",
        "status": "completed" if completed else execution_status or "budget_exhausted",
        "stop_reason": stop_reason,
        "total_turns": turn,
        "total_tokens": total_usage.total_tokens,
        "usage": total_usage.to_dict(),
        "messages": messages,
    }
    costs = calculate_costs(llm_client, total_usage)
    if costs:
        done_event["cost"] = costs
    yield done_event


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    """按工具索引登记调用编号和名称，并拼接流式参数片段。"""
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """把已收齐的工具片段转换成完整调用，供执行器统一处理。"""
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_context(user_message: str, skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"


def _drain_skill_context(skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return ""
    return skill_context_buffer.drain()


def _prepend_skill_candidates(user_message: str, cwd: str, config: PaiCliConfig) -> str:
    if not config.features.skill:
        return user_message
    candidates = SkillRegistry(cwd).match(user_message, top_k=5)
    if not candidates:
        return user_message
    lines = [
        "Relevant skill candidates for this request:",
        "Call load_skill(name) before proceeding when a candidate applies.",
    ]
    for skill in candidates:
        description = " ".join(skill.description.split())
        if len(description) > 300:
            description = description[:297] + "..."
        tags = f" [tags: {', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(f"- {skill.name}: {description}{tags}")
    candidate_text = "\n".join(lines)
    return f"{candidate_text}\n\n---\nUser request:\n{user_message}"
