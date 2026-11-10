"""工具执行入口：调度调用、校验权限和输入，并保存原始执行证据。"""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any

from paicli.harness.contracts import active_policy
from paicli.harness.evidence import current_run, evidence_context
from paicli.harness.execution import active_control
from paicli.policy import AuditLog
from paicli.tools.base import Tool, ToolContext, ToolDecision, ToolResult
from paicli.tools.registry import ToolRegistry


class ToolExecutor:
    """连接结构化工具请求与本地处理函数，统一处理并发、审批和错误反馈。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute_all(
        self,
        calls: list[dict[str, Any]],
        context: ToolContext,
    ) -> list[ToolResult]:
        """并行执行标记为只读且并发安全的工具，其余调用在读取批次后串行执行。"""
        read_calls: list[tuple[dict[str, Any], Tool]] = []
        sequential_calls: list[tuple[dict[str, Any], Tool | None]] = []

        for call in calls:
            name = _tool_call_name(call)
            tool = self.registry.get(name)
            if tool and tool.is_read_only and tool.is_concurrency_safe:
                read_calls.append((call, tool))
            else:
                sequential_calls.append((call, tool))

        results: list[ToolResult] = []
        if read_calls:
            semaphore = asyncio.Semaphore(context.config.tools.max_concurrent_read)

            async def run_read(call: dict[str, Any], tool: Tool) -> ToolResult:
                async with semaphore:
                    return await self._execute_single(call, tool, context)

            results.extend(
                await asyncio.gather(*(run_read(call, tool) for call, tool in read_calls))
            )

        for call, tool in sequential_calls:
            results.append(await self._execute_single(call, tool, context))

        return results

    async def _execute_single(
        self,
        call: dict[str, Any],
        tool: Tool | None,
        context: ToolContext,
    ) -> ToolResult:
        """检查共享预算和任务契约，执行后保存原始输出，再生成有界的模型可见结果。"""
        run = current_run()
        name = _tool_call_name(call)
        call_id = str(call.get("id") or "")
        started = perf_counter()
        if run:
            artifact = run.save_text(
                "tool.input",
                json.dumps(_redact(_tool_call_arguments(call)), ensure_ascii=False, default=str),
            )
            run.record("tool.started", tool_name=name, tool_call_id=call_id, input=artifact)
        try:
            if control := active_control():
                control.tool()
            policy = active_policy()
            rejection = policy.check(name, _tool_call_arguments(call)) if policy else ""
            result = (
                ToolResult(rejection, is_error=True, tool_use_id=call_id)
                if rejection
                else await self._execute_unrecorded(call, tool, context)
            )
        except BaseException as exc:
            if run:
                run.record(
                    "tool.interrupted",
                    tool_name=name,
                    tool_call_id=call_id,
                    error_type=type(exc).__name__,
                    duration=perf_counter() - started,
                )
            raise
        # 先保存完整原始结果，再缩短交给模型的文本，保证长输出仍可追溯。
        if run:
            if name != "read_evidence" or result.is_error:
                result.evidence = run.save_text("tool.output", result.content)
            run.record(
                "tool.finished",
                tool_name=name,
                tool_call_id=call_id,
                is_error=result.is_error,
                output=result.evidence,
                duration=perf_counter() - started,
            )
            if name not in {"read_evidence", "load_skill"} and len(result.content) > 8000:
                result.content = evidence_context(result.content, result.evidence)
            if policy:
                policy.book.observe(name, _tool_call_arguments(call), result)
        return result

    async def _execute_unrecorded(
        self,
        call: dict[str, Any],
        tool: Tool | None,
        context: ToolContext,
    ) -> ToolResult:
        """查找并校验工具输入，经过审批后执行；普通工具异常转换为可反馈给模型的错误。"""
        tool_call_id = str(call.get("id") or "")
        name = _tool_call_name(call)
        payload = _tool_call_arguments(call)

        if not tool:
            return ToolResult(
                tool_use_id=tool_call_id,
                content=(
                    f'Tool "{name}" not found. Available tools: '
                    f"{', '.join(self.registry.list_names())}"
                ),
                is_error=True,
            )

        audit = AuditLog(context.config.policy.audit_log_path)
        approver = "none"
        try:
            if payload.get("__invalid_tool_json__") is True:
                raise ValueError(
                    "Malformed JSON arguments: resubmit concise valid JSON, not a raw string"
                )
            data = tool.validate(payload)
            decision = await self._approval_decision(tool, data, context)
            if decision in {"deny", "skip"}:
                approver = "hitl"
                audit.record(
                    tool_name=tool.name,
                    input_data=data,
                    outcome=decision,
                    approver=approver,
                    cwd=context.cwd,
                )
                return ToolResult(
                    tool_use_id=tool_call_id,
                    content=f'Tool "{tool.name}" was {decision}ed by approval policy.',
                    is_error=True,
                )
            if tool.requires_approval or context.config.policy.hitl_mode == "always":
                approver = "hitl"

            result = await tool.execute(data, context)
            result.tool_use_id = tool_call_id
            if not tool.is_read_only and context.config.features.audit_log:
                audit.record(
                    tool_name=tool.name,
                    input_data=data,
                    outcome="allow" if not result.is_error else "error",
                    approver=approver,
                    cwd=context.cwd,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - 工具错误需要反馈给模型继续判断。
            if context.config.features.audit_log and tool and not tool.is_read_only:
                audit.record(
                    tool_name=tool.name,
                    input_data=payload,
                    outcome="error",
                    approver=approver,
                    cwd=context.cwd,
                )
            return ToolResult(
                tool_use_id=tool_call_id,
                content=f'Tool "{name}" execution error: {exc}',
                is_error=True,
            )

    async def _approval_decision(
        self,
        tool: Tool,
        payload: dict[str, Any],
        context: ToolContext,
    ) -> ToolDecision:
        """综合当前策略、工具风险和审批回调，决定执行、拒绝或跳过。"""
        mode = context.config.policy.hitl_mode
        if mode == "never":
            return "approve"
        if mode == "auto" and not tool.requires_approval:
            return "approve"
        if not context.approval_callback:
            return "deny"
        result = context.approval_callback(
            {
                "tool_name": tool.name,
                "input": payload,
                "danger_level": tool.danger_level,
                "description": tool.description,
            }
        )
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "")


def _tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(arguments, str):
        import json

        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": arguments, "__invalid_tool_json__": True}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return arguments if isinstance(arguments, dict) else {}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        sensitive = (
            "api_key",
            "apikey",
            "password",
            "authorization",
            "secret",
            "token",
            "credential",
        )
        return {
            key: "[redacted]"
            if any(word in str(key).lower() for word in sensitive)
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
