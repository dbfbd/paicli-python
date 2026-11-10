"""工具协议：把模型可见的说明与本地执行函数、权限和结果分开定义。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from paicli.config import PaiCliConfig

DangerLevel = Literal["safe", "medium", "high"]
ToolDecision = Literal["approve", "deny", "skip"]


@dataclass(slots=True)
class ToolResult:
    """工具统一返回值，区分内容、错误、调用编号及原始证据引用。"""

    content: str
    is_error: bool = False
    display_summary: str | None = None
    tool_use_id: str | None = None
    evidence: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolContext:
    """本次工具执行共享的工作目录、配置、审批回调和 Skill 缓冲区。"""

    cwd: str
    config: PaiCliConfig
    approval_callback: Callable[[dict[str, Any]], Awaitable[ToolDecision] | ToolDecision] | None = (
        None
    )
    skill_context_buffer: Any | None = None


@dataclass(slots=True)
class Tool:
    """一项可调用工具，包含参数说明、本地处理函数、审批标记和超时限制。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]
    is_read_only: bool = True
    is_concurrency_safe: bool = True
    danger_level: DangerLevel = "safe"
    requires_approval: bool = False
    timeout: float | None = 60.0
    required_keys: list[str] = field(default_factory=list)

    def definition(self) -> dict[str, Any]:
        """只导出模型需要的名称、用途和参数结构，本地处理函数仍由执行器持有。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """检查输入对象和必需字段；具体参数约束由处理函数继续校验。"""
        if not isinstance(payload, dict):
            raise ValueError(f'tool "{self.name}" input must be an object')
        for key in self.required_keys:
            if key not in payload:
                raise ValueError(f'tool "{self.name}" missing required input: {key}')
        return payload

    async def execute(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        """验证输入后，在该工具的超时限制内等待本地处理函数完成。"""
        data = self.validate(payload)
        return await asyncio.wait_for(self.handler(data, context), timeout=self.timeout)


def object_schema(
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }
