from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, JSONRPCMessage
from pydantic import AnyUrl, ValidationError

from paicli.mcp.config import McpServerSpec, load_mcp_server_specs
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema


class _InvalidArguments(ValueError):
    """本地 MCP 工具在连接服务前发现的参数错误。"""


class McpClientManager:
    def __init__(self, project_root: str | Path):
        self.project_root = str(Path(project_root).resolve())
        self.specs = load_mcp_server_specs(self.project_root)
        self.last_errors: dict[str, str] = {}

    async def load_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        self.last_errors.clear()
        for spec in self.specs.values():
            if not spec.enabled:
                continue
            try:
                server_tools = await self._tools_for_server(spec)
                server_tools.extend(self._virtual_resource_tools(spec))
                server_tools.extend(self._virtual_prompt_tools(spec))
                for tool in server_tools:
                    tool.handler = _with_error_feedback(tool.handler, read_only=tool.is_read_only)
                    # MCP 会话负责总超时，避免外层默认超时先取消会话并丢掉错误类型。
                    tool.timeout = None
                tools.extend(server_tools)
            except Exception as exc:  # noqa: BLE001 - 隔离异常 MCP 服务。
                kind, detail = _request_failure(exc)
                if kind == "timeout":
                    detail = f"MCP startup or tool discovery timed out (limit {spec.timeout:g}s)."
                self.last_errors[spec.name] = detail
        return tools

    async def list_server_tools(self, spec: McpServerSpec) -> list[Any]:
        async with self._session(spec) as session:
            result = await session.list_tools()
            return list(result.tools)

    async def call_server_tool(
        self,
        spec: McpServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.call_tool(tool_name, arguments)
        return ToolResult(content=_content_to_text(result.content), is_error=bool(result.isError))

    async def list_resources(self, spec: McpServerSpec) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.list_resources()
        lines = [
            f"{resource.uri} {resource.name or ''} {resource.description or ''}".strip()
            for resource in result.resources
        ]
        return ToolResult("\n".join(lines) or "(no resources)")

    async def read_resource(self, spec: McpServerSpec, uri: str) -> ToolResult:
        try:
            resource_uri = AnyUrl(uri)
        except ValidationError:
            raise _InvalidArguments("A valid MCP resource URI is required.") from None
        async with self._session(spec) as session:
            result = await session.read_resource(resource_uri)
        return ToolResult(_content_to_text(result.contents))

    async def list_prompts(self, spec: McpServerSpec) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.list_prompts()
        lines = [f"{prompt.name} {prompt.description or ''}".strip() for prompt in result.prompts]
        return ToolResult("\n".join(lines) or "(no prompts)")

    async def get_prompt(
        self,
        spec: McpServerSpec,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> ToolResult:
        async with self._session(spec) as session:
            result = await session.get_prompt(name, arguments or {})
        return ToolResult(_content_to_text(result.messages))

    async def _tools_for_server(self, spec: McpServerSpec) -> list[Tool]:
        remote_tools = await self.list_server_tools(spec)
        wrapped: list[Tool] = []
        for remote_tool in remote_tools:
            tool_name = str(remote_tool.name)
            local_name = f"mcp__{spec.name}__{tool_name}"
            schema = remote_tool.inputSchema or object_schema({})
            annotations = getattr(remote_tool, "annotations", None)
            read_only = bool(getattr(annotations, "readOnlyHint", False))

            async def handler(
                payload: dict[str, Any],
                context: ToolContext,
                *,
                server_spec: McpServerSpec = spec,
                remote_name: str = tool_name,
            ) -> ToolResult:
                _ = context
                return await self.call_server_tool(server_spec, remote_name, payload)

            wrapped.append(
                Tool(
                    name=local_name,
                    description=remote_tool.description or f"MCP tool {tool_name}",
                    parameters=schema,
                    handler=handler,
                    is_read_only=read_only,
                    is_concurrency_safe=False,
                    danger_level="safe" if read_only else "medium",
                    requires_approval=not read_only,
                )
            )
        return wrapped

    def _virtual_resource_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_resources(spec)

        async def read_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            return await self.read_resource(spec, str(payload["uri"]))

        return [
            Tool(
                name=f"mcp__{spec.name}__list_resources",
                description=f"List MCP resources from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
            ),
            Tool(
                name=f"mcp__{spec.name}__read_resource",
                description=f"Read an MCP resource from {spec.name}.",
                parameters=object_schema(
                    {"uri": {"type": "string", "description": "Resource URI"}},
                    ["uri"],
                ),
                required_keys=["uri"],
                handler=read_handler,
                is_read_only=True,
            ),
        ]

    def _virtual_prompt_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_prompts(spec)

        async def get_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            arguments = payload.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise _InvalidArguments("arguments must be an object")
            return await self.get_prompt(
                spec,
                str(payload["name"]),
                {str(k): str(v) for k, v in (arguments or {}).items()},
            )

        return [
            Tool(
                name=f"mcp__{spec.name}__list_prompts",
                description=f"List MCP prompts from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
            ),
            Tool(
                name=f"mcp__{spec.name}__get_prompt",
                description=f"Get an MCP prompt from {spec.name}.",
                parameters=object_schema(
                    {
                        "name": {"type": "string", "description": "Prompt name"},
                        "arguments": {"type": "object", "description": "Prompt arguments"},
                    },
                    ["name"],
                ),
                required_keys=["name"],
                handler=get_handler,
                is_read_only=True,
            ),
        ]

    @asynccontextmanager
    async def _session(self, spec: McpServerSpec):
        """将连接、握手和后续请求纳入同一个服务超时。"""
        async with asyncio.timeout(spec.timeout), self._connect(spec) as session:
            await session.initialize()
            yield session

    @asynccontextmanager
    async def _connect(self, spec: McpServerSpec):
        if spec.type in {"stdio", "local"}:
            if not spec.command:
                raise ValueError(f"MCP server {spec.name} is missing command")
            params = StdioServerParameters(
                command=spec.command,
                args=spec.args,
                env={**os.environ, **spec.env},
                cwd=spec.cwd or self.project_root,
            )
            with open(os.devnull, "w", encoding="utf-8") as errlog:
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    yield session
            return
        if spec.type in {"http", "streamable_http", "streamable-http"}:
            if not spec.url:
                raise ValueError(f"MCP server {spec.name} is missing url")
            async with (
                httpx.AsyncClient(
                    headers=spec.headers or None,
                    timeout=httpx.Timeout(spec.timeout, read=spec.timeout),
                    follow_redirects=True,
                    event_hooks={"response": [_validate_http_response]},
                ) as http_client,
                streamable_http_client(
                    spec.url,
                    http_client=http_client,
                ) as (read, write, _session_id),
                ClientSession(read, write) as session,
            ):
                yield session
            return
        raise ValueError(f"Unsupported MCP transport: {spec.type}")


async def _validate_http_response(response: httpx.Response) -> None:
    """在 SDK 记录原始响应前拒绝非 MCP 数据，防止异常页面内容进入回溯。"""
    if response.request.method != "POST" or not response.is_success:
        return
    if not response.headers.get("content-type", "").lower().startswith("application/json"):
        return
    content = await response.aread()
    # 通知请求通常返回没有正文的 202 响应。
    if not content and response.status_code in {202, 204}:
        return
    try:
        JSONRPCMessage.model_validate_json(content)
    except ValidationError:
        detail = ""
        try:
            payload = json.loads(content)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("code"), int):
            detail = f" (service code {payload['code']})"
        raise ValueError(
            f"Server returned invalid MCP JSON-RPC{detail}; check the MCP URL and credentials."
        ) from None


def _with_error_feedback(
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]],
    *,
    read_only: bool,
) -> Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]:
    async def execute(payload: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            result = await handler(payload, context)
        except Exception as exc:  # noqa: BLE001 - 将单个 MCP 调用错误反馈给模型。
            kind, detail = _request_failure(exc)
        else:
            if not result.is_error:
                return result
            kind, detail = "tool_error", result.content or "The MCP tool reported a failure."

        if kind == "invalid_arguments":
            guidance = "Check the tool name and correct the arguments before retrying."
        elif read_only:
            guidance = (
                "Correct any input error reported by the tool. For a failed read, use an available "
                "built-in tool only if it can read the same source. Otherwise report the MCP step "
                "as unverified."
            )
        else:
            guidance = "Verify whether the operation took effect before attempting it again."
        return ToolResult(f"MCP {kind}: {detail}\n{guidance}", is_error=True)

    return execute


def _request_failure(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, BaseExceptionGroup):
        failures = [_request_failure(child) for child in exc.exceptions]
        for kind, detail in failures:
            if kind != "unavailable":
                return kind, detail
        return "unavailable", "; ".join(dict.fromkeys(detail for _, detail in failures))
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "timeout", "The MCP request timed out."
    if isinstance(exc, _InvalidArguments):
        return "invalid_arguments", str(exc)
    if isinstance(exc, McpError):
        if exc.error.code == INVALID_PARAMS:
            return "invalid_arguments", str(exc)
        if exc.error.code == httpx.codes.REQUEST_TIMEOUT:
            return "timeout", "The MCP request timed out."
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return (
                "access_denied",
                f"The MCP server denied access (HTTP {status}). Check credentials or permissions.",
            )
        if status in {408, 504}:
            return "timeout", f"The MCP server timed out (HTTP {status})."
        return "unavailable", f"The MCP server returned HTTP {status}."
    if isinstance(exc, httpx.RequestError):
        return "unavailable", f"The MCP connection failed ({type(exc).__name__})."
    if isinstance(exc, ValidationError):
        return "unavailable", "The MCP response did not match the expected format."
    return "unavailable", str(exc) or type(exc).__name__


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_to_text(item) for item in content)))
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "data") and hasattr(content, "mimeType"):
        data = str(content.data)
        return f"[image {content.mimeType} base64 chars={len(data)}]"
    if hasattr(content, "resource"):
        return _content_to_text(content.resource)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
    return str(content)
