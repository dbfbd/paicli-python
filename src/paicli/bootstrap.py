"""工具装配入口：统一注册内置工具和已启用的 MCP 工具。"""

from __future__ import annotations

from paicli.config import PaiCliConfig
from paicli.mcp import McpClientManager
from paicli.tools import ToolRegistry, get_builtin_tools


async def build_tool_registry(
    *,
    config: PaiCliConfig,
    cwd: str,
) -> tuple[ToolRegistry, McpClientManager | None]:
    """创建工具注册表；启用 MCP 时连接外部服务并追加其真实工具定义。"""
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    manager: McpClientManager | None = None
    if config.features.mcp:
        manager = McpClientManager(cwd)
        registry.register_all(await manager.load_tools())
    return registry, manager
