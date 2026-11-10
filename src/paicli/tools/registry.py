"""以工具名管理本地工具对象，供模型说明导出和实际执行查找共用。"""

from __future__ import annotations

from paicli.tools.base import Tool


class ToolRegistry:
    """保存名称到工具对象的映射，并按执行权限生成受限注册表。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict]:
        """导出当前可见工具的参数说明，交给模型生成结构化调用请求。"""
        return [self._tools[name].definition() for name in self.list_names()]

    def read_only(self) -> ToolRegistry:
        """同时收窄模型可见工具和实际查找范围，供并行只读任务使用。"""
        registry = ToolRegistry()
        registry.register_all(
            [
                tool
                for tool in self._tools.values()
                if tool.is_read_only and tool.is_concurrency_safe
            ]
        )
        return registry
