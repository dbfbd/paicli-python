"""统一查询接口：组装普通会话和 Team 入口。"""

from __future__ import annotations

import asyncio

from paicli.agent.agent import Agent
from paicli.agent.team import TeamCoordinator
from paicli.config import PaiCliConfig
from paicli.llm.base import LlmClient
from paicli.prompt import PromptAssembler
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, QueryResult, Usage


class QueryEngine:
    """持有共享配置、模型客户端和工具注册表，按请求创建对应执行器。"""

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback=None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.system_prompt = PromptAssembler(
            config=config,
            cwd=cwd,
            tool_names=tool_registry.list_names(),
            model=llm_client.model_name,
            provider=llm_client.provider_name,
        ).build_static()

    async def ask(self, message: str, history: list[Message] | None = None):
        """创建普通 Agent 并传入调用方历史；跨请求对话由调用方保存和回传。"""
        agent = Agent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
        )
        agent.history = list(history or [])
        async for event in agent.run(message):
            yield event

    async def team(self, message: str, *, worker_mode: str = "react"):
        """运行协作 Team，Worker 通过各自的工具循环执行。"""
        if worker_mode != "react":
            raise ValueError("Team supports only react workers")
        orchestrator = TeamCoordinator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
        )
        async for event in orchestrator.run(message):
            yield event

    async def ask_complete_async(
        self,
        message: str,
        history: list[Message] | None = None,
    ) -> QueryResult:
        return await self._complete_from_events(self.ask(message, history))

    async def team_complete_async(self, message: str, *, worker_mode: str = "react") -> QueryResult:
        return await self._complete_from_events(self.team(message, worker_mode=worker_mode))

    def ask_complete(self, message: str, history: list[Message] | None = None) -> QueryResult:
        return asyncio.run(self.ask_complete_async(message, history))

    def team_complete(self, message: str, *, worker_mode: str = "react") -> QueryResult:
        return asyncio.run(self.team_complete_async(message, worker_mode=worker_mode))

    async def _complete_from_events(self, events) -> QueryResult:
        """将流式事件收集为查询结果，保留用量和证据目录，并拒绝非完成状态。"""
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        cost = {}
        run_id = evidence_dir = None
        async for event in events:
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
            elif event.get("type") == "done":
                if event.get("status", "completed") != "completed":
                    reason = event.get("reason")
                    detail = f"\n{reason}" if reason else ""
                    raise RuntimeError(f"Query did not complete: {event['status']}{detail}")
                tokens += int(event.get("total_tokens") or 0)
                turns += int(event.get("total_turns") or 0)
                usage = usage + Usage.from_mapping(event.get("usage") or {})
                cost = dict(event.get("cost") or cost)
                run_id, evidence_dir = event.get("run_id"), event.get("evidence_dir")
        return QueryResult(
            text=text,
            total_tokens=tokens,
            turns=turns,
            usage=usage,
            cost=cost,
            run_id=run_id,
            evidence_dir=evidence_dir,
        )
