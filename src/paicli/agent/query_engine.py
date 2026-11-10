"""统一查询接口：组装普通、计划、团队和专用调试入口。"""

from __future__ import annotations

import asyncio

from paicli.agent.agent import Agent
from paicli.agent.debug import DebugCoordinator
from paicli.agent.orchestrator import AgentOrchestrator
from paicli.agent.plan_execute import PlanExecuteAgent
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

    async def plan(self, message: str):
        """创建 Plan 执行器，透传规划、执行和最终结果事件。"""
        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
        )
        async for event in agent.run(message):
            yield event

    async def team(self, message: str, *, worker_mode: str = "react"):
        """创建通用多 Agent 编排器，并指定工作 Agent 的执行模式。"""
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode=worker_mode,
        )
        async for event in orchestrator.run(message):
            yield event

    async def ask_complete_async(
        self,
        message: str,
        history: list[Message] | None = None,
    ) -> QueryResult:
        return await self._complete_from_events(self.ask(message, history))

    async def debug(
        self,
        message: str,
        *,
        targets: list[str] | None = None,
        max_attempts: int = 2,
        timeout: float = 60,
        resume: str | None = None,
        max_tokens: int | None = None,
        max_tools: int | None = None,
        max_models: int | None = None,
        use_knowledge: bool = False,
    ):
        """创建专用 Debug 编排器，传入固定验收目标、共享预算与恢复选项。"""
        coordinator = DebugCoordinator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            targets=targets or [],
            max_attempts=max_attempts,
            timeout=timeout,
            max_tokens=max_tokens,
            max_tools=max_tools,
            max_models=max_models,
            use_knowledge=use_knowledge,
        )
        async for event in coordinator.run(message, resume=resume):
            yield event

    async def debug_complete_async(
        self, message: str, *, targets: list[str] | None = None, **options
    ) -> QueryResult:
        return await self._complete_from_events(self.debug(message, targets=targets, **options))

    async def plan_complete_async(self, message: str) -> QueryResult:
        return await self._complete_from_events(self.plan(message))

    async def team_complete_async(self, message: str, *, worker_mode: str = "react") -> QueryResult:
        return await self._complete_from_events(self.team(message, worker_mode=worker_mode))

    def ask_complete(self, message: str, history: list[Message] | None = None) -> QueryResult:
        return asyncio.run(self.ask_complete_async(message, history))

    def plan_complete(self, message: str) -> QueryResult:
        return asyncio.run(self.plan_complete_async(message))

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
                    raise RuntimeError(f"Query did not complete: {event['status']}")
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
