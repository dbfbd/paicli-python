"""终端 Agent 的会话入口。

维护消息历史和单次执行统计，普通输入在同一循环里按需委派，
通过统一事件协议向命令行、交互终端和 SDK 返回进度。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from typing import Any, Literal

from paicli.agent.collaboration import Collaboration, LEAD_INSTRUCTIONS
from paicli.agent.query import query as _query
from paicli.agent.specialized import SpecializedWorkflows
from paicli.config import PaiCliConfig
from paicli.harness.evidence import recorded_run
from paicli.llm.base import LlmClient
from paicli.llm.pricing import calculate_costs
from paicli.prompt import PromptAssembler
from paicli.skill import SkillContextBuffer
from paicli.snapshot import SnapshotService
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, QueryResult, Usage

AgentMode = Literal["react", "team"]


class Agent:
    """维护一个 Agent 会话，并连接模型、工具与运行配置。

    普通输入复用 query 循环，由主 Agent 自行处理或按需路由。Team 可以显式启动。
    对象保留历史，调用方可复用实例继续对话。
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback: Callable | None = None,
        mode: AgentMode = "react",
        system_prompt: str | None = None,
        max_turns: int = 20,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        if mode not in {"react", "team"}:
            raise ValueError("mode must be react or team")
        self.mode = mode
        self.max_turns = max_turns

        # 组装稳定的系统提示；请求相关信息由执行循环动态补充。
        self.system_prompt = (
            system_prompt
            or PromptAssembler(
                config=config,
                cwd=cwd,
                tool_names=tool_registry.list_names(),
                model=llm_client.model_name,
                provider=llm_client.provider_name,
            ).build_static()
        )

        # 会话历史与当前角色独立的 Skill 缓冲区。
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()

        # 最近一次结束的执行所产生的用量和费用。
        self.last_usage = Usage()
        self.last_cost: dict[str, Any] = {}

        # 启动前尽早发现明显配置错误。
        self._validate_config()

    # ------------------------------------------------------------------
    # 公共执行接口
    # ------------------------------------------------------------------

    @recorded_run()
    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """按当前模式执行用户任务，持续产出进度事件。

        text_delta/thinking_delta 表示文字片段，tool_call/tool_result 表示工具过程，
        usage 表示本轮用量，done 汇总执行状态、历史与统计。
        普通模式在执行前后创建快照，Team 在自己的路径管理工作区。
        """
        if self.mode == "team":
            async for event in self._run_team(message):
                yield event
            return

        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")

        try:
            async with aclosing(self._run_react(message)) as stream:
                async for event in stream:
                    yield event
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")

    async def run_complete(self, message: str) -> QueryResult:
        """异步消费全部流式事件，汇总为 QueryResult；遇到失败状态时抛出异常。"""
        text = ""
        tokens = 0
        turns = 0
        usage = Usage()
        cost: dict[str, Any] = {}
        run_id = evidence_dir = None
        async for event in self.run(message):
            event_type = event.get("type")
            if event_type == "text_delta":
                text += str(event.get("text") or "")
            elif event_type == "error":
                raise event["error"]  # type: ignore[arg-type]
            elif event_type == "done":
                if event.get("status", "completed") != "completed":
                    raise RuntimeError(f"Agent did not complete: {event['status']}")
                tokens = int(event.get("total_tokens") or 0)
                turns = int(event.get("total_turns") or 0)
                usage = Usage.from_mapping(event.get("usage") or {})
                cost = dict(event.get("cost") or {})
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

    # ------------------------------------------------------------------
    # 会话历史管理
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """清空会话消息、Skill 缓冲区及最近执行统计，使下一次任务从新上下文开始。"""
        self.history = []
        self.skill_context_buffer.clear()
        self.last_usage = Usage()
        self.last_cost = {}

    # ------------------------------------------------------------------
    # 按模式转交执行
    # ------------------------------------------------------------------

    async def _run_react(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """同一会话负责用户任务，并在有委派时把 Worker 用量并入最终事件。"""
        collaboration = Collaboration(
            llm_client=self.llm_client, registry=self.tool_registry,
            config=self.config, cwd=self.cwd, approval_callback=self.approval_callback,
        )
        workflows = SpecializedWorkflows(self.llm_client, collaboration)
        routing = workflows.routing_context(message, self.cwd)
        registry = collaboration.registry()
        workflows.register(registry, auto_plan=self.config.plan.auto_route)
        stream = _query(
            llm_client=self.llm_client, tool_registry=registry,
            system_prompt=f"{self.system_prompt}\n\n{LEAD_INSTRUCTIONS}{routing}",
            user_message=message, history=self.history, cwd=self.cwd,
            config=self.config, approval_callback=collaboration.approval,
            skill_context_buffer=self.skill_context_buffer, max_turns=self.max_turns,
            before_finish=collaboration.pending_instruction,
        )
        try:
            async with aclosing(stream):
                async for event in stream:
                    if event.get("type") == "done":
                        await collaboration.close()
                        self.history = list(event.get("messages") or [])
                        self.last_usage = (
                            Usage.from_mapping(event.get("usage") or {})
                            + collaboration.budget.usage
                            + workflows.usage
                        )
                        self.last_cost = calculate_costs(self.llm_client, self.last_usage) or {}
                        event = {
                            **event, "usage": self.last_usage.to_dict(), "cost": self.last_cost,
                            "total_tokens": self.last_usage.total_tokens,
                            "total_turns": (
                                event["total_turns"] + collaboration.budget.calls + workflows.calls
                            ),
                        }
                    yield event
        finally:
            await collaboration.close()

    async def _run_team(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """交给多 Agent 编排器执行任务，再接收其历史和统计。"""
        from paicli.agent.team import TeamCoordinator

        orchestrator = TeamCoordinator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
        )
        async for event in orchestrator.run(message):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    # ------------------------------------------------------------------
    # 配置校验
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """提前拒绝缺失模型或必需凭据等明显配置错误。"""
        if not self.config.llm.api_key:
            raise ValueError(
                "LLM API key is not configured. "
                "Set PAICLI_API_KEY (or DEEPSEEK_API_KEY / GLM_API_KEY / etc.) "
                "in the environment or in ~/.paicli/config.json."
            )
        if not self.llm_client.max_context_window:
            raise ValueError(
                "LLM context window is not configured. "
                "Set a PAICLI_CONTEXT_WINDOW environment variable or configure "
                "it in ~/.paicli/config.json."
            )
