"""终端 Agent 的会话入口。

维护消息历史和单次执行统计，按 react、plan、team 模式转交执行，
通过统一事件协议向命令行、交互终端和 SDK 返回进度。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any, Literal

from paicli.agent.orchestrator import AgentOrchestrator
from paicli.agent.plan_execute import PlanExecuteAgent
from paicli.agent.query import query as _query
from paicli.config import PaiCliConfig
from paicli.harness.evidence import recorded_run
from paicli.llm.base import LlmClient
from paicli.prompt import PromptAssembler
from paicli.skill import SkillContextBuffer
from paicli.snapshot import SnapshotService
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, QueryResult, Usage

AgentMode = Literal["react", "plan", "team"]


class Agent:
    """维护一个 Agent 会话，并连接模型、工具与运行配置。

    普通模式复用 query 循环；Plan 和 Team 分别交给各自的编排器。
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
        普通模式在执行前后创建快照；Plan 和 Team 在各自路径管理快照。
        """
        # Plan 和 Team 在各自路径管理快照；此处只为普通模式创建快照。
        if self.mode in {"plan", "team"}:
            runner = self._run_plan if self.mode == "plan" else self._run_team
            async for event in runner(message):
                yield event
            return

        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")

        try:
            async for event in self._run_react(message):
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
        """运行共享的模型与工具循环，并从最终事件保存会话历史和用量。"""
        async for event in _query(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            user_message=message,
            history=self.history,
            cwd=self.cwd,
            config=self.config,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
            max_turns=self.max_turns,
        ):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    async def _run_plan(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """交给 Plan 编排器生成并执行任务图，再接收其历史和统计。"""
        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            max_task_turns=self.max_turns,
        )
        agent.history = list(self.history)

        async for event in agent.run(message):
            if event.get("type") == "done":
                self.history = list(event.get("messages") or [])
                self.last_usage = Usage.from_mapping(event.get("usage") or {})
                self.last_cost = dict(event.get("cost") or {})
            yield event

    async def _run_team(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """交给多 Agent 编排器执行任务，再接收其历史和统计。"""
        orchestrator = AgentOrchestrator(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            default_worker_mode="react",
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
