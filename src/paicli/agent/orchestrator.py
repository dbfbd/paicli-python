"""隔离角色上下文的子 Agent 执行单元。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from paicli.agent.query import query
from paicli.config import PaiCliConfig
from paicli.llm.base import LlmClient
from paicli.skill import SkillContextBuffer
from paicli.tools.registry import ToolRegistry
from paicli.types import Message, Usage


class AgentRole(StrEnum):
    INVESTIGATOR = "INVESTIGATOR"
    REPAIRER = "REPAIRER"
    AUDITOR = "AUDITOR"


class AgentMessageType(StrEnum):
    TASK = "TASK"
    RESULT = "RESULT"
    FEEDBACK = "FEEDBACK"
    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    ERROR = "ERROR"


@dataclass(slots=True)
class AgentMessage:
    """角色之间的结构化任务或结果，携带消息类型、执行状态、用量和轮数。"""

    from_agent: str
    from_role: AgentRole | None
    content: str
    type: AgentMessageType
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    status: str = "completed"
    stop_reason: str = ""

    @classmethod
    def task(cls, from_agent: str, content: str) -> AgentMessage:
        return cls(from_agent, None, content, AgentMessageType.TASK)

    @classmethod
    def result(
        cls,
        from_agent: str,
        role: AgentRole,
        content: str,
        usage: Usage | None = None,
        turns: int = 0,
        status: str = "completed",
        stop_reason: str = "",
    ) -> AgentMessage:
        return cls(
            from_agent,
            role,
            content,
            AgentMessageType.RESULT,
            usage or Usage(),
            turns,
            status,
            stop_reason,
        )

    @classmethod
    def error(
        cls,
        from_agent: str,
        role: AgentRole,
        content: str,
        usage: Usage | None = None,
        turns: int = 0,
        stop_reason: str = "",
    ) -> AgentMessage:
        return cls(
            from_agent,
            role,
            content,
            AgentMessageType.ERROR,
            usage or Usage(),
            turns,
            "failed",
            stop_reason,
        )


class SubAgent:
    """隔离角色指令、消息历史和 Skill 缓冲区的执行单元。

    工作角色可通过工具完成任务，规划或通用评审角色可采用无工具调用。
    """

    def __init__(
        self,
        *,
        name: str,
        role: AgentRole,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback=None,
        skill_context_buffer: SkillContextBuffer | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        instructions: str = "",
        max_turns: int = 8,
        system_prompt: str | None = None,
    ):
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.skill_context_buffer = skill_context_buffer or SkillContextBuffer()
        self.event_callback = event_callback
        self.instructions = instructions
        self._prompt_override = system_prompt
        self.max_turns = max_turns
        self.history: list[Message] = []

    async def execute(
        self,
        task: AgentMessage,
        context: str = "",
    ) -> AgentMessage:
        """组合任务和上下文，根据角色使用工具循环或无工具调用。"""
        content = f"{context}\n\nCurrent task:\n{task.content}".strip() if context else task.content
        if self.role in {
            AgentRole.INVESTIGATOR,
            AgentRole.REPAIRER,
            AgentRole.AUDITOR,
        }:
            return await self._execute_worker(content)
        return await self._execute_without_tools(content)

    async def review(self, original_task: str, execution_result: str) -> AgentMessage:
        """将原任务和执行结果组装成审核输入，复用当前角色的执行入口。"""
        return await self.execute(
            AgentMessage.task(
                "orchestrator",
                f"Original task:\n{original_task}\n\nExecution result:\n{execution_result}",
            )
        )

    def clear_history(self) -> None:
        """同时清空角色消息与 Skill 缓冲区，避免后续任务继承旧上下文。"""
        self.history = []
        self.skill_context_buffer.clear()

    async def _execute_worker(self, content: str) -> AgentMessage:
        """复用 query 执行工具任务，转发过程事件，并保存该角色的消息历史和统计。"""
        text = ""
        tool_results: list[str] = []
        usage = Usage()
        turns = 0
        status = "completed"
        stop_reason = ""
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._system_prompt(),
                user_message=content,
                history=self.history,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=self.skill_context_buffer,
                max_turns=self.max_turns,
            ):
                self._forward_event(event)
                kind = event.get("type")
                if kind == "text_delta":
                    text += str(event.get("text") or "")
                elif kind == "tool_result":
                    tool_results.append(str(event.get("result") or ""))
                elif kind == "usage":
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                elif kind == "turn_complete":
                    turns = int(event.get("turn") or turns)
                elif kind == "done":
                    self.history = list(event.get("messages") or [])
                    usage = Usage.from_mapping(event.get("usage") or {})
                    turns = int(event.get("total_turns") or 0)
                    status = str(event.get("status") or "completed")
                    stop_reason = str(event.get("stop_reason") or "")
                elif kind == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return replace(
                AgentMessage.error(self.name, self.role, str(exc), usage, turns),
                status=getattr(exc, "status", "failed"),
            )
        result = text.strip() or "\n".join(item for item in tool_results if item).strip()
        return AgentMessage.result(self.name, self.role, result, usage, turns, status, stop_reason)

    async def _execute_without_tools(self, content: str) -> AgentMessage:
        """不提供工具定义，收集角色的文本结果和用量，供规划或通用评审使用。"""
        text = ""
        usage = Usage()
        stop_reason = "end_turn"
        messages = [*self.history, Message(role="user", content=content)]
        try:
            async for event in self.llm_client.chat(
                messages,
                [],
                system_prompt=self._system_prompt(),
            ):
                self._forward_event(event)
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "usage":
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                elif event.get("type") == "message_end":
                    stop_reason = str(event.get("stop_reason") or "end_turn")
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc), usage, 1)
        if stop_reason == "max_tokens":
            return AgentMessage.error(
                self.name, self.role, "Model output token budget exhausted", usage, 1, stop_reason
            )
        self.history = [*messages, Message(role="assistant", content=text)]
        return AgentMessage.result(self.name, self.role, text, usage, 1)

    def _forward_event(self, event: dict[str, Any]) -> None:
        if self.event_callback and event.get("type") in {
            "thinking_delta",
            "tool_call",
            "tool_result",
            "context_compressed",
            "usage",
        }:
            self.event_callback(event)

    def _system_prompt(self) -> str:
        if self._prompt_override is not None:
            return self._prompt_override
        role_prompt = {
            AgentRole.INVESTIGATOR: (
                "You investigate a specific debugging question using only read tools. "
                "Focus on assigned sources; follow dependencies when needed. Return file/line "
                "evidence, expected versus actual behavior, uncertainties and repair advice. "
                "Keep the investigation local to the assigned question. Finish as soon as "
                "you can answer it; do not map the whole application or duplicate other agents. "
                "Mark boundaries without existing tests as uncovered rather than searching "
                "indefinitely. Do not investigate interpreter versions or external libraries "
                "unless explicitly assigned. Submit the finding with finish_task when available. "
                "Do not claim edits or test execution."
            ),
            AgentRole.REPAIRER: (
                "You implement a focused Python bug fix using the supplied evidence. "
                "Use supplied source or read files before editing. Tests, config and dependencies "
                "are frozen. Do not add tests. Only Python application files can be edited. "
                "Use run_tests when provided; an independent verifier owns final acceptance. "
                "After the intended edits, use finish_task to hand off to the verifier. "
                "Report speculative compatibility concerns as uncertainties; do not investigate "
                "the environment or patch unrelated behavior. Do not claim a test pass."
            ),
            AgentRole.AUDITOR: (
                "You independently audit a Python repair using actual source and verification "
                "receipts. You have read-only tools. Separate observed behavior from inference. "
                "Do not treat the repairer's claims as evidence. Review scope expansion, changed "
                "imports, public behavior and error handling. Return only JSON: "
                '{"decision":"accept|needs_evidence|needs_patch|unverifiable",'
                '"checks":[{"requirement":"...","status":"met|violated|unverified",'
                '"ref":"full evidence reference"}],"issues":["..."],"request":"..."}. '
                "Only accept when the supplied evidence supports the task and frozen acceptance."
            ),
        }[self.role]
        return f"{role_prompt}\n{self.instructions}\nAgent name: {self.name}"
