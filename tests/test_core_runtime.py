"""核心运行时回归测试：执行循环、缓存前缀稳定、上下文压缩、快照与审批保护。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from paicli.agent.agent import Agent
from paicli.agent.query import query
from paicli.agent.team import TeamCoordinator, _TeamBudget, _TeamModel
from paicli.agent.team_checks import _checked_argv
from paicli.config import PaiCliConfig
from paicli.context import ContextBudget, ContextWindowManager
from paicli.harness.contracts import EvidenceBook, TaskSpec, task_contract
from paicli.harness.evidence import _CURRENT, EvidenceRun
from paicli.snapshot import SnapshotService
from paicli.tools.base import Tool, ToolResult
from paicli.tools.builtins import get_builtin_tools
from paicli.tools.registry import ToolRegistry
from paicli.types import Message


class _RecordingMockClient:
    provider_name = "mock"
    model_name = "mock-model"
    max_context_window = 32_000
    price_profile = None

    def __init__(self, scripted_turns: list[list[dict]]):
        self.scripted = list(scripted_turns)
        self.seen_tools: list[list[str]] = []
        self.seen_system_prompts: list[str] = []
        self.seen_messages: list[list[Message]] = []

    async def chat(self, messages, tools, *, system_prompt):
        self.seen_messages.append(list(messages))
        self.seen_system_prompts.append(system_prompt)
        self.seen_tools.append([t["function"]["name"] for t in tools])
        turn_events = (
            self.scripted.pop(0)
            if self.scripted
            else [
                {"type": "text_delta", "text": "done"},
                {"type": "message_end", "stop_reason": "end_turn"},
            ]
        )
        for event in turn_events:
            yield event


def test_query_hides_exhausted_read_tools_on_next_turn(tmp_path: Path):
    """TaskPolicy.max_read_calls 耗尽后，下一轮模型请求不再暴露超配额只读工具。"""
    sample = tmp_path / "a.py"
    sample.write_text("x = 1\n", encoding="utf-8")

    async def dummy_read(_data, _ctx):
        return ToolResult("x = 1")

    async def dummy_finish(_data, _ctx):
        return ToolResult("submitted")

    reg = ToolRegistry()
    reg.register(
        Tool(
            "read_file",
            "read",
            {"type": "object", "properties": {"path": {"type": "string"}}},
            dummy_read,
        )
    )
    reg.register(
        Tool(
            "finish_task",
            "finish",
            {"type": "object", "properties": {}},
            dummy_finish,
            is_concurrency_safe=False,
        )
    )

    client = _RecordingMockClient(
        [
            [
                {
                    "type": "tool_call_delta",
                    "tool_call": {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    },
                },
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            [
                {"type": "text_delta", "text": "finished"},
                {"type": "message_end", "stop_reason": "end_turn"},
            ],
        ]
    )

    cfg = PaiCliConfig()
    cfg.features.skill = False
    cfg.features.memory = False

    async def _run():
        run = EvidenceRun.create(str(tmp_path), "debug")
        token = _CURRENT.set(run)
        try:
            book = EvidenceBook(str(tmp_path))
            spec = TaskSpec("t1", "root_cause", "check a.py", ["."], max_read_calls=1)
            with task_contract(spec, book):
                async for _ in query(
                    llm_client=client,
                    tool_registry=reg,
                    system_prompt="sys",
                    user_message="inspect",
                    history=[],
                    cwd=str(tmp_path),
                    config=cfg,
                    max_turns=4,
                ):
                    pass
        finally:
            _CURRENT.reset(token)
            run.finish("completed")

    asyncio.run(_run())
    assert len(client.seen_tools) == 2
    assert "read_file" in client.seen_tools[0]
    assert "read_file" not in client.seen_tools[1]


def test_team_check_resolves_target_project_venv_python(tmp_path: Path):
    """目标项目存在 .venv 时，team_checks 的 python 解析到目标项目的 .venv。"""
    venv_py = (
        tmp_path / ".venv" / "Scripts" / "python.exe"
        if Path("C:/").exists()
        else tmp_path / ".venv" / "bin" / "python"
    )
    venv_py.parent.mkdir(parents=True, exist_ok=True)
    venv_py.write_bytes(b"")

    resolved = _checked_argv(("python", "-m", "pytest"), tmp_path, PaiCliConfig())
    assert Path(resolved[0]).resolve() == venv_py.resolve()


def test_query_recovers_from_max_tokens_truncation(tmp_path: Path):
    """单轮触发 max_tokens 截断时不直接崩溃，而是引导拆步继续完成。"""
    client = _RecordingMockClient(
        [
            [
                {"type": "text_delta", "text": "trying to write 5 files at once..."},
                {"type": "message_end", "stop_reason": "max_tokens"},
            ],
            [
                {"type": "text_delta", "text": "recovered cleanly"},
                {"type": "message_end", "stop_reason": "end_turn"},
            ],
        ]
    )
    cfg = PaiCliConfig()
    cfg.features.skill = False
    cfg.features.memory = False

    async def _run():
        done = {}
        async for ev in query(
            llm_client=client,
            tool_registry=ToolRegistry(),
            system_prompt="sys",
            user_message="do big work",
            history=[],
            cwd=str(tmp_path),
            config=cfg,
            max_turns=4,
        ):
            if ev["type"] == "done":
                done = ev
        return done

    done = asyncio.run(_run())
    assert done["status"] == "completed"
    assert done["total_turns"] == 2
    assert any("max_tokens" in str(m.content) for m in client.seen_messages[1])


def test_system_prompt_is_static_without_timestamp(tmp_path: Path):
    """跨轮对话 system_prompt 保持静态，不含每秒变化的 Current time。"""
    client = _RecordingMockClient([[], []])
    cfg = PaiCliConfig()
    cfg.features.skill = False
    cfg.features.memory = False

    async def _run():
        for msg in ("turn 1", "turn 2"):
            async for _ in query(
                llm_client=client,
                tool_registry=ToolRegistry(),
                system_prompt="BASE_STATIC",
                user_message=msg,
                history=[],
                cwd=str(tmp_path),
                config=cfg,
                max_turns=2,
            ):
                pass

    asyncio.run(_run())
    assert client.seen_system_prompts[0] == client.seen_system_prompts[1]
    assert "Current time:" not in client.seen_system_prompts[0]
    assert "Current time:" in str(client.seen_messages[0][-1].content)


def test_tool_registry_deduplicates_aliases_while_resolving_get():
    """重复工具别名不再占 Schema token，但 registry.get() 仍兼容解析别名。"""
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    exported = {d["function"]["name"] for d in reg.definitions()}
    assert "glob" in exported and "glob_files" not in exported
    assert "grep" in exported and "grep_code" not in exported
    assert "execute_command" in exported and "bash" not in exported
    assert reg.get("glob_files") is reg.get("glob")
    assert reg.get("grep_code") is reg.get("grep")
    assert reg.get("bash") is reg.get("execute_command")


def test_context_manager_uses_user_summary_and_respects_input_cap():
    """摘要作为 user 角色插入，且支持超大上下文窗口的绝对压缩软顶。"""
    budget = ContextBudget(
        context_window=1_000_000,
        max_output_tokens=16_384,
        max_input_tokens_cap=128_000,
    )
    assert budget.available_input_tokens == 128_000

    small_budget = ContextBudget(context_window=1200, max_output_tokens=200, reserve_tokens=100)
    mgr = ContextWindowManager(small_budget, min_recent_messages=2)
    history = [
        Message(role="user", content="initial goal " + ("x" * 1200)),
        Message(role="assistant", content="step 1 " + ("y" * 1200)),
        Message(role="user", content="follow up"),
        Message(role="assistant", content="latest answer"),
    ]
    res = mgr.prepare(history)
    assert res.compressed is True
    assert res.messages[0].role == "user"
    assert "<conversation-summary" in str(res.messages[0].content)


def test_lazy_snapshot_skips_read_only_turn_and_prunes_limit(tmp_path: Path):
    """纯只读对话不创建整仓快照，且 SnapshotService 自动按上限裁剪。"""
    (tmp_path / "hello.py").write_text("x = 1\n", encoding="utf-8")
    svc = SnapshotService(tmp_path, max_snapshots=2)
    svc.clean()

    # 纯只读 Agent 会话不应产生任何快照。
    cfg = PaiCliConfig()
    cfg.llm.api_key = "test-key"
    cfg.features.skill = False
    cfg.features.memory = False
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    client = _RecordingMockClient(
        [[{"type": "text_delta", "text": "hi"}, {"type": "message_end", "stop_reason": "end_turn"}]]
    )
    agent = Agent(llm_client=client, tool_registry=reg, config=cfg, cwd=str(tmp_path))

    async def _run():
        async for _ in agent.run("explain"):
            pass

    asyncio.run(_run())
    assert svc.list() == []

    # 创建 4 次快照应自动裁剪到仅保留最近 2 次。
    for idx in range(4):
        svc.create(f"p{idx}")
    remaining = svc.list()
    assert len(remaining) == 2
    assert [r.phase for r in remaining] == ["p3", "p2"]
    svc.clean()


def test_team_coordinator_fails_fast_without_approval_callback(tmp_path: Path):
    """非交互入口未配置审批回调且 hitl_mode!=never 时立即报错而非空耗模型调用。"""
    client = _RecordingMockClient([])
    cfg = PaiCliConfig()
    cfg.policy.hitl_mode = "auto"
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    team = TeamCoordinator(
        llm_client=client, tool_registry=reg, config=cfg, cwd=str(tmp_path), approval_callback=None
    )

    async def _run():
        done = {}
        async for ev in team.run("do team task"):
            if ev["type"] == "done":
                done = ev
        return done

    done = asyncio.run(_run())
    assert done["status"] == "failed"
    assert done["total_turns"] == 0
    assert client.seen_messages == []


def test_team_model_preserves_updates_without_mutating_system_prompt():
    """团队更新持久化写入消息历史，不改写静态 system_prompt。"""
    client = _RecordingMockClient([[], []])
    cfg = PaiCliConfig()
    budget = _TeamBudget(cfg.team)
    inbox = ["taskA 请求 [request-1]：请补充空列表处理"]

    def pop_updates():
        batch = list(inbox)
        inbox.clear()
        return batch

    model = _TeamModel(client, budget, before=pop_updates)

    async def _run():
        msgs = [Message(role="user", content="start")]
        async for _ in model.chat(msgs, [], system_prompt="STATIC_SYS"):
            pass
        async for _ in model.chat(msgs, [], system_prompt="STATIC_SYS"):
            pass

    asyncio.run(_run())
    assert client.seen_system_prompts[0] == "STATIC_SYS"
    assert client.seen_system_prompts[1] == "STATIC_SYS"
    assert any("request-1" in str(m.content) for m in client.seen_messages[1])
