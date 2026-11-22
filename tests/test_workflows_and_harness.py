"""代码感知、多 Agent 协作与代码向 Harness 单元与集成测试。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from paicli.agent.agent import Agent
from paicli.agent.collaboration import Collaboration
from paicli.agent.debug import DebugCoordinator
from paicli.agent.explore import ExploreRunner
from paicli.agent.query import query
from paicli.agent.team_board import TeamBoard, TeamPlan, parse_team_plan
from paicli.agent.team_checks import run_candidate_checks
from paicli.agent.team_workspace import TeamWorkspace
from paicli.config import PaiCliConfig
from paicli.harness.code_map import build_code_map, static_callers_for_file
from paicli.harness.evidence import _CURRENT, EvidenceRun
from paicli.harness.execution import BudgetedClient, RunControl, RunStopped
from paicli.harness.verification import _digest, _interpreter
from paicli.lsp.diagnostics import diagnose_python_edit
from paicli.tools.base import Tool, ToolContext, ToolResult
from paicli.tools.builtins import _edit_file, _read_file, get_builtin_tools
from paicli.tools.registry import ToolRegistry


def _init_git_repo(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)


def _tc_event(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "tool_call_delta",
        "tool_call": {
            "index": 0,
            "id": call_id,
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        },
    }


def _test_config() -> PaiCliConfig:
    cfg = PaiCliConfig()
    return replace(cfg, llm=replace(cfg.llm, api_key="test-key"))


class _ScriptedClient:
    provider_name = "mock"
    model_name = "mock-model"
    max_context_window = 64_000
    max_tokens = 4096
    price_profile = None

    def __init__(self, turns: list[list[dict]]):
        self._turns = list(turns)
        self.calls: list[dict] = []

    async def chat(self, messages, tools, *, system_prompt):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": [t["function"]["name"] for t in tools],
                "system_prompt": system_prompt,
            }
        )
        events = self._turns.pop(0) if self._turns else [
            {"type": "text_delta", "text": "done"},
            {"type": "usage", "usage": {"input_tokens": 50, "output_tokens": 10}},
            {"type": "message_end", "stop_reason": "end_turn"},
        ]
        for ev in events:
            yield ev


# ---------------------------------------------------------------------------
# Code Map / Explore / Workflow Dispatch
# ---------------------------------------------------------------------------


def test_code_map_overview_and_focused_callers(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "core.py").write_text(
        "def compute(x: int, y: int = 1) -> int:\n    return x + y\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "service.py").write_text(
        "from app.core import compute\n\ndef run(v: int) -> int:\n    return compute(v, 2)\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_service.py").write_text(
        "from app.service import run\n\ndef test_run():\n    assert run(3) == 5\n",
        encoding="utf-8",
    )

    overview = build_code_map(tmp_path)
    assert "app/core.py" in overview["text"]
    assert "def compute" in overview["text"]

    focused = build_code_map(tmp_path, paths=["app/core.py"])
    assert "app/service.py" in focused["text"]
    assert "tests/test_service.py" in focused["text"]

    callers = static_callers_for_file(tmp_path, "app/core.py")
    assert "app/service.py" in callers


def test_explore_parallel_subagents_and_usage_rollup(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 10\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 20\n", encoding="utf-8")

    class _ParallelExploreClient:
        provider_name = "mock"
        model_name = "mock-model"
        max_context_window = 64_000
        price_profile = None

        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.seen_tool_sets: list[list[str]] = []

        async def chat(self, messages, tools, *, system_prompt):
            self.seen_tool_sets.append([t["function"]["name"] for t in tools])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.05)
            self.active -= 1
            prompt_text = str(messages[-1].content)
            yield {"type": "text_delta", "text": f"Found evidence for: {prompt_text[:40]}"}
            yield {"type": "usage", "usage": {"input_tokens": 120, "output_tokens": 30}}
            yield {"type": "message_end", "stop_reason": "end_turn"}

    client = _ParallelExploreClient()
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    events: list[dict] = []
    runner = ExploreRunner(
        llm_client=client,
        base_registry=reg,
        config=PaiCliConfig(),
        cwd=str(tmp_path),
        event_sink=events.append,
    )

    async def _go():
        ctx = ToolContext(cwd=str(tmp_path), config=PaiCliConfig())
        return await runner.run_questions(
            {
                "questions": [
                    {"topic": "alpha", "question": "Check a.py", "focus_paths": ["a.py"]},
                    {"topic": "beta", "question": "Check b.py", "focus_paths": ["b.py"]},
                ]
            },
            ctx,
        )

    res = asyncio.run(_go())
    assert not res.is_error
    payload = json.loads(res.content)
    assert len(payload["reports"]) == 2
    assert client.max_active >= 2, "Explore subagents should run concurrently"
    assert runner.calls == 2
    assert runner.usage.total_tokens == 300
    assert any(ev.get("type") == "explore_status" and ev.get("phase") == "started" for ev in events)
    assert all("write_file" not in tset and "edit_file" not in tset for tset in client.seen_tool_sets)


def test_no_langgraph_in_specialized_or_team_checks() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for rel in ("src/paicli/agent/specialized.py", "src/paicli/agent/team_checks.py"):
        text = (repo_root / rel).read_text(encoding="utf-8")
        assert "langgraph" not in text, f"{rel} should not depend on langgraph"


# ---------------------------------------------------------------------------
# Team Worktree Reuse, Board Contracts, Collaboration Auto-Checks
# ---------------------------------------------------------------------------


def test_team_board_preserves_accepted_and_allows_empty_checks() -> None:
    plan = parse_team_plan(
        json.dumps(
            {
                "goal": "Implement two parsers",
                "tasks": [
                    {"id": "csv", "goal": "Write csv parser", "paths": ["parsers/csv.py"]},
                    {
                        "id": "ini",
                        "goal": "Write ini parser",
                        "paths": ["parsers/ini.py"],
                        "publishes": "parse_ini(text) -> dict",
                    },
                ],
            }
        ),
        allow_empty_checks=True,
    )
    assert isinstance(plan, TeamPlan)
    assert plan.checks == ()

    board = TeamBoard(plan)
    assert board.claim("ini")
    req = board.request("csv", "ini", "export parse_ini(text) -> dict")
    board.publish("ini", "a" * 40, "v1")
    board.respond("ini", req.id, "done")
    board.accept("csv", req.id)
    assert board.requirements[req.id].accepted is True

    # Re-publishing the same task must NOT wipe already-accepted non-check requirements
    board.publish("ini", "b" * 40, "v2")
    assert board.requirements[req.id].accepted is True


def test_team_reusable_check_worktree_and_serial_checks(tmp_path: Path) -> None:
    _init_git_repo(
        tmp_path,
        {
            "pkg.py": "def add(a, b):\n    return a + b\n",
            "test_pkg.py": (
                "from pkg import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
    )
    plan = parse_team_plan(
        json.dumps(
            {
                "goal": "Verify serial checks reuse one worktree",
                "tasks": [{"id": "w1", "goal": "touch pkg", "paths": ["pkg.py"]}],
                "checks": [
                    {"id": "c1", "argv": ["python", "-c", "import pkg; assert pkg.add(1, 2) == 3"]},
                    {"id": "c2", "argv": ["pytest", "-q", "test_pkg.py"]},
                ],
            }
        )
    )

    async def _go():
        with TeamWorkspace(str(tmp_path)) as ws:
            ws.integrate(())
            out = []
            async for _check, result in run_candidate_checks(
                plan.checks,
                ws,
                config=PaiCliConfig(),
                approval_callback=lambda _req: "approve",
            ):
                out.append(result)
            return out

    results = asyncio.run(_go())
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_collaboration_auto_inferred_checks_and_live_events(tmp_path: Path) -> None:
    _init_git_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/mod_a.py": "def val_a() -> int:\n    return 0\n",
            "app/mod_b.py": "def val_b() -> int:\n    return 0\n",
            "tests/test_mod_a.py": "from app.mod_a import val_a\n\ndef test_a():\n    assert val_a() == 1\n",
            "tests/test_mod_b.py": "from app.mod_b import val_b\n\ndef test_b():\n    assert val_b() == 2\n",
        },
    )

    class _WorkerMockClient:
        provider_name = "mock"
        model_name = "mock-model"
        max_context_window = 64_000
        max_tokens = 4096
        price_profile = None

        def __init__(self) -> None:
            self.seen_system_prompts: list[str] = []
            self.seen_first_users: list[str] = []

        async def chat(self, messages, tools, *, system_prompt):
            self.seen_system_prompts.append(system_prompt)
            has_tool_result = any(m.role == "tool" for m in messages)
            first_user = next(str(m.content) for m in messages if m.role == "user")
            self.seen_first_users.append(first_user)
            if "你的工作 [a]" in first_user:
                target, body = "app/mod_a.py", "def val_a() -> int:\n    return 1\n"
            else:
                target, body = "app/mod_b.py", "def val_b() -> int:\n    return 2\n"

            if not has_tool_result:
                yield _tc_event(f"call_{target}", "write_file", {"path": target, "content": body})
                yield {"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 30}}
                yield {"type": "message_end", "stop_reason": "tool_use"}
            else:
                yield {"type": "text_delta", "text": f"Updated {target}"}
                yield {"type": "usage", "usage": {"input_tokens": 120, "output_tokens": 20}}
                yield {"type": "message_end", "stop_reason": "end_turn"}

    client = _WorkerMockClient()
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    live_events: list[dict] = []
    collab = Collaboration(
        llm_client=client,
        registry=reg,
        config=_test_config(),
        cwd=str(tmp_path),
        approval_callback=lambda _req: "approve",
        event_sink=live_events.append,
    )

    async def _go():
        collab_reg = collab.registry()
        ctx = ToolContext(
            cwd=str(tmp_path),
            config=_test_config(),
            approval_callback=lambda _req: "approve",
        )
        r1 = await collab_reg.get("delegate_work").execute(
            {
                "goal": "Implement val_a and val_b",
                "shared_context": "CONTRACT: val_a returns 1, val_b returns 2",
                "tasks": [
                    {"id": "a", "goal": "Implement app/mod_a.py", "paths": ["app/mod_a.py"]},
                    {"id": "b", "goal": "Implement app/mod_b.py", "paths": ["app/mod_b.py"]},
                ],
                # checks intentionally omitted to verify auto-inference!
            },
            ctx,
        )
        assert not r1.is_error, r1.content
        r2 = await collab_reg.get("collect_work").execute({}, ctx)
        assert not r2.is_error, r2.content
        r3 = await collab_reg.get("apply_work").execute({}, ctx)
        assert not r3.is_error, r3.content
        await collab.close()

    run = EvidenceRun.create(str(tmp_path), "react")
    token = _CURRENT.set(run)
    try:
        asyncio.run(_go())
    finally:
        _CURRENT.reset(token)
        run.finish("completed")
    assert "return 1" in (tmp_path / "app" / "mod_a.py").read_text(encoding="utf-8")
    assert "return 2" in (tmp_path / "app" / "mod_b.py").read_text(encoding="utf-8")
    assert all("CONTRACT: val_a returns 1" in u for u in client.seen_first_users)
    assert len(set(client.seen_system_prompts)) == 1
    assert any(ev.get("type") == "team_status" for ev in live_events)


# ---------------------------------------------------------------------------
# Code Harness: Diagnostics, Freshness, Anti-Loop, Verification & Budget Control
# ---------------------------------------------------------------------------


def test_l1_diagnostics_diff_scoped_ruff_and_ast_contract(tmp_path: Path) -> None:
    (tmp_path / "api.py").write_text(
        "import os\n\ndef greet(name: str) -> str:\n    return f'hi {name}'\n",
        encoding="utf-8",
    )
    (tmp_path / "consumer.py").write_text(
        "from api import greet\n\nprint(greet('world'))\n",
        encoding="utf-8",
    )
    before = (tmp_path / "api.py").read_text(encoding="utf-8")
    # Keep pre-existing unused `import os`, introduce new F821 undefined name AND change signature!
    (tmp_path / "api.py").write_text(
        "import os\n\ndef greet(name: str, title: str) -> str:\n    return f'{title} {missing_var}'\n",
        encoding="utf-8",
    )
    notes = diagnose_python_edit(tmp_path, tmp_path / "api.py", before_text=before)
    joined = "\n".join(notes)
    # Pre-existing F401 (`os` imported but unused) should NOT be reported
    assert "F401" not in joined
    # Newly introduced undefined name F821 SHOULD be reported (when ruff is installed)
    assert "F821" in joined or "missing_var" in joined
    # Public signature change should warn about static caller consumer.py
    assert "[Contract]" in joined
    assert "consumer.py" in joined


def test_l2_read_before_edit_freshness_rejects_external_stale_write(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = ToolContext(cwd=str(tmp_path), config=PaiCliConfig())

    async def _go():
        r_read = await _read_file({"path": "sample.py"}, ctx)
        assert not r_read.is_error

        # External modification changes file content and size
        target.write_text("x = 1\ny = 999\n", encoding="utf-8")
        r_stale = await _edit_file(
            {"path": "sample.py", "old_text": "x = 1", "new_text": "x = 2"},
            ctx,
        )
        assert r_stale.is_error
        assert "changed on disk" in r_stale.content

        # Re-read refreshes version -> edit now succeeds
        await _read_file({"path": "sample.py"}, ctx)
        r_ok = await _edit_file(
            {"path": "sample.py", "old_text": "x = 1", "new_text": "x = 2"},
            ctx,
        )
        assert not r_ok.is_error

    run = EvidenceRun.create(str(tmp_path), "react")
    token = _CURRENT.set(run)
    try:
        asyncio.run(_go())
    finally:
        _CURRENT.reset(token)
        run.finish("completed")
    assert "x = 2" in target.read_text(encoding="utf-8")


def test_l2_anti_doom_loop_injects_hint_after_repeated_failures(tmp_path: Path) -> None:
    reg = ToolRegistry()

    async def _always_fail(_payload, _ctx):
        return ToolResult("boom: bad arg", is_error=True)

    reg.register(
        Tool(
            name="flaky_tool",
            description="Always fails",
            parameters={"type": "object", "properties": {"k": {"type": "string"}}},
            handler=_always_fail,
        )
    )
    client = _ScriptedClient(
        [
            [
                _tc_event("c1", "flaky_tool", {"k": "same"}),
                {"type": "usage", "usage": {"input_tokens": 20, "output_tokens": 10}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            [
                _tc_event("c2", "flaky_tool", {"k": "same"}),
                {"type": "usage", "usage": {"input_tokens": 30, "output_tokens": 10}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            [
                {"type": "text_delta", "text": "Switched strategy"},
                {"type": "usage", "usage": {"input_tokens": 40, "output_tokens": 10}},
                {"type": "message_end", "stop_reason": "end_turn"},
            ],
        ]
    )

    async def _go():
        events = []
        async for ev in query(
            llm_client=client,
            tool_registry=reg,
            system_prompt="test",
            user_message="run tool",
            history=[],
            cwd=str(tmp_path),
            config=PaiCliConfig(),
            max_turns=5,
        ):
            events.append(ev)
        return events

    run = EvidenceRun.create(str(tmp_path), "react")
    token = _CURRENT.set(run)
    try:
        asyncio.run(_go())
    finally:
        _CURRENT.reset(token)
        run.finish("completed")
    tool_msgs = [m for m in client.calls[2]["messages"] if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert "Repeated identical tool failure" not in str(tool_msgs[0].content)
    assert "Repeated identical tool failure" in str(tool_msgs[1].content)


def test_l3_unverified_write_before_finish_reminder(tmp_path: Path) -> None:
    _init_git_repo(
        tmp_path,
        {
            "app.py": "def f():\n    return 1\n",
            "tests/test_app.py": "from app import f\n\ndef test_f():\n    assert f() == 2\n",
        },
    )
    client = _ScriptedClient(
        [
            # Turn 1: edit app.py
            [
                _tc_event("e1", "write_file", {"path": "app.py", "content": "def f():\n    return 2\n"}),
                {"type": "usage", "usage": {"input_tokens": 50, "output_tokens": 20}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            # Turn 2: attempt to finish WITHOUT running pytest -> should trigger L3 reminder!
            [
                {"type": "text_delta", "text": "Done editing."},
                {"type": "usage", "usage": {"input_tokens": 60, "output_tokens": 10}},
                {"type": "message_end", "stop_reason": "end_turn"},
            ],
            # Turn 3: after reminder, finishes cleanly
            [
                {"type": "text_delta", "text": "Verified."},
                {"type": "usage", "usage": {"input_tokens": 70, "output_tokens": 10}},
                {"type": "message_end", "stop_reason": "end_turn"},
            ],
        ]
    )
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    agent = Agent(
        llm_client=client,
        tool_registry=reg,
        config=_test_config(),
        cwd=str(tmp_path),
        approval_callback=lambda _req: "approve",
        max_turns=6,
    )

    async def _go():
        return [ev async for ev in agent.run("Fix f() to return 2")]

    run = EvidenceRun.create(str(tmp_path), "react")
    token = _CURRENT.set(run)
    try:
        asyncio.run(_go())
    finally:
        _CURRENT.reset(token)
        run.finish("completed")
    assert len(client.calls) == 3
    last_call_messages = client.calls[2]["messages"]
    assert any("[Harness]" in str(m.content) for m in last_call_messages if m.role == "user")


def test_l4_atomic_in_flight_token_reservation(tmp_path: Path) -> None:
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    started_first = asyncio.Event()
    release_first = asyncio.Event()

    class _SlowClient:
        provider_name = "mock"
        model_name = "mock-model"
        max_tokens = 2048

        async def chat(self, messages, tools, *, system_prompt):
            started_first.set()
            await release_first.wait()
            yield {"type": "usage", "usage": {"input_tokens": 500, "output_tokens": 100}}

    base = _SlowClient()
    run = EvidenceRun.create(str(tmp_path), "debug")
    token = _CURRENT.set(run)
    try:
        async def _run_body():
            control = RunControl(
                str(tmp_path),
                ["test_x.py"],
                base,
                {"tokens": 2500, "models": 10, "tools": 10},
            )
            c1 = BudgetedClient(base, control)
            c2 = BudgetedClient(base, control)

            async def _consume(client):
                return [ev async for ev in client.chat([], [], system_prompt="x" * 1500)]

            t1 = asyncio.create_task(_consume(c1))
            await started_first.wait()
            assert control.in_flight_tokens > 2000
            assert control.remaining_tokens <= 0
            blocked = False
            try:
                await _consume(c2)
            except RunStopped as exc:
                blocked = exc.status == "budget_exhausted"
            finally:
                release_first.set()
                await t1
            assert blocked
            assert control.in_flight_tokens == 0
            assert control.remaining_tokens == 1900

        asyncio.run(_run_body())
    finally:
        _CURRENT.reset(token)
        run.finish("completed")


def test_l4_interpreter_and_digest_caching(tmp_path: Path) -> None:
    f = tmp_path / "data.py"
    f.write_text("a = 1\n", encoding="utf-8")
    # Same-length immediate overwrite within racy window MUST NOT return stale digest
    d1 = _digest(f)
    f.write_text("a = 2\n", encoding="utf-8")
    d2 = _digest(f)
    assert d1 != d2

    # Backdated file (> 1s old) hits stat cache cleanly
    past = time.time() - 5.0
    os.utime(f, (past, past))
    assert _digest(f) == d2

    # _interpreter cache returns identical identity on second call without error
    run = EvidenceRun.create(str(tmp_path), "verification")
    try:
        env = os.environ.copy()
        id1 = asyncio.run(_interpreter(run, sys.executable, env))
        id2 = asyncio.run(_interpreter(run, sys.executable, env))
        assert id1 == id2
    finally:
        run.finish("completed")


def test_l3_l4_debug_receipt_reuse_and_proposed_edit_attempt(tmp_path: Path) -> None:
    _init_git_repo(
        tmp_path,
        {
            "calc.py": "def mul(a, b):\n    return a + b\n",
            "tests/test_calc.py": (
                "from calc import mul\n\n"
                "def test_mul():\n    assert mul(3, 4) == 12\n"
            ),
        },
    )

    # Lead proposes a WRONG edit (`a - b`), max_attempts=1:
    # Because proposed_edit does NOT consume Repairer's attempt, Repairer STILL gets its full
    # attempt=1, fixes `calc.py` to `a * b`, calls `run_tests` (gets accepted), and calls `finish_task`.
    # Then `_verify` reuses the accepted receipt instead of re-running pytest!
    client = _ScriptedClient(
        [
            [
                _tc_event(
                    "e1",
                    "write_file",
                    {"path": "calc.py", "content": "def mul(a, b):\n    return a * b\n"},
                ),
                {"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 30}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            [
                _tc_event("t1", "run_tests", {}),
                {"type": "usage", "usage": {"input_tokens": 120, "output_tokens": 15}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
            [
                _tc_event("f1", "finish_task", {"summary": "Fixed mul to return a * b"}),
                {"type": "usage", "usage": {"input_tokens": 150, "output_tokens": 20}},
                {"type": "message_end", "stop_reason": "tool_use"},
            ],
        ]
    )
    reg = ToolRegistry()
    reg.register_all(get_builtin_tools())
    coord = DebugCoordinator(
        llm_client=client,
        tool_registry=reg,
        config=PaiCliConfig(),
        cwd=str(tmp_path),
        targets=["tests/test_calc.py"],
        approval_callback=lambda _req: "approve",
        max_attempts=1,
        python_executable=sys.executable,
        proposed_edit=(
            "write_file",
            {"path": "calc.py", "content": "def mul(a, b):\n    return a - b\n"},
        ),
    )

    async def _go():
        done = None
        async for ev in coord.run("Fix mul(3, 4) to return 12"):
            if ev.get("type") == "done":
                done = ev
        return done

    done = asyncio.run(_go())
    assert done is not None
    assert done["status"] == "completed", done
    assert done["attempts"] == 1
    assert done["verification"]["status"] == "accepted"

    # Confirm events.jsonl shows reused_receipt=True on final debug.verification!
    ev_path = Path(done["evidence_dir"]) / "events.jsonl"
    events = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines() if line]
    verifications = [ev for ev in events if ev.get("type") == "debug.verification"]
    assert any(ev.get("reused_receipt") is True for ev in verifications)
