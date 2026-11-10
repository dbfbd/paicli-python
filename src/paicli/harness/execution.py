"""Debug 累计预算、工作区一致性检查及阶段边界恢复。"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from paicli.harness.evidence import current_run
from paicli.harness.verification import _fingerprints
from paicli.types import Usage

_CONTROL: ContextVar[RunControl | None] = ContextVar("run_control", default=None)
_DEFAULTS = {"tokens": 160000, "tools": 40, "models": 40}


def active_control():
    return _CONTROL.get()


class RunStopped(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class RunControl:
    """统一累计模型、工具和用量，并跟踪任务拥有的文件状态。

    预算在修复尝试和角色之间共享；恢复后继续累计，取消也保留已有开销。
    """

    def __init__(self, cwd: str, targets: list[str], client, limits: dict):
        self.run = current_run()
        self.cwd, self.targets = cwd, targets
        self.stop: RunStopped | None = None
        self.cancel_path = self.run.directory / "cancel.requested"
        identity = [client.provider_name, client.model_name]
        self.data = self.run.manifest.setdefault(
            "execution",
            {
                "limits": dict(_DEFAULTS),
                "models": 0,
                "tools": 0,
                "usage": {},
                "pending": {},
                "unreported_calls": 0,
                "unreported_token_reserve": 0,
                "model": identity,
            },
        )
        if self.data["model"] != identity:
            raise ValueError("Resume requires the original provider and model")
        updates = {name: value for name, value in limits.items() if value is not None}
        if any(type(value) is not int or value < 1 for value in updates.values()):
            raise ValueError("Execution limits must be positive integers")
        self.data["limits"].update(updates)
        self.data.pop("stopped", None)
        # 进程可能在流式调用结算前退出，此类调用也要保守计入预算。
        for pending in list(self.data["pending"].values()):
            self.data["unreported_calls"] += 1
            self.data["unreported_token_reserve"] += pending
        self.data["pending"].clear()
        self.expected = self.fingerprints()
        self.run.record("execution.opened", limits=self.data["limits"])

    @property
    def usage(self):
        return Usage.from_mapping(self.data["usage"])

    def fingerprints(self):
        return _fingerprints(Path(self.cwd), self.targets)[1]

    def check_files(self):
        """比较预期文件指纹和当前工作区，发现外部改动时使本次执行失效。"""
        actual = self.fingerprints()
        if actual != self.expected:
            changed = sorted(
                k
                for k in actual.keys() | self.expected.keys()
                if actual.get(k) != self.expected.get(k)
            )
            self.halt(
                "invalidated", "Workspace changed outside this execution: " + ", ".join(changed[:8])
            )

    def wrote(self, path: Path):
        """登记本执行器完成的写入，更新下一次一致性检查所期望的文件状态。"""
        from paicli.harness.evidence import _file_facts

        relative = path.relative_to(self.cwd).as_posix()
        if path.exists():
            self.expected[relative] = _file_facts(path)["sha256"]
        else:
            self.expected.pop(relative, None)

    def halt(self, status, reason):
        self.stop = RunStopped(status, reason)
        self.data["stopped"] = {"status": status, "reason": reason}
        self.run.record("execution.stopped", status=status, reason=reason)
        raise self.stop

    def check_cancel(self):
        """发现取消标记时停止后续动作派发，保留已有证据和累计用量。"""
        if self.cancel_path.exists():
            self.halt("cancelled", "Cancellation requested; no further actions dispatched")

    def tool(self):
        """在工具执行前检查取消和累计调用上限，再登记本次派发。"""
        self.check_cancel()
        if self.stop:
            raise self.stop
        if self.data["tools"] >= self.data["limits"]["tools"]:
            self.halt("budget_exhausted", "Shared tool request limit reached")
        self.data["tools"] += 1
        self.run.record("budget.tool", used=self.data["tools"])

    def begin_model(self, client, messages, tools, system_prompt):
        """派发前检查共享阈值并预留未报告用量；最终一次请求仍可能越过 token 阈值。"""
        self.check_cancel()
        if self.stop:
            raise self.stop
        if reason := self.model_budget_reason():
            self.halt("budget_exhausted", reason)
        # 字节数加输出上限仅用于保守预留；真实用量以服务方报告为准。
        payload = json.dumps(
            [system_prompt, [asdict(m) for m in messages], tools], ensure_ascii=False
        )
        reservation = len(payload.encode("utf-8")) + int(getattr(client, "max_tokens", 8192))
        ticket = uuid4().hex
        self.data["models"] += 1
        self.data["pending"][ticket] = reservation
        self.run.record("budget.model_started", request=ticket, reserved_if_unreported=reservation)
        return ticket

    @property
    def remaining_tokens(self):
        return (
            self.data["limits"]["tokens"]
            - self.usage.total_tokens
            - self.data["unreported_token_reserve"]
        )

    @property
    def remaining_models(self):
        return self.data["limits"]["models"] - self.data["models"]

    def model_budget_reason(self):
        if self.remaining_models <= 0:
            return "Shared model request limit reached"
        if self.remaining_tokens <= 0:
            return "Shared token dispatch threshold reached"
        return ""

    def end_model(self, ticket, usage, reported):
        """回收请求登记并累计官方用量；缺少用量报告时保守计入预留值。"""
        reservation = self.data["pending"].pop(ticket)
        self.data["usage"] = (self.usage + usage).to_dict()
        if not reported:
            self.data["unreported_calls"] += 1
            self.data["unreported_token_reserve"] += reservation
        self.run.record(
            "budget.model_finished", request=ticket, reported=reported, usage=usage.to_dict()
        )

    def save(self, payload):
        """核对文件后保存状态产物，并在清单中登记恢复所需的下一阶段。"""
        self.check_files()
        artifact = self.run.save_text(
            "debug.checkpoint",
            json.dumps(
                {"version": 1, "workspace": self.expected, **payload},
                ensure_ascii=False,
                default=asdict,
            ),
        )
        self.run.manifest["checkpoint"] = artifact["id"]
        self.run.manifest["resume_next"] = payload["state"]["next"]
        self.run.manifest.pop("active_node", None)
        self.run.record("debug.checkpoint", artifact=artifact["id"], next=payload["state"]["next"])

    def load(self):
        """读取并验证检查点产物，再核对工作区与保存时的文件版本是否一致。"""
        identifier = self.run.manifest.get("checkpoint")
        if not identifier:
            raise ValueError("This run has no resumable phase checkpoint")
        self.run.read_text(identifier, limit=1)  # 校验产物归属和完整内容哈希。
        path = self.run.directory / self.run.manifest["artifacts"][identifier]["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Unsupported debug checkpoint version")
        self.expected = payload["workspace"]
        self.check_files()
        return payload

    @contextmanager
    def activate(self):
        """在当前异步上下文绑定预算控制器，供模型和工具入口共同检查。"""
        token = _CONTROL.set(self)
        try:
            yield
        finally:
            _CONTROL.reset(token)

    async def watch_cancel(self, owner):
        """轮询取消标记并取消所属执行任务，使正在等待的异步调用也能退出。"""
        while True:
            await asyncio.sleep(0.2)
            if self.cancel_path.exists():
                owner.cancel()
                return


class BudgetedClient:
    """包装原模型客户端，为每次流式请求追加共享预算及角色预算检查。"""

    def __init__(self, client, control, *, token_limit=None, reserve_tokens=0, reserve_models=0):
        self.client, self.control = client, control
        self.token_limit, self.reserve_tokens = token_limit, reserve_tokens
        self.reserve_models = reserve_models
        self.used_tokens = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def chat(self, messages, tools, *, system_prompt):
        """检查阶段预留预算，透传模型事件，并在流关闭或异常时完成用量结算。"""
        if reason := self.control.model_budget_reason():
            self.control.halt("budget_exhausted", reason)
        if (
            (self.token_limit is not None and self.used_tokens >= self.token_limit)
            or self.control.remaining_tokens <= self.reserve_tokens
            or self.control.remaining_models <= self.reserve_models
        ):
            raise RunStopped(
                "budget_exhausted", "Stage dispatch budget reached; reserved capacity retained"
            )
        ticket = self.control.begin_model(self.client, messages, tools, system_prompt)
        usage, reported = Usage(), False
        stream = None
        try:
            stream = self.client.chat(messages, tools, system_prompt=system_prompt)
            async for event in stream:
                if event.get("type") == "usage":
                    usage += Usage.from_mapping(event.get("usage") or {})
                    reported = True
                yield event
        finally:
            try:
                if close := getattr(stream, "aclose", None):
                    await close()
            finally:
                self.used_tokens += (
                    usage.total_tokens if reported else self.control.data["pending"][ticket]
                )
                self.control.end_model(ticket, usage, reported)
