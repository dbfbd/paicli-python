"""执行证据存储：记录事件、校验产物、关联任务并为 Debug 提供项目写锁。"""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

_CURRENT: ContextVar[EvidenceRun | None] = ContextVar("paicli_run", default=None)
_TASK: ContextVar[dict[str, Any] | None] = ContextVar("paicli_task", default=None)


def current_run() -> EvidenceRun | None:
    return _CURRENT.get()


@contextmanager
def task_scope(task_id: str, attempt: int):
    token = _TASK.set({"task_id": task_id, "attempt": attempt})
    try:
        yield
    finally:
        _TASK.reset(token)


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid evidence identifier")
    return value


def _file_facts(path: Path) -> dict:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    size = chars = 0
    with path.open("rb") as stream:
        while block := stream.read(65536):
            digest.update(block)
            size += len(block)
            chars += len(decoder.decode(block))
    chars += len(decoder.decode(b"", final=True))
    return {"sha256": digest.hexdigest(), "bytes": size, "chars": chars}


class EvidenceRun:
    """管理独立运行目录，追加事件并以哈希校验保存的原始产物。"""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self._lock = RLock()
        self.manifest = json.loads((self.directory / "run.json").read_text(encoding="utf-8"))
        self.run_id = self.manifest["run_id"]

    @classmethod
    def create(cls, cwd: str, mode: str) -> EvidenceRun:
        """创建带运行编号和工作目录信息的证据记录。"""
        run_id = uuid4().hex
        directory = Path.home() / ".paicli" / "runs" / run_id
        directory.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "cwd": str(Path(cwd).resolve()),
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "sequence": 0,
            "artifacts": {},
        }
        (directory / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        run = cls(directory)
        run.record("run.started", mode=mode)
        return run

    def _save(self) -> None:
        temporary = self.directory / f"manifest-{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            for attempt in range(6):
                try:
                    temporary.replace(self.directory / "run.json")
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 5:
                        raise
                    # Windows 的读取程序或扫描程序可能短暂占用目标文件。
                    time.sleep(0.01 * 2**attempt)
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    def record(self, event_type: str, **fields) -> None:
        """追加带执行上下文的事件，同时保存更新后的运行清单。"""
        with self._lock:
            self.manifest["sequence"] += 1
            event = {
                "id": uuid4().hex,
                "run_id": self.run_id,
                "sequence": self.manifest["sequence"],
                "time": datetime.now(UTC).isoformat(),
                "type": event_type,
                **(_TASK.get() or {}),
                **fields,
            }
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._save()

    def save_text(self, kind: str, text: str) -> dict:
        """将原始文本保存为产物，并返回可供后续引用的元数据。"""
        path = self.directory / f"{uuid4().hex}.txt"
        path.write_text(text, encoding="utf-8", newline="")
        return self.adopt_file(kind, path)

    def adopt_file(self, kind: str, path: Path) -> dict:
        """登记运行目录内的已有文件，保存其大小、哈希和产物编号。"""
        path = Path(path).resolve()
        if path.parent != self.directory or not path.is_file():
            raise ValueError("Evidence files must belong to this run directory")
        with self._lock:
            artifact = {
                "id": uuid4().hex,
                "run_id": self.run_id,
                "kind": kind,
                "path": path.name,
                **_file_facts(path),
            }
            self.manifest["artifacts"][artifact["id"]] = artifact
            self.record("artifact.saved", artifact=artifact)
            return dict(artifact)

    def read_text(self, artifact_id: str, offset: int = 0, limit: int = 8000) -> str:
        """验证产物归属及完整内容哈希后，返回指定区间的文本。"""
        artifact = self.manifest["artifacts"].get(_identifier(artifact_id))
        if not artifact:
            raise ValueError("Evidence does not belong to this run")
        path = (self.directory / artifact["path"]).resolve()
        if path.parent != self.directory or _file_facts(path)["sha256"] != artifact["sha256"]:
            raise ValueError("Evidence content has changed; reference is invalid")
        offset = min(max(0, int(offset)), artifact["chars"])
        limit = max(1, min(int(limit), 16000))
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            remaining = offset
            while remaining:
                block = stream.read(min(remaining, 65536))
                if not block:
                    break
                remaining -= len(block)
            content = stream.read(limit)
        return (
            f"Evidence {self.run_id}:{artifact_id}; chars {offset}:{offset + len(content)} "
            f"of {artifact['chars']}; sha256={artifact['sha256']}\n{content}"
        )

    def finish(self, status: str, **fields) -> None:
        with self._lock:
            self.manifest.update(status=status, finished_at=datetime.now(UTC).isoformat(), **fields)
            self.record("run.finished", status=status)


def load_run(run_id: str) -> EvidenceRun:
    return EvidenceRun(Path.home() / ".paicli" / "runs" / _identifier(run_id))


@contextmanager
def project_lease(cwd: str):
    """按解析后的工作目录取得 Debug 单写锁；进程退出后由操作系统释放锁。"""
    root = Path(cwd).resolve()
    directory = (root / ".paicli").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Debug lock directory must stay inside the project")
    directory.mkdir(exist_ok=True)
    with (directory / "debug.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another debug execution owns this workspace") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_reference(reference: str, offset: int = 0, limit: int = 8000) -> str:
    """按运行编号和产物编号读取原始证据，并核对引用完整性。"""
    run = current_run()
    if run is None:
        raise ValueError("No active evidence run")
    if ":" in reference:
        run_id, reference = reference.split(":", 1)
        source = load_run(run_id)
        if source.manifest["cwd"] != run.manifest["cwd"]:
            raise ValueError("Evidence belongs to a different project")
        run = source
    elif reference not in run.manifest["artifacts"]:
        raise ValueError(
            "Artifact is not in the active run; for earlier turns use run_id:artifact_id"
        )
    return run.read_text(reference, offset, limit)


def evidence_context(text: str, artifact: dict | None, limit: int = 1800) -> str:
    if not artifact or len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n[Preview only; {artifact['chars']} characters in evidence "
        f"{artifact['run_id']}:{artifact['id']}, sha256={artifact['sha256']}. "
        "Use read_evidence(artifact_id, offset, limit) to retrieve the remaining evidence.]"
    )


def recorded_run(mode: str | None = None) -> Callable:
    """为一次执行建立证据上下文，避免向事件消费者泄漏上下文，并为 Debug 管理项目锁。"""

    def decorate(method):
        @wraps(method)
        async def wrapped(self, *args, **kwargs):
            if current_run() is not None:
                async for event in method(self, *args, **kwargs):
                    yield event
                return
            lease = project_lease(self.cwd) if mode == "debug" else nullcontext()
            with lease:
                consumer = consume(self, args, kwargs)
                try:
                    async for event in consumer:
                        yield event
                finally:
                    await consumer.aclose()

        async def consume(self, args, kwargs):
            resume = kwargs.get("resume") if mode == "debug" else None
            run = (
                load_run(resume)
                if resume
                else EvidenceRun.create(self.cwd, mode or getattr(self, "mode", "react"))
            )
            if resume:
                if (
                    run.manifest["cwd"] != str(Path(self.cwd).resolve())
                    or run.manifest["mode"] != "debug"
                ):
                    raise ValueError("Resume run must be a debug execution of this project")
                if run.manifest["status"] == "completed":
                    raise ValueError("This debug execution has already completed")
                run.manifest.update(status="running")
                run.manifest.pop("finished_at", None)
                (run.directory / "cancel.requested").unlink(missing_ok=True)
                run.record("run.resumed")
            stream = method(self, *args, **kwargs).__aiter__()
            status = "interrupted"
            try:
                while True:
                    token = _CURRENT.set(run)
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        break
                    finally:
                        _CURRENT.reset(token)
                    kind = event.get("type", "unknown")
                    if kind not in {"text_delta", "thinking_delta", "tool_call", "tool_result"}:
                        selected = {
                            key: event[key]
                            for key in (
                                "task_id",
                                "attempt",
                                "phase",
                                "status",
                                "turn",
                                "stop_reason",
                                "approved",
                                "total_turns",
                                "total_tokens",
                                "usage",
                            )
                            if key in event
                        }
                        run.record(kind, **selected)
                    if kind == "error":
                        status = "failed"
                        run.record("run.error", error=str(event.get("error")))
                    elif kind == "done":
                        status = str(event.get("status") or "completed")
                        event = {**event, "run_id": run.run_id, "evidence_dir": str(run.directory)}
                    yield event
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                status = str(getattr(exc, "status", "failed"))
                raise
            finally:
                token = _CURRENT.set(run)
                try:
                    await stream.aclose()
                finally:
                    _CURRENT.reset(token)
                    run.finish(status)

        return wrapped

    return decorate
