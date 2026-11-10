"""任务交接契约、源码版本证据及实际执行的工具权限检查。"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path

from paicli.harness.evidence import current_run

_ACTIVE: ContextVar[TaskPolicy | None] = ContextVar("task_policy", default=None)
INTERNAL_DIRS = {".git", ".venv", "venv", ".paicli", "__pycache__", "node_modules"}
WRITE_TOOLS = {"write_file", "edit_file"}


def active_policy():
    return _ACTIVE.get()


@dataclass
class TaskSpec:
    """明确子任务问题、允许范围和读写配额，作为执行和结果校验的共同约束。"""

    task_id: str
    kind: str
    question: str
    read_scope: list[str]
    max_read_calls: int | None = 4
    max_write_calls: int | None = 0
    write_scope: list[str] = field(default_factory=list)
    auto_checkpoint: bool = True  # 默认沿用专员提交及检查点行为。

    @classmethod
    def parse(cls, task_id: str, data: dict, root: str):
        """验证委派数据、项目内路径和读取配额，拒绝不存在或越界的任务范围。"""
        if not isinstance(data, dict):
            raise ValueError("Delegation requires a TaskSpec object, not a vague question")
        scopes = data.get("read_scope", [])
        if not isinstance(scopes, list) or not scopes or len(scopes) > 16:
            raise ValueError("TaskSpec needs 1 to 16 read_scope paths")
        base = Path(root).resolve()
        normalised = []
        for name in scopes:
            if not isinstance(name, str) or not (base / name).resolve().is_relative_to(base):
                raise ValueError("Task scope must stay inside the project")
            path = (base / name).resolve()
            relative = path.relative_to(base)
            if not path.exists():
                raise ValueError(f"Scope path does not exist: {name}; locate it with glob_files")
            if any(p.casefold() in INTERNAL_DIRS for p in relative.parts):
                raise ValueError("Internal directories cannot be delegated")
            normalised.append(relative.as_posix())
        question = data.get("question", "")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("TaskSpec needs a concrete bounded question")
        calls = data.get("max_read_calls", 4)
        if type(calls) is not int or not 1 <= calls <= 12:
            raise ValueError("Task tool quota must be between 1 and 12")
        kind = data.get("kind", "root_cause")
        if kind not in {"acceptance", "root_cause", "impact", "repair", "review"}:
            raise ValueError("Unsupported investigation kind")
        return cls(task_id, kind, question, list(dict.fromkeys(normalised)), calls)

    def contains(self, root: str, path: Path) -> bool:
        """检查目标路径是否位于任务可读范围，并排除项目内部状态目录。"""
        if not path.is_relative_to(root) or any(
            p.casefold() in INTERNAL_DIRS for p in path.relative_to(root).parts
        ):
            return False
        return any(
            path == candidate or (candidate.is_dir() and path.is_relative_to(candidate))
            for name in self.read_scope
            for candidate in [(Path(root) / name).resolve()]
        )


class EvidenceBook:
    """维护证据引用和源码版本，将角色结论关联到可复核的原始材料。"""

    def __init__(self, root: str):
        self.root = str(Path(root).resolve())
        self.items: dict[str, dict] = {}
        self.sources: dict[tuple[str, str], dict] = {}

    def bind(self, artifact: dict, *, path: str = "", sha: str = "") -> str:
        reference = f"{artifact['run_id']}:{artifact['id']}"
        self.items[reference] = {"artifact": artifact, "path": path, "source_sha": sha}
        return reference

    def source(self, path: Path) -> dict:
        """为源码登记内容哈希和可定位信息，保留不同版本以识别过期引用。"""
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Source must be a project file")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        relative = path.relative_to(self.root).as_posix()
        key = (relative, digest)
        if key not in self.sources:
            text = raw.decode("utf-8")
            artifact = current_run().save_text("source.snapshot", text)
            reference = self.bind(artifact, path=relative, sha=digest)
            self.sources[key] = {
                "ref": reference,
                "path": relative,
                "sha": digest,
                "text": text,
                "lines": len(text.splitlines()),
                "focus": set().union(
                    *(s.get("focus", set()) for s in self.sources.values() if s["path"] == relative)
                ),
            }
            current_run().record("source.observed", path=relative, sha256=digest, ref=reference)
        return self.sources[key]

    def valid(self, reference: str, *, current: bool = True) -> dict:
        """核对引用归属、内容完整性及所关联源码的当前版本。"""
        item = self.items.get(reference)
        if item is None:
            raise ValueError("Unknown evidence reference")
        # read_text 会校验产物原始字节、归属及清单中的哈希。
        from paicli.harness.evidence import read_reference

        read_reference(reference, limit=1)
        if current and item["path"]:
            path = Path(self.root) / item["path"]
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["source_sha"]
            ):
                raise ValueError("Stale source evidence")
        return item

    def observe(self, name: str, payload: dict, result) -> None:
        """登记工具执行产生的证据，补充读取位置和源码版本信息。"""
        if result.is_error or not result.evidence or name in {"finish_task", "investigate"}:
            return
        source = None
        if name in {"read_file", "grep", "grep_code", *WRITE_TOOLS}:
            path = (Path(self.root) / str(payload.get("path", ""))).resolve()
            if path.is_file() and path.suffix == ".py":
                source = self.source(path)
        reference = self.bind(
            result.evidence,
            path=source["path"] if source else "",
            sha=source["sha"] if source else "",
        )
        if (policy := active_policy()) and policy.book is self:
            policy.evidence_refs.append(reference)
        result.content += f"\n[Evidence ref: {reference}]"
        if source:
            result.content += (
                f"\n[Source ref: {source['ref']}; path={source['path']}; sha={source['sha']}]"
            )

    def support(self, reference: str, *, current: bool = True) -> dict:
        item = self.valid(reference, current=current)
        kind = item["artifact"]["kind"]
        if kind == "task-result" or kind.startswith(("agent.response.", "knowledge.", "impact.")):
            raise ValueError(
                "Agent reports, impact hints and retrieved guidance are navigation only; "
                "cite current source or verification evidence"
            )
        return item

    def packet(self, paths: list[str] | None = None, limit: int = 16000) -> str:
        """按关注文件挑选有限证据上下文，供执行角色核对而非重复堆入全部原文。"""
        sources = []
        latest = {source["path"]: source for source in self.sources.values()}
        for source in latest.values():
            if paths is not None and not any(
                source["path"] == name or source["path"].startswith(name.rstrip("/") + "/")
                for name in paths
            ):
                continue
            try:
                self.valid(source["ref"])
            except (OSError, ValueError):
                continue
            sources.append(source)
        sources.sort(
            key=lambda s: (
                (paths or []).index(s["path"]) if s["path"] in (paths or []) else len(paths or [])
            )
        )
        parts, used = [], 0
        for index, source in enumerate(sources):
            lines = source["text"].splitlines()
            focus = source.get("focus", set())
            selected = (
                sorted(
                    {
                        n
                        for line in focus
                        for n in range(max(1, line - 4), min(len(lines), line + 4) + 1)
                    }
                )
                if focus
                else range(1, len(lines) + 1)
            )
            numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in selected)
            header = f"\nSOURCE {source['ref']} path={source['path']} sha={source['sha']}\n"
            room = min(8000, (limit - used) // (len(sources) - index) - len(header) - 1)
            if room < 48:
                continue
            if len(numbered) > room:
                marker = "\n[Omitted; read the source ref]\n"
                head = (room - len(marker)) // 2
                numbered = numbered[:head] + marker + numbered[-(room - len(marker) - head) :]
            part = header + numbered
            parts.append(part)
            used += len(part)
        return "\n".join(parts)


@dataclass
class Finding:
    """调查员的有限结论，包含状态、观察、假设和不确定项。"""

    task_id: str
    status: str
    observations: list[dict]
    hypothesis: str
    unresolved: list[str]

    @classmethod
    def parse(cls, data: dict, spec: TaskSpec, book: EvidenceBook):
        """校验调查结果结构、引用及任务范围，拒绝缺乏有效证据的确定性结论。"""
        if not isinstance(data, dict) or set(data) - {
            "task_id",
            "status",
            "observations",
            "hypothesis",
            "unresolved",
        }:
            raise ValueError("Finding contains unsupported fields")
        status = data.get("status")
        if status not in {"answered", "needs_scope", "inconclusive", "budget_exhausted"}:
            raise ValueError("Invalid finding status")
        observations = data.get("observations", [])
        if not isinstance(observations, list) or len(observations) > 3:
            raise ValueError("Finding observations must be a bounded list")
        for observation in observations:
            if not isinstance(observation, dict) or not isinstance(observation.get("claim"), str):
                raise ValueError("An observation needs a claim")
            item = book.support(str(observation.get("ref", "")))
            if item["path"] and not spec.contains(book.root, Path(book.root) / item["path"]):
                raise ValueError("Finding cites a source outside its assigned scope")
            bounds = observation.get("lines") or None
            if bounds is not None:
                if (
                    not isinstance(bounds, list)
                    or any(type(n) is not int for n in bounds)
                    or any(n < 1 for n in bounds)
                ):
                    raise ValueError("Invalid evidence line range")
                text = (
                    (Path(book.root) / item["path"]).read_text(encoding="utf-8")
                    if item["path"]
                    else ""
                )
                if not text or max(bounds) > len(text.splitlines()):
                    raise ValueError("Evidence line range does not exist")
                source = book.sources.get((item["path"], item["source_sha"]))
                if source:
                    source["focus"].update(bounds)
        hypothesis = data.get("hypothesis", "")
        unresolved = data.get("unresolved", [])
        if (
            not isinstance(hypothesis, str)
            or not isinstance(unresolved, list)
            or any(not isinstance(value, str) for value in unresolved)
        ):
            raise ValueError("Finding needs a hypothesis string and unresolved string list")
        if status == "answered" and (not observations or not hypothesis.strip()):
            raise ValueError("An answered investigation needs actual evidence and a hypothesis")
        return cls(spec.task_id, status, observations, hypothesis, unresolved)

    def text(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class TaskPolicy:
    """任务执行期间生效的工具策略，记录配额、提交结果和停止原因。"""

    spec: TaskSpec
    book: EvidenceBook
    read_calls: int = 0
    write_calls: int = 0
    submission: dict | None = None
    finish_attempts: int = 0
    halt_reason: str = ""
    modified: set[str] = field(default_factory=set)
    evidence_refs: list[str] = field(default_factory=list)

    def checkpoint(self) -> bool:
        return (
            self.spec.auto_checkpoint
            and bool(self.spec.write_scope)
            and set(self.spec.write_scope) <= self.modified
        )

    def allows(self, name: str) -> bool:
        limit, used = (
            (self.spec.max_write_calls, self.write_calls)
            if name in WRITE_TOOLS
            else (self.spec.max_read_calls, self.read_calls)
        )
        return name == "finish_task" or limit is None or used < limit

    def check(self, name: str, payload: dict) -> str:
        """在调用工具前校验工具类别、路径和配额；写文件白名单由守卫注册表进一步检查。"""
        if self.halt_reason:
            return self.halt_reason
        if self.submission is not None:
            return "Task has already been submitted"
        if name == "finish_task":
            self.finish_attempts += 1
            if self.finish_attempts > 2:
                self.halt_reason = "Invalid submission repeated; returning without model claims"
                current_run().record("delegation.submission_abandoned", task_id=self.spec.task_id)
                return self.halt_reason
            return ""
        if not self.allows(name):
            return "Task quota exhausted for this tool. Use remaining edit tools or finish_task."
        if name in WRITE_TOOLS:
            self.write_calls += 1
            return ""  # 受限工具注册表还会单独检查允许写入的文件白名单。
        if name == "read_evidence":
            reference = str(payload.get("artifact_id", ""))
            try:
                item = self.book.valid(reference, current=self.spec.kind != "review")
                if item["path"] and not self.spec.contains(
                    self.book.root, Path(self.book.root) / item["path"]
                ):
                    return "needs_scope: evidence is outside the assigned source scope"
            except (OSError, ValueError) as exc:
                return f"Evidence rejected: {exc}"
        else:
            name_path = payload.get("path", ".")
            if name in {"glob", "glob_files"}:
                prefix = re.split(r"[?*\[]", str(payload.get("pattern", "")), maxsplit=1)[0]
                name_path = prefix.rsplit("/", 1)[0] if "/" in prefix else "."
            path = (Path(self.book.root) / str(name_path)).resolve()
            if not self.spec.contains(self.book.root, path):
                return (
                    "needs_scope: request a narrower or explicitly expanded scope "
                    "from the coordinator"
                )
        self.read_calls += 1
        return ""


@contextmanager
def task_contract(spec: TaskSpec, book: EvidenceBook):
    """把当前任务策略绑定到异步上下文，退出时恢复上层策略，避免角色间串用。"""
    policy = TaskPolicy(spec, book)
    token = _ACTIVE.set(policy)
    try:
        current_run().record("delegation.started", spec=asdict(spec))
        yield policy
    finally:
        _ACTIVE.reset(token)
        current_run().record(
            "delegation.finished",
            task_id=spec.task_id,
            read_calls=policy.read_calls,
            write_calls=policy.write_calls,
            tool_calls=policy.read_calls + policy.write_calls,
        )
