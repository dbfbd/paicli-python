"""Team 的任务分工、产物版本和跨任务需求。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from paicli.plan.validation import validate_plan_nodes

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\Z")
_INVALID_PATH_CHARACTERS = re.compile(r"[<>:\"|?*\x00-\x1f]")


@dataclass(frozen=True, slots=True)
class TeamTask:
    id: str
    goal: str
    paths: tuple[str, ...]
    needs: tuple[str, ...]
    publishes: str = ""


@dataclass(frozen=True, slots=True)
class TeamCheck:
    id: str
    argv: tuple[str, ...]
    expected_exit: int = 0
    stdout_contains: str = ""
    stdout_equals: str | None = None
    owner: str = ""
    parallel_safe: bool = False


@dataclass(frozen=True, slots=True)
class TeamPlan:
    tasks: tuple[TeamTask, ...]
    checks: tuple[TeamCheck, ...]


@dataclass(frozen=True, slots=True)
class TeamPublication:
    commit: str
    text: str
    version: int


@dataclass(frozen=True, slots=True)
class TeamRequirement:
    id: str
    requester: str
    owner: str
    condition: str
    check: TeamCheck | None
    response: str = ""
    response_version: int = 0
    accepted: bool = False


def parse_team_plan(text: str) -> TeamPlan:
    """把队长的结构化输出校验成可调度任务和检查项。"""
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw[:-3], flags=re.IGNORECASE)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Team plan must be a JSON object")
    task_data = _array(data.get("tasks"), "tasks", 24)
    tasks = []
    owners: dict[str, str] = {}
    for task in task_data:
        if not isinstance(task, dict):
            raise ValueError("Team tasks must be objects")
        task_id = _identifier(task.get("id"), "task id")
        paths = tuple(
            normalize_team_path(path)
            for path in _array(task.get("paths", []), "paths", 24)
        )
        for path in paths:
            key = path.casefold()
            if key in owners:
                raise ValueError(
                    f"Writable path {path} conflicts with the scope of {owners[key]}"
                )
            if any(
                key.startswith(other + "/") or other.startswith(key + "/")
                for other in owners
            ):
                raise ValueError(f"Writable path {path} overlaps another file scope")
            owners[key] = task_id
        needs = tuple(
            _identifier(dep, "dependency id")
            for dep in _array(task.get("needs", []), "needs", 23)
        )
        publication = task.get("publishes", "")
        if publication is True:
            publication = "先交付可调用的中间成果。"
        if publication is None or publication is False:
            publication = ""
        tasks.append(
            TeamTask(
                task_id,
                _string(task.get("goal"), "task goal", 1600, required=True),
                paths,
                tuple(dict.fromkeys(needs)),
                _string(publication, "publication", 700),
            )
        )
    validate_plan_nodes(
        {
            "tasks": [
                {"id": task.id, "dependencies": list(task.needs)} for task in tasks
            ]
        },
        primary_key="tasks",
        id_prefix="task",
    )
    checks = tuple(
        parse_team_check(check)
        for check in _array(data.get("checks", []), "checks", 20)
    )
    if len({check.id for check in checks}) != len(checks):
        raise ValueError("Team checks must have unique IDs")
    known = {task.id for task in tasks}
    for check in checks:
        if check.owner and check.owner not in known:
            raise ValueError(f"Check {check.id} has unknown task owner {check.owner}")
    if owners and not checks:
        raise ValueError(
            "A Team plan with writable tasks needs at least one runnable check"
        )
    return TeamPlan(tuple(tasks), checks)


def parse_team_check(data: Any, *, check_id: str | None = None) -> TeamCheck:
    """检查 argv 和实际期望，不在此处运行命令。"""
    if not isinstance(data, dict):
        raise ValueError("A Team check must be an object")
    identifier = _identifier(
        check_id if check_id is not None else data.get("id"), "check id"
    )
    argv = tuple(_array(data.get("argv"), "check argv", 32))
    if (
        not argv
        or any(
            not isinstance(arg, str) or "\0" in arg or len(arg) > 6000 for arg in argv
        )
        or not argv[0].strip()
    ):
        raise ValueError(
            "Check argv must contain a command and optional string arguments"
        )
    status = data.get("expected_exit", 0)
    if type(status) is not int or not -255 <= status <= 255:
        raise ValueError("Expected exit code must be an integer")
    exact = data.get("stdout_equals")
    if exact is not None and (not isinstance(exact, str) or len(exact) > 4000):
        raise ValueError("stdout_equals must be a string of at most 4000 characters")
    contains = data.get("stdout_contains", "")
    if not isinstance(contains, str) or len(contains) > 2000:
        raise ValueError("stdout_contains must be a string of at most 2000 characters")
    owner = data.get("owner")
    parallel_safe = data.get("parallel_safe", False)
    if not isinstance(parallel_safe, bool):
        raise ValueError("parallel_safe must be a boolean")
    return TeamCheck(
        identifier,
        argv,
        status,
        contains,
        exact,
        _identifier(owner, "owner") if owner is not None and owner != "" else "",
        parallel_safe,
    )


def normalize_team_path(value: Any) -> str:
    """任务分工使用仓库相对文件名，避免目录、命令或 Git 元数据作为写入范围。"""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 320
    ):
        raise ValueError("Writable path must be a non-empty relative filename")
    path = value.replace("\\", "/").removeprefix("./")
    parts = path.split("/")
    if _INVALID_PATH_CHARACTERS.search(path) or any(
        not part
        or part in {".", ".."}
        or part.endswith((".", " "))
        or part.casefold() in {".git", ".paicli"}
        or re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
        for part in parts
    ):
        raise ValueError(f"Invalid writable path: {value}")
    return path


class TeamBoard:
    """协调任务认领、版本发布和各下游提出的交付要求。"""

    def __init__(self, plan: TeamPlan):
        self.plan = plan
        self.tasks = {task.id: task for task in plan.tasks}
        self.status = {task.id: "pending" for task in plan.tasks}
        self.publications: dict[str, TeamPublication] = {}
        self.requirements: dict[str, TeamRequirement] = {}
        self._lock = RLock()

    def _dependency_ready(self, task_id: str) -> bool:
        if self.status[task_id] == "failed":
            return False
        task = self.tasks[task_id]
        return (
            task_id in self.publications
            if task.publishes
            else self.status[task_id] == "finished"
        )

    def ready(self) -> tuple[TeamTask, ...]:
        with self._lock:
            return tuple(
                task
                for task in self.plan.tasks
                if self.status[task.id] == "pending"
                and all(map(self._dependency_ready, task.needs))
            )

    def claim(self, task_id: str) -> bool:
        """认领的临界区只覆盖当前状态与依赖检查。"""
        with self._lock:
            if self.status[task_id] != "pending":
                return False
            if not all(map(self._dependency_ready, self.tasks[task_id].needs)):
                return False
            self.status[task_id] = "claimed"
            return True

    def publish(self, task_id: str, commit: str, text: str) -> TeamPublication:
        with self._lock:
            task = self.tasks[task_id]
            if self.status[task_id] != "claimed" or not task.publishes:
                raise ValueError(
                    "Only a claimed task with a planned handoff can publish"
                )
            summary = _string(text, "handoff summary", 1800, required=True)
            if not isinstance(commit, str) or not re.fullmatch(
                r"(?:[a-f0-9]{40}|[a-f0-9]{64})", commit
            ):
                raise ValueError("Published handoff needs a Git commit ID")
            current = self.publications.get(task_id)
            if (
                current is not None
                and current.commit == commit
                and current.text == summary
            ):
                return current
            publication = TeamPublication(
                commit, summary, current.version + 1 if current else 1
            )
            self.publications[task_id] = publication
            for request_id, requirement in self.requirements.items():
                if requirement.owner == task_id and requirement.accepted:
                    self.requirements[request_id] = replace(requirement, accepted=False)
            return publication

    def finish(self, task_id: str) -> None:
        with self._lock:
            if self.status[task_id] != "claimed":
                raise ValueError("Only a claimed task can finish")
            if self.tasks[task_id].publishes and task_id not in self.publications:
                raise ValueError(
                    f"Task {task_id} must publish its planned handoff before finishing"
                )
            self.status[task_id] = "finished"

    def fail(self, task_id: str) -> None:
        with self._lock:
            self.status[task_id] = "failed"

    def reopen(self, task_id: str) -> None:
        with self._lock:
            if self.status[task_id] != "finished":
                raise ValueError("Only a finished task can reopen")
            self.status[task_id] = "pending"

    def request(
        self, requester: str, owner: str, condition: str, check: TeamCheck | None = None
    ) -> TeamRequirement:
        detail = _string(condition, "requirement", 1200, required=True)
        with self._lock:
            if (
                requester == owner
                or requester not in self.tasks
                or owner not in self.tasks
            ):
                raise ValueError("A request requires two different Team task IDs")
            for current in self.requirements.values():
                if (current.requester, current.owner, current.condition) != (
                    requester,
                    owner,
                    detail,
                ):
                    continue
                if _check_behavior(current.check) == _check_behavior(check):
                    return current
            if len(self.requirements) >= 48:
                raise ValueError("The Team already recorded 48 requirements")
            sequence = len(self.requirements) + 1
            requirement_id = f"request-{sequence}"
            if check is not None:
                used_ids = {item.id for item in self.required_checks()}
                new_id = f"peer-check-{sequence}"
                while new_id in used_ids:
                    new_id = "p-" + new_id
                check = replace(check, id=new_id, owner=owner)
            item = TeamRequirement(requirement_id, requester, owner, detail, check)
            self.requirements[requirement_id] = item
            return item

    def respond(self, owner: str, request_id: str, message: str) -> TeamRequirement:
        response = _string(message, "requirement response", 1400, required=True)
        with self._lock:
            item = self.requirements.get(request_id)
            if item is None or item.owner != owner:
                raise ValueError("Only the receiving task can answer this request")
            version = self._publication_version(owner)
            if item.response == response and item.response_version == version:
                return item
            item = replace(
                item, response=response, response_version=version, accepted=False
            )
            self.requirements[request_id] = item
            return item

    def accept(self, requester: str, request_id: str) -> TeamRequirement:
        with self._lock:
            item = self.requirements.get(request_id)
            if item is None or item.requester != requester:
                raise ValueError("Only the requesting task can confirm this response")
            if not item.response or item.response_version != self._publication_version(
                item.owner
            ):
                raise ValueError(
                    "The current publication still needs a response for this request"
                )
            item = replace(item, accepted=True)
            self.requirements[request_id] = item
            return item

    def mailbox(self, task_id: str) -> tuple[TeamRequirement, ...]:
        with self._lock:
            return tuple(
                item
                for item in self.requirements.values()
                if task_id in (item.requester, item.owner)
            )

    def required_checks(self) -> tuple[TeamCheck, ...]:
        with self._lock:
            return (
                *self.plan.checks,
                *(item.check for item in self.requirements.values() if item.check),
            )

    def unresolved(self) -> tuple[TeamRequirement, ...]:
        with self._lock:
            return tuple(
                item
                for item in self.requirements.values()
                if item.check is None
                and (
                    not item.accepted
                    or item.response_version != self._publication_version(item.owner)
                )
            )

    def _publication_version(self, task_id: str) -> int:
        publication = self.publications.get(task_id)
        return publication.version if publication else 0


def _check_behavior(check: TeamCheck | None) -> tuple | None:
    if check is None:
        return None
    return (check.argv, check.expected_exit, check.stdout_contains, check.stdout_equals)


def _array(value: Any, name: str, limit: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} must be an array with at most {limit} entries")
    return value


def _string(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (required and not value.strip())
    ):
        raise ValueError(f"Invalid {name}; use a string of at most {limit} characters")
    return value.strip()


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Invalid {name}: use letters, digits, '-' and '_'")
    return value
