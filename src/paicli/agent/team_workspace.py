"""为 Team 提供独立工作区、可检出的交接提交和组合补丁。"""

from __future__ import annotations

import os
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


class TeamWorkspaceError(RuntimeError):
    """Git 工作区不能继续执行或提交到 Team。"""


@dataclass(frozen=True, slots=True)
class TeamChange:
    task_id: str
    paths: tuple[str, ...]
    patch: bytes


@dataclass(frozen=True, slots=True)
class TeamSource:
    """主会话当前的已跟踪改动，以及指定的新增文件。"""

    patch: bytes
    added: tuple[tuple[str, bytes, bool], ...]


def team_path(value: str) -> str:
    """把用户路径变成 Git 可用的工作区内文件名。"""
    if not isinstance(value, str) or any(
        char in value for char in ("\x00", "\n", "\r", ":")
    ):
        raise TeamWorkspaceError("Team paths must be repository-relative files")
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or not path.parts
        or any(part == ".." or part.casefold() == ".git" for part in path.parts)
    ):
        raise TeamWorkspaceError(f"Invalid Team path: {value!r}")
    return path.as_posix()


def _path_key(path: str) -> str:
    return path.casefold() if os.name == "nt" else path


class TeamWorkspace:
    """从当前 Git HEAD 创建隔离目录，最终只返回通过验收的补丁。"""

    def __init__(self, cwd: str):
        self._start = Path(cwd).resolve()
        self._temporary: TemporaryDirectory | None = None
        self._root: Path | None = None
        self._workers: dict[str, Path] = {}
        self._handoffs: dict[str, tuple[str, ...]] = {}
        self._seeded: dict[str, dict[str, str]] = {}
        self._worktrees: list[Path] = []
        self._candidate: Path | None = None
        self._candidate_tree: str | None = None
        self._integration_base: str | None = None
        self.candidate_revision: str | None = None
        self._next = 0

    def __enter__(self) -> TeamWorkspace:
        self.repository = Path(
            self._git(self._start, "rev-parse", "--show-toplevel").decode().strip()
        ).resolve()
        self.base = (
            self._git(self.repository, "rev-parse", "--verify", "HEAD^{commit}")
            .decode()
            .strip()
        )
        self.source_head = self.base
        self._temporary = TemporaryDirectory(prefix="paicli-team-")
        self._root = Path(self._temporary.name).resolve()
        (self._root / "no-hooks").mkdir()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        failures = []
        for path in reversed(self._worktrees.copy()):
            try:
                self._remove(path)
            except TeamWorkspaceError as error:
                failures.append(str(error))
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except OSError as error:
                failures.append(f"Team temporary workspace cleanup failed: {error}")
        self._root = None
        self._temporary = None
        if failures and exc is None:
            raise TeamWorkspaceError("; ".join(failures))

    def create(self, task_id: str) -> str:
        """创建或复用相同任务的工作区，保留等待消息前已完成的编辑。"""
        if not task_id:
            raise TeamWorkspaceError("Team task ID must be non-empty")
        if task_id not in self._workers:
            self._workers[task_id] = self._add("worker")
            self._seeded[task_id] = {}
        return str(self._workers[task_id])

    def source_state(
        self, lead_paths: tuple[str, ...], worker_paths: tuple[str, ...]
    ) -> TeamSource:
        """读取主工作区的内容，不改它的索引；拒绝覆盖已被主会话或用户改动的文件。"""
        if (
            self._git(self.repository, "rev-parse", "HEAD").decode().strip()
            != self.source_head
        ):
            raise TeamWorkspaceError("主工作区的 HEAD 已改变，请基于新版本重新委派")
        if self._git(self.repository, "ls-files", "--unmerged"):
            raise TeamWorkspaceError("主工作区存在未解决的 Git 冲突")
        changed = {
            path.casefold()
            for path in self._changed(self.repository, self.source_head).values()
        }
        tracked = (
            self._git(self.repository, "ls-files", "-z", "--", *worker_paths)
            .decode("utf-8", errors="surrogateescape")
            .split("\0")
            if worker_paths
            else []
        )
        for path in worker_paths:
            target = self.repository / path
            key = path.casefold()
            if (
                not target.resolve().is_relative_to(self._start)
                or target.is_symlink()
                or target.is_dir()
                or (target.exists() and path not in tracked)
                or key in changed
            ):
                raise TeamWorkspaceError(
                    f"Worker 路径已被修改、未纳入 Git 或超出范围：{path}"
                )

        extra = []
        for path in lead_paths:
            target = self.repository / path
            if (
                not target.resolve().is_relative_to(self._start)
                or target.is_symlink()
                or target.is_dir()
            ):
                raise TeamWorkspaceError(
                    f"主会话新增路径必须是工作目录内的普通文件：{path}"
                )
            if target.is_file() and self._untracked(self.repository, path):
                mode = target.stat().st_mode
                if not stat.S_ISREG(mode):
                    raise TeamWorkspaceError(f"主会话新增路径必须是普通文件：{path}")
                extra.append((path, target.read_bytes(), bool(mode & stat.S_IXUSR)))
        diff = self._git(
            self.repository,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            self.source_head,
            "--",
        )
        return TeamSource(diff, tuple(extra))

    def freeze_source(self, source: TeamSource) -> str:
        """把主会话状态固化为临时提交，供分派或组合验收使用。"""
        if not source.patch and not source.added:
            return self.source_head
        path = self._add("lead", self.source_head)
        try:
            if source.patch:
                self._apply_patch(path, source.patch, index=True)
            for name, content, executable in source.added:
                target = path / name
                if (
                    not target.resolve().is_relative_to(path)
                    or target.exists()
                    or target.is_symlink()
                ):
                    raise TeamWorkspaceError(f"无法在候选工作区引入主会话文件：{name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                if executable:
                    target.chmod(target.stat().st_mode | stat.S_IXUSR)
                self._git(path, "add", "--", name)
            tree = self._git(path, "write-tree").decode().strip()
            return self._commit(
                path, tree, "PaiCLI lead snapshot", parent=self.source_head
            )
        finally:
            self._remove(path)

    def apply_checked(
        self,
        patch: bytes,
        expected: TeamSource,
        lead_paths: tuple[str, ...],
        worker_paths: tuple[str, ...],
    ) -> None:
        """只有主工作区仍是实际验收的基线时，才应用经检查的 Worker 补丁。"""
        if self.source_state(lead_paths, worker_paths) != expected:
            raise TeamWorkspaceError("主会话文件已变更，此补丁需要重新组合验收")
        self._git(
            self.repository,
            "apply",
            "--check",
            "--binary",
            "--whitespace=nowarn",
            "-",
            stdin=patch,
        )
        self._apply_patch(self.repository, patch)

    def _apply_patch(self, cwd: Path, patch: bytes, *, index: bool = False) -> None:
        args = ["--index"] if index else []
        self._git(
            cwd, "apply", *args, "--binary", "--whitespace=nowarn", "-", stdin=patch
        )

    def handoff(self, producer: str, paths: tuple[str, ...]) -> str:
        """快照由发布者持有的文件，供其他 Worker 按明确提交读取。"""
        worker = self._worker(producer)
        self.capture(producer, paths)
        selected = tuple(team_path(path) for path in paths)
        if not any((worker / path).is_file() for path in selected):
            raise TeamWorkspaceError(
                "Publishing requires at least one existing owned file"
            )
        index_paths = self._git(worker, "ls-files", "-z", "--", *selected)
        # 也记录已经退回基础版本的路径，避免下游保留上一版的内容。
        self._git(
            worker, "add", "--all", "--", *self._present(worker, selected, index_paths)
        )
        tree = self._git(worker, "write-tree").decode().strip()
        commit = self._commit(worker, tree, f"PaiCLI team handoff: {producer}")
        self._handoffs[commit] = selected
        return commit

    def handoff_matches(
        self, task_id: str, paths: tuple[str, ...], revision: str
    ) -> bool:
        """比较已收集的实际内容，避免下游继续使用过期的早期发布。"""
        worker = self._worker(task_id)
        return not self._git(
            worker, "diff", "--cached", "--name-only", revision, "--", *paths
        )

    def seed(self, consumer: str, commit: str) -> None:
        """将上游已发布的文件带入下游副本，不重置下游的其他改动。"""
        worker = self._worker(consumer)
        paths = self._handoffs.get(commit)
        if not paths:
            raise TeamWorkspaceError("Unknown Team handoff commit")
        seeded = self._seeded[consumer]
        available = self._changed(worker, self.base)
        for path in paths:
            prior = seeded.get(path)
            if prior:
                differences = self._git(
                    worker, "diff", "--name-only", prior, "--", path
                )
                if differences or self._untracked(worker, path):
                    raise TeamWorkspaceError(
                        f"Worker has local changes to handoff file {path}"
                    )
            elif _path_key(path) in available:
                raise TeamWorkspaceError(
                    f"Worker has local changes to handoff file {path}"
                )
        version = self._git(
            worker, "ls-tree", "-z", "--name-only", "-r", commit, "--", *paths
        )
        index = self._git(worker, "ls-files", "-z", "--", *paths)
        if restore := self._present(worker, paths, version + index):
            self._git(
                worker,
                "restore",
                f"--source={commit}",
                "--staged",
                "--worktree",
                "--",
                *restore,
            )
        seeded.update(dict.fromkeys(paths, commit))

    @staticmethod
    def _present(worker: Path, choices: tuple[str, ...], git_paths: bytes) -> list[str]:
        listed = {
            _path_key(path)
            for path in git_paths.decode("utf-8", errors="surrogateescape").split("\0")
        }
        return [
            path
            for path in choices
            if _path_key(path) in listed or (worker / path).is_file()
        ]

    def capture(self, task_id: str, paths: tuple[str, ...]) -> TeamChange:
        """收集从基线到本任务完成状态的补丁，排除导入的上游文件。"""
        worker = self._worker(task_id)
        owned: dict[str, str] = {}
        for item in paths:
            path = team_path(item)
            key = _path_key(path)
            target = worker / path
            if (
                not target.resolve().is_relative_to(worker)
                or target.is_symlink()
                or target.is_dir()
            ):
                raise TeamWorkspaceError(
                    f"Team scope must identify a regular file: {path}"
                )
            if key in owned:
                raise TeamWorkspaceError(f"Duplicate file in Team ownership: {path}")
            owned[key] = path

        staged_elsewhere = self._changed(worker, self.base)
        for seeded, revision in self._seeded[task_id].items():
            key = _path_key(seeded)
            if key in owned:
                raise TeamWorkspaceError(
                    f"Cannot claim an upstream file already provided to this worker: {seeded}"
                )
            differs = self._git(worker, "diff", "--name-only", revision, "--", seeded)
            if not differs and not self._untracked(worker, seeded):
                staged_elsewhere.pop(key, None)
        outside = staged_elsewhere.keys() - owned.keys()
        if outside:
            names = ", ".join(sorted(staged_elsewhere[key] for key in outside))
            raise TeamWorkspaceError(f"Task {task_id} changed unowned files: {names}")

        if not (
            selected := tuple(
                owned[key] for key in owned.keys() & staged_elsewhere.keys()
            )
        ):
            return TeamChange(task_id, (), b"")
        selected = tuple(sorted(selected))
        self._git(worker, "add", "--all", "--", *selected)
        patch = self._git(
            worker,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            self.base,
            "--",
            *selected,
        )
        return TeamChange(task_id, selected, patch)

    def integrate(self, changes: list[TeamChange], *, base: str | None = None) -> str:
        """在从基线新建的工作区组合改动，供实际验收。"""
        base = base or self.base
        result = self._add("integration", base)
        owners: dict[str, str] = {}
        try:
            for change in changes:
                if not change.patch:
                    continue
                for path in change.paths:
                    key = _path_key(path)
                    if key in owners:
                        raise TeamWorkspaceError(
                            f"{change.task_id} and {owners[key]} both update {path}"
                        )
                    owners[key] = change.task_id
                self._apply_patch(result, change.patch, index=True)
            tree = self._git(result, "write-tree").decode().strip()
            commit = self._commit(
                result, tree, "PaiCLI team integrated result", parent=base
            )
            self._git(result, "reset", "--mixed", "--quiet", commit)
        except TeamWorkspaceError:
            self._remove(result)
            raise
        if self._candidate is not None:
            self._remove(self._candidate)
        self._candidate = result
        self._candidate_tree = tree
        self.candidate_revision = commit
        self._integration_base = base
        return str(result)

    def deliver(self) -> tuple[str, str]:
        """返回组合补丁。校验运行若动了代码，拒绝发布不同版本。"""
        candidate = self._candidate
        if (
            candidate is None
            or self._candidate_tree is None
            or self.candidate_revision is None
        ):
            raise TeamWorkspaceError("No Team integration candidate is available")
        tree = self._git(candidate, "write-tree").decode().strip()
        head = self._git(candidate, "rev-parse", "HEAD").decode().strip()
        if (
            head != self.candidate_revision
            or tree != self._candidate_tree
            or self._git(candidate, "diff", "--name-only")
        ):
            raise TeamWorkspaceError("Integration checks changed the candidate files")
        diff = self._git(
            candidate,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            self._integration_base or self.base,
        )
        return diff.decode("utf-8", errors="surrogateescape"), head

    @contextmanager
    def check_workspace(self):
        """每条检查从同一个组合提交开始，产生的文件不会影响下一条检查。"""
        if self.candidate_revision is None:
            raise TeamWorkspaceError("No Team integration candidate is available")
        workspace = self._add("check", self.candidate_revision)
        try:
            yield str(workspace)
        finally:
            self._remove(workspace)

    @contextmanager
    def worker_check_workspace(self, task_id: str, paths: tuple[str, ...]):
        """让本地检查读 Worker 的真实改动，同时隔离测试的写入。"""
        self.capture(task_id, paths)
        cwd = self._worker(task_id)
        tree = self._git(cwd, "write-tree").decode().strip()
        revision = self._commit(cwd, tree, "PaiCLI worker check")
        workspace = self._add("worker-check", revision)
        try:
            yield str(workspace), revision
        finally:
            self._remove(workspace)

    def _commit(
        self, cwd: Path, tree: str, message: str, *, parent: str | None = None
    ) -> str:
        return (
            self._git(
                cwd,
                "-c",
                "user.name=PaiCLI Team",
                "-c",
                "user.email=team@paicli.local",
                "-c",
                "commit.gpgsign=false",
                "commit-tree",
                tree,
                "-p",
                parent or self.base,
                "-m",
                message,
            )
            .decode()
            .strip()
        )

    def _worker(self, task_id: str) -> Path:
        if (worker := self._workers.get(task_id)) is None:
            raise TeamWorkspaceError(f"Unknown Team worktree: {task_id}")
        return worker

    def _add(self, label: str, revision: str | None = None) -> Path:
        if self._root is None:
            raise TeamWorkspaceError("Team workspace must be entered before use")
        self._next += 1
        path = self._root / f"{label}-{self._next}"
        try:
            self._git(
                self.repository,
                "worktree",
                "add",
                "--detach",
                str(path),
                revision or self.base,
            )
        except TeamWorkspaceError:
            if path.exists():
                try:
                    self._git(
                        self.repository, "worktree", "remove", "--force", str(path)
                    )
                except TeamWorkspaceError:
                    pass
            raise
        self._worktrees.append(path)
        return path

    def _remove(self, path: Path) -> None:
        if self._root is None or path.resolve().parent != self._root:
            raise TeamWorkspaceError(
                "Refusing to remove a directory outside the Team workspace"
            )
        self._git(self.repository, "worktree", "remove", "--force", str(path))
        self._worktrees.remove(path)

    def _untracked(self, cwd: Path, *paths: str) -> bytes:
        return self._git(
            cwd, "ls-files", "--others", "--exclude-standard", "-z", "--", *paths
        )

    def _changed(self, cwd: Path, revision: str) -> dict[str, str]:
        tracked = self._git(
            cwd, "diff", "--name-only", "--no-renames", "-z", revision, "--"
        )
        paths = (tracked + self._untracked(cwd)).decode(
            "utf-8", errors="surrogateescape"
        )
        return {_path_key(path): path for path in paths.split("\0") if path}

    def _git(self, cwd: Path, *args: str, stdin: bytes | None = None) -> bytes:
        command = ["git", "--literal-pathspecs"]
        if self._root is not None:
            command.extend(("-c", f"core.hooksPath={self._root / 'no-hooks'}"))
        environment = os.environ.copy()
        for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            environment.pop(key, None)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        try:
            result = subprocess.run(
                [*command, *args],
                cwd=cwd,
                input=stdin,
                capture_output=True,
                timeout=90,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TeamWorkspaceError(f"Git {args[0]} did not run: {error}") from error
        if result.returncode:
            detail = (
                (result.stderr or result.stdout)
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise TeamWorkspaceError(f"Git {args[0]} failed: {detail[:1800]}")
        return result.stdout
