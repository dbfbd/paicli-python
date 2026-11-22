from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    ".paicli",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pycache__",
}


@dataclass(slots=True)
class SnapshotRecord:
    id: str
    phase: str
    created_at: str
    path: Path


class SnapshotService:
    def __init__(self, project_root: str | Path, *, max_snapshots: int = 20):
        self.project_root = Path(project_root).resolve()
        self.max_snapshots = max(1, int(max_snapshots))
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()[:16]
        self.root = Path.home() / ".paicli" / "snapshots" / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def create(self, phase: str) -> SnapshotRecord:
        snapshot_id = f"{phase}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        target = self.root / snapshot_id
        target.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.project_root, target)
        record = SnapshotRecord(
            id=snapshot_id,
            phase=phase,
            created_at=datetime.now(UTC).isoformat(),
            path=target,
        )
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": record.id,
                        "phase": record.phase,
                        "created_at": record.created_at,
                        "path": str(record.path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._prune()
        return record

    def _prune(self) -> None:
        """保留最近 max_snapshots 份快照，自动回收历史目录避免磁盘无限增长。"""
        if not self.index_path.exists():
            return
        lines = [
            line for line in self.index_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if len(lines) <= self.max_snapshots:
            return
        expired, kept = lines[: len(lines) - self.max_snapshots], lines[-self.max_snapshots :]
        for raw in expired:
            try:
                item = json.loads(raw)
                old_path = Path(item["path"]).resolve()
                if old_path.parent == self.root.resolve() and old_path.exists():
                    shutil.rmtree(old_path, ignore_errors=True)
            except (OSError, ValueError, KeyError):
                continue
        self.index_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def list(self, limit: int = 20) -> list[SnapshotRecord]:
        if not self.index_path.exists():
            return []
        records = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                SnapshotRecord(
                    id=item["id"],
                    phase=item["phase"],
                    created_at=item["created_at"],
                    path=Path(item["path"]),
                )
            )
        return records[-limit:][::-1]

    def restore(self, snapshot_ref: str) -> SnapshotRecord:
        records = self.list(limit=200)
        record = None
        if snapshot_ref.isdigit():
            index = int(snapshot_ref) - 1
            if 0 <= index < len(records):
                record = records[index]
        else:
            record = next((item for item in records if item.id == snapshot_ref), None)
        if not record:
            raise ValueError(f"snapshot not found: {snapshot_ref}")
        self.create("pre-restore")
        self._restore_tree(record.path, self.project_root)
        return record

    def clean(self) -> int:
        count = len(self.list(limit=10_000))
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return count

    def _copy_tree(self, source: Path, target: Path) -> None:
        for item in source.iterdir():
            if _skip(item):
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, ignore=_ignore)
            elif item.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)

    def _restore_tree(self, source: Path, target: Path) -> None:
        for item in target.iterdir():
            if _skip(item):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self._copy_tree(source, target)


def _skip(path: Path) -> bool:
    return path.name in SKIP_DIRS


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_DIRS}
