"""为每个评测任务构建干净、可复现的独立临时 Git 仓库。"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.types import EvalTask


def _run_git(cwd: Path, *args: str) -> str:
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=30,
    )
    return result.stdout


def materialize_repo(task: EvalTask, root: Path) -> Path:
    """将任务文件写入目录并初始化为带初始提交的 Git 仓库。"""
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")

    # 避免 pytest / Python 在仓库目录生成干扰 git diff 的缓存。
    gitignore = root / ".gitignore"
    ignore_lines = ["__pycache__/", "*.pyc", ".pytest_cache/", ".paicli/", ".eval_hidden/"]
    if gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8")
        missing = [line for line in ignore_lines if line not in existing]
        if missing:
            gitignore.write_text(
                existing.rstrip() + "\n" + "\n".join(missing) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    else:
        gitignore.write_text("\n".join(ignore_lines) + "\n", encoding="utf-8", newline="\n")

    _run_git(root, "init", "-b", "main")
    _run_git(root, "config", "user.name", "PaiCLI Eval")
    _run_git(root, "config", "user.email", "eval@paicli.local")
    _run_git(root, "config", "commit.gpgsign", "false")
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "-m", f"Initial fixture for {task.id}")

    # 如果任务配置了项目规范文档，尝试建立本地知识库索引（仅当依赖可用时）。
    if task.knowledge_paths:
        try:
            from paicli.rag.knowledge import KnowledgeIndex

            KnowledgeIndex(root).index(list(task.knowledge_paths))
        except Exception:
            # 离线或无 fastembed 依赖时允许降级，由词法/文件读取继续完成。
            pass

    return root


@contextmanager
def task_workspace(task: EvalTask):
    """在临时目录中实例化任务 Git 仓库并在退出后自动清理。"""
    with TemporaryDirectory(prefix=f"paicli-eval-{task.id}-") as tmp:
        root = Path(tmp).resolve()
        materialize_repo(task, root)
        yield root
