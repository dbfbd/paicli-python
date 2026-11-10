"""显式创建和检索项目文档索引，全程无需聊天模型调用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from paicli.rag.knowledge import KnowledgeIndex

knowledge_app = typer.Typer(help="Index and retrieve project specs and runbooks locally.")


@knowledge_app.command("index")
def index_docs(
    sources: Annotated[
        list[str], typer.Argument(help="Project-relative Markdown/text files or directories")
    ],
    cwd: Annotated[Path, typer.Option("--cwd")] = Path("."),
):
    """原子替换当前项目索引；首次使用需要下载本地嵌入模型。"""
    try:
        result = KnowledgeIndex(cwd).index(sources)
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 - 在命令行报告依赖、模型及资料错误。
        typer.echo(f"Knowledge indexing failed: {exc}", err=True)
        raise typer.Exit(2) from exc


@knowledge_app.command("search")
def search_docs(
    query: str,
    cwd: Annotated[Path, typer.Option("--cwd")] = Path("."),
    mode: Annotated[str, typer.Option("--mode", help="hybrid, lexical or semantic")] = "hybrid",
    k: Annotated[int, typer.Option("--k", min=1, max=10)] = 3,
):
    """返回数量受限且经过版本核对的来源；混合检索使用本地缓存模型。"""
    try:
        result = KnowledgeIndex(cwd).search(query, mode=mode, k=k)
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 - 模型缓存异常使用简明错误提示。
        typer.echo(f"Knowledge retrieval failed: {exc}", err=True)
        raise typer.Exit(2) from exc
