"""检查 Debug 阶段记录并请求取消，无需调用模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from paicli.harness.evidence import load_run

runs_app = typer.Typer(help="Inspect debug runs and request cancellation.")


def _show(manifest):
    return {
        name: manifest.get(name)
        for name in (
            "run_id",
            "cwd",
            "status",
            "created_at",
            "finished_at",
            "active_node",
            "checkpoint",
            "resume_next",
            "execution",
        )
    }


@runs_app.command("list")
def list_runs(limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 10):
    root = Path.home() / ".paicli" / "runs"
    manifests = []
    if root.exists():
        for path in root.glob("*/run.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("mode") == "debug":
                    manifests.append(item)
            except (OSError, ValueError):
                continue
    manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    typer.echo(
        json.dumps([_show(item) for item in manifests[:limit]], ensure_ascii=False, indent=2)
    )


@runs_app.command("show")
def show_run(run_id: str):
    try:
        run = load_run(run_id)
        if run.manifest["mode"] != "debug":
            raise ValueError("This command requires a debug run")
        typer.echo(json.dumps(_show(run.manifest), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@runs_app.command("cancel")
def cancel_run(run_id: str):
    try:
        run = load_run(run_id)
        if run.manifest["mode"] != "debug" or run.manifest["status"] == "completed":
            raise ValueError("Only unfinished debug executions can be cancelled")
        (run.directory / "cancel.requested").touch()
        typer.echo(json.dumps({"run_id": run.run_id, "cancel_requested": True}))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
