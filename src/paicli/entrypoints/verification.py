"""面向用户的 pytest 基线和验收命令，无需模型参与。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from paicli.harness.verification import create_baseline, verify_run

verify_app = typer.Typer(help="Record a failing pytest baseline and independently verify a repair.")


def _show(report: dict, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"{report['status']}: {report['reason']}")
        typer.echo(f"Run: {report['run_id']}")
        typer.echo(f"Report: {report['report_path']}")
        if "counts" in report:
            typer.echo(f"Tests: {report['total']}; results: {report['counts']}")
    if report["status"] not in {"reproduced", "accepted"}:
        raise typer.Exit(1 if report["status"] == "rejected" else 2)


@verify_app.command("baseline")
def baseline(
    targets: Annotated[
        list[str], typer.Argument(help="Test files, directories, or pytest node selections")
    ],
    cwd: Annotated[Path, typer.Option("--cwd", help="Python project directory")] = Path("."),
    python: Annotated[
        str | None, typer.Option("--python", help="Target Python interpreter path")
    ] = None,
    timeout: Annotated[float, typer.Option("--timeout", help="Test timeout in seconds")] = 60,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the structured verification report")
    ] = False,
) -> None:
    """固定验收文件并记录已有断言失败。"""
    try:
        report = asyncio.run(create_baseline(cwd, targets, python, timeout))
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        typer.echo(f"Verification error: {exc}", err=True)
        raise typer.Exit(2) from exc
    _show(report, json_output)


@verify_app.command("check")
def check(
    run_id: Annotated[str, typer.Argument(help="Run ID returned by baseline")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the structured verification report")
    ] = False,
) -> None:
    """修改实现代码后，按原始基线契约重新验收。"""
    try:
        report = asyncio.run(verify_run(run_id))
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        typer.echo(f"Verification error: {exc}", err=True)
        raise typer.Exit(2) from exc
    _show(report, json_output)
