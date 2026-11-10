"""命令行入口：解析参数、加载配置并启动交互、单次请求或专用子命令。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from paicli import __version__
from paicli.agent import QueryEngine
from paicli.bootstrap import build_tool_registry
from paicli.config import get_config_paths, load_config
from paicli.entrypoints.knowledge import knowledge_app
from paicli.entrypoints.repl import start_repl
from paicli.entrypoints.runs import runs_app
from paicli.entrypoints.verification import verify_app
from paicli.harness.evidence import load_run
from paicli.llm import create_llm_client
from paicli.mcp import load_mcp_server_specs, serve_http, serve_stdio, write_chrome_devtools_config
from paicli.runtime import RuntimeApiServer
from paicli.runtime.api import runtime_api_key
from paicli.tools import ToolRegistry, get_builtin_tools

app = typer.Typer(
    name="paicli",
    help="PaiCLI — Terminal AI Agent in Python (Claude Code alternative)",
    invoke_without_command=True,
    no_args_is_help=False,
)
mcp_app = typer.Typer(help="MCP server management")
app.add_typer(mcp_app, name="mcp")
app.add_typer(verify_app, name="verify")
app.add_typer(runs_app, name="runs")
app.add_typer(knowledge_app, name="knowledge")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"paicli {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    # 交互方式
    prompt: Annotated[
        str | None,
        typer.Option("-p", "--prompt", help="Single prompt (non-interactive mode)"),
    ] = None,
    # 模型配置
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="LLM API key (overrides env/config)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("-m", "--model", help="Override LLM model name"),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Override LLM provider (e.g. deepseek, glm)"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override LLM API base URL"),
    ] = None,
    # 输出渲染
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Use plain text rendering (no rich formatting)"),
    ] = False,
    # Agent 执行模式
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Agent mode: react, plan, or team"),
    ] = None,
    worker_mode: Annotated[
        str,
        typer.Option("--worker-mode", help="Sub-Agent worker mode in team runs: react or plan"),
    ] = "react",
    # 输出格式
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit result, usage, and cost as JSON (single-prompt only)"),
    ] = False,
    # 目标工作目录
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory (default: current dir)"),
    ] = None,
    # 版本信息
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version"),
    ] = False,
) -> None:
    """在指定工作目录运行终端 Agent。

    默认进入交互模式；使用 -p 执行单次任务，也可通过子命令调试、验收或启动服务。
    """
    _ = version
    if ctx.invoked_subcommand is not None:
        return

    root = (cwd or Path.cwd()).resolve()

    # 将显式命令行参数整理为配置覆盖项。
    overrides: dict = {}
    llm_overrides: dict = {}
    if api_key is not None:
        llm_overrides["api_key"] = api_key
    if provider is not None:
        llm_overrides["provider"] = provider
    if model is not None:
        llm_overrides["model"] = model
    if base_url is not None:
        llm_overrides["base_url"] = base_url
    if llm_overrides:
        overrides["llm"] = llm_overrides
    if plain:
        overrides["render_mode"] = "plain"

    config = load_config(project_root=root, overrides=overrides)
    if plain:
        config.render_mode = "plain"

    if prompt is not None:
        selected_mode = (mode or config.prompt.agent_mode or "react").lower()
        if selected_mode not in {"react", "plan", "team"}:
            raise typer.BadParameter("mode must be react, plan, or team", param_hint="--mode")
        if worker_mode not in {"react", "plan"}:
            raise typer.BadParameter(
                "worker-mode must be react or plan", param_hint="--worker-mode"
            )
        asyncio.run(
            _run_prompt(
                prompt,
                str(root),
                config,
                mode=selected_mode,
                worker_mode=worker_mode,
                json_output=json_output,
            )
        )
    else:
        asyncio.run(start_repl(str(root), config))


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


@app.command("debug")
def debug_command(
    prompt: Annotated[str, typer.Option("-p", "--prompt", help="Bug and intended behavior")] = "",
    tests: Annotated[
        list[str] | None, typer.Option("--tests", help="Existing pytest target; repeatable")
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Target Python project")] = None,
    max_attempts: Annotated[int, typer.Option("--max-attempts", min=1, max=5)] = 2,
    timeout: Annotated[float, typer.Option("--timeout", min=0.001, max=3600)] = 60,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    resume: Annotated[str | None, typer.Option("--resume", help="Continue a debug run ID")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", min=1)] = None,
    max_tools: Annotated[int | None, typer.Option("--max-tools", min=1)] = None,
    max_models: Annotated[int | None, typer.Option("--max-models", min=1)] = None,
    knowledge: Annotated[
        bool, typer.Option("--knowledge", help="Retrieve indexed project reference docs")
    ] = False,
) -> None:
    """执行带独立 pytest 验收的调试任务，支持按需角色协作及阶段恢复。"""
    try:
        if not resume and (not prompt.strip() or not tests):
            raise ValueError("Provide --prompt and --tests, or --resume RUN_ID")
        if resume and tests:
            raise ValueError("Resume reuses the original tests; do not pass --tests")
        root = (cwd or (Path(load_run(resume).manifest["cwd"]) if resume else Path.cwd())).resolve()
        config = load_config(project_root=root)
        if not config.llm.api_key:
            raise ValueError("Debug requires the configured model API key")
        result = asyncio.run(
            _run_debug(
                prompt,
                root,
                config,
                tests or [],
                max_attempts,
                timeout,
                resume=resume,
                max_tokens=max_tokens,
                max_tools=max_tools,
                max_models=max_models,
                use_knowledge=knowledge,
            )
        )
    except KeyboardInterrupt:
        typer.echo("Debug cancelled. Use `paicli runs list` to find its checkpoint.", err=True)
        raise typer.Exit(130) from None
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Debug error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result, ensure_ascii=False) if json_output else result["text"])
    if result.get("status") != "completed":
        raise typer.Exit(1)


async def _run_debug(prompt, root, config, tests, max_attempts, timeout, **options) -> dict:
    """组装调试引擎，消费事件并收集最终状态，供命令行展示与退出码判断。"""
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=create_llm_client(config.llm),
        tool_registry=registry,
        config=config,
        cwd=str(root),
    )
    text, result = "", {"status": "failed"}
    async for event in engine.debug(
        prompt, targets=tests, max_attempts=max_attempts, timeout=timeout, **options
    ):
        if event["type"] == "debug_started":
            typer.echo(f"Debug run: {event['run_id']}", err=True)
        elif event["type"] == "text_delta":
            text += event["text"]
        elif event["type"] == "done":
            result = {key: value for key, value in event.items() if key != "messages"}
    return {**result, "text": text}


@app.command("doctor")
def doctor(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """检查 Python、uv、Node、模型凭据和配置等启动条件。"""
    root = (cwd or Path.cwd()).resolve()
    config = load_config(project_root=root)
    checks = {
        "python": sys.version.split()[0],
        "uv": shutil.which("uv") or "missing",
        "node": _version_of("node"),
        "npx": shutil.which("npx") or "missing",
        "rg": shutil.which("rg") or "missing",
        "api_key": "configured" if config.llm.api_key else "missing",
        "provider": config.llm.provider,
        "model": config.llm.model,
        "cwd": str(root),
        "config_paths": [str(path) for path in get_config_paths(root)],
    }
    console.print_json(json.dumps(checks, ensure_ascii=False))


@app.command("serve")
def runtime_serve(
    http: Annotated[bool, typer.Option("--http", help="Serve Runtime API over HTTP")] = True,
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 8080,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Runtime API key. Defaults to PAICLI_RUNTIME_API_KEY."),
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """启动提供持久化任务能力的 HTTP 运行服务。"""
    _ = http
    root = (cwd or Path.cwd()).resolve()
    config = load_config(project_root=root)
    try:
        key = runtime_api_key(api_key)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    RuntimeApiServer(cwd=str(root), config=config, api_key=key, port=port).serve_forever()


@app.command("worker")
def task_worker(
    workers: Annotated[int, typer.Option("--workers", help="Concurrent task workers")] = 2,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """启动持久化后台任务消费者，无需同时开放 HTTP 接口。"""
    if workers < 1:
        raise typer.BadParameter("workers must be at least 1")
    root = (cwd or Path.cwd()).resolve()
    config = load_config(project_root=root)
    RuntimeApiServer(
        cwd=str(root),
        config=config,
        api_key="worker-only-not-exposed",
        workers=workers,
    ).work_forever()


# ---------------------------------------------------------------------------
# MCP 子命令
# ---------------------------------------------------------------------------


@mcp_app.command("serve")
def mcp_serve(
    transport: Annotated[
        str,
        typer.Option("--transport", help="Transport type: stdio or http"),
    ] = "stdio",
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 3000,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """通过 MCP 协议向外暴露当前内置工具。"""
    root = str((cwd or Path.cwd()).resolve())
    if transport == "http":
        serve_http(port=port, cwd=root)
    elif transport == "stdio":
        asyncio.run(serve_stdio(cwd=root))
    else:
        raise typer.BadParameter("transport must be stdio or http")


@mcp_app.command("init-chrome")
def mcp_init_chrome(
    scope: Annotated[
        str,
        typer.Option("--scope", help="Config scope: user or project"),
    ] = "project",
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    browser_url: Annotated[
        str | None,
        typer.Option("--browser-url", help="Connect to an existing Chrome remote debugging URL"),
    ] = None,
    headless: Annotated[bool, typer.Option("--headless", help="Start Chrome headless")] = False,
    slim: Annotated[bool, typer.Option("--slim", help="Use Chrome DevTools slim mode")] = False,
) -> None:
    """向项目级或用户级配置写入 Chrome DevTools MCP 连接信息。"""
    if scope not in {"user", "project"}:
        raise typer.BadParameter("scope must be user or project")
    root = None if scope == "user" else (cwd or Path.cwd()).resolve()
    path = write_chrome_devtools_config(
        scope_root=root,
        browser_url=browser_url,
        headless=headless,
        slim=slim,
    )
    typer.echo(f"Wrote Chrome DevTools MCP config to {path}")


@mcp_app.command("list")
def mcp_list(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """列出当前配置的 MCP 服务。"""
    root = (cwd or Path.cwd()).resolve()
    specs = load_mcp_server_specs(root)
    if not specs:
        typer.echo("No MCP servers configured.")
        return
    for spec in specs.values():
        target = spec.url or f"{spec.command} {' '.join(spec.args)}".strip()
        typer.echo(f"{spec.name}\t{spec.type}\t{target}")


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


async def _run_prompt(
    prompt: str,
    cwd: str,
    config,
    *,
    mode: str = "react",
    worker_mode: str = "react",
    json_output: bool = False,
) -> None:
    """组装单次请求引擎，按模式执行并输出文字、过程或 JSON 结果。"""
    config.render_mode = "plain"
    if not config.llm.api_key:
        typer.echo(
            "Fatal error: LLM API key is not configured. Set it with --api-key, "
            "via the PAICLI_API_KEY environment variable, or in "
            "~/.paicli/config.json or .paicli/config.json.",
            err=True,
        )
        raise typer.Exit(1)
    registry, manager = await build_tool_registry(config=config, cwd=cwd)
    if manager and manager.last_errors:
        for name, error in manager.last_errors.items():
            typer.echo(f"MCP server {name} failed to load: {error}", err=True)
    engine = QueryEngine(
        llm_client=create_llm_client(config.llm),
        tool_registry=registry,
        config=config,
        cwd=cwd,
    )
    try:
        if mode == "plan":
            result = await engine.plan_complete_async(prompt)
        elif mode == "team":
            result = await engine.team_complete_async(prompt, worker_mode=worker_mode)
        else:
            result = await engine.ask_complete_async(prompt)
    except Exception as exc:  # noqa: BLE001 - 命令行统一展示模型或配置错误。
        typer.echo(f"Fatal error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "text": result.text,
                    "mode": mode,
                    "worker_mode": worker_mode if mode == "team" else None,
                    "turns": result.turns,
                    "total_tokens": result.total_tokens,
                    "usage": result.usage.to_dict(),
                    "cost": result.cost,
                    "run_id": result.run_id,
                    "evidence_dir": result.evidence_dir,
                },
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(result.text)


def _version_of(command: str) -> str:
    if not shutil.which(command):
        return "missing"
    try:
        result = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return (result.stdout or result.stderr).strip() or "unknown"
