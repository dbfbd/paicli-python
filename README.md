# PaiCLI

基于[开源 PaiCLI](https://github.com/itwanger/PaiCLI-Python)二次开发的本地 Python 调试工具。使用 LangGraph 协调缺陷复现、调查和修复，并通过独立 pytest 验收补丁。支持带来源的项目文档检索和阶段恢复。

## 安装与使用

需要 Python 3.11 或以上版本以及 uv。以下命令在仓库根目录运行，环境变量示例使用 PowerShell。

```powershell
uv sync --locked --extra dev
uv run paicli --help
$env:DEEPSEEK_API_KEY = 'your-key'
uv run paicli
```

指定目标项目和已有失败的 pytest 用例。

```powershell
uv run paicli debug --cwd ../your-project --tests tests/test_example.py -p '描述缺陷和预期行为'
uv run paicli debug --resume RUN_ID
```

目标项目已有 `.venv` 时，Debug 优先使用该环境运行 pytest。请在该环境安装 pytest 和目标项目的依赖。

需要项目文档检索时，运行 `uv sync --locked --extra dev --extra retrieval`，再用 `uv run paicli knowledge index docs --cwd ../your-project` 建索引。启动 Debug 时加 `--knowledge`。

## 许可证

[MIT](LICENSE)
