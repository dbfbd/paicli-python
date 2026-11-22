"""Suite C：多模块并行开发、跨模块迁移与并行只读审计（Team / Collaboration 核心试金石）。"""

from __future__ import annotations

import re
from typing import Any

from evals.types import EvalTask

# ---------------------------------------------------------------------------
# C1：3 个独立格式解析器 + 统一注册表与批处理（并行功能实现）
# ---------------------------------------------------------------------------

_C1_FILES: dict[str, str] = {
    "ingest/__init__.py": "",
    "ingest/models.py": '''"""Shared data contracts for multi-format record ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParseIssue:
    line: int
    code: str
    message: str


@dataclass(frozen=True)
class Record:
    source: str
    line: int
    data: dict[str, Any]


@dataclass
class ParseResult:
    records: list[Record] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def coerce_scalar(raw: str) -> Any:
    """Coerce trimmed scalar text to None / bool / int / float / str."""
    text = raw.strip()
    if text == "" or text.lower() in {"null", "none"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if text.startswith(("0", "-0")) and len(text) > 1 and not text.startswith(("0.", "-0.")):
            return text
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
''',
    "ingest/parsers/__init__.py": "",
    "ingest/parsers/csv_parser.py": '''"""CSV parser module (to be implemented)."""

from __future__ import annotations

from ingest.models import ParseResult


def parse_csv(text: str, *, source: str = "memory.csv", strict: bool = False) -> ParseResult:
    """Parse CSV text with header row, comment skipping (#), and scalar coercion.

    Requirements:
    - Skip blank lines and comment lines starting with '#' (even before header).
    - First non-empty, non-comment line is the header; normalize headers with strip().
    - Reject duplicate or empty header names with ParseIssue(code="invalid_header").
    - Support standard RFC-4180 quoting (commas and escaped quotes inside quoted fields).
    - Track 1-based physical line numbers for each data record and issue.
    - If a row has wrong column count: record ParseIssue(line, "column_mismatch", ...).
      In strict mode, stop and return issues with empty records; otherwise skip bad row.
    - Coerce unquoted/string field values via coerce_scalar().
    """
    raise NotImplementedError("implement parse_csv")
''',
    "ingest/parsers/jsonl_parser.py": '''"""JSONL parser module (to be implemented)."""

from __future__ import annotations

from ingest.models import ParseResult


def parse_jsonl(text: str, *, source: str = "memory.jsonl", strict: bool = False) -> ParseResult:
    """Parse newline-delimited JSON objects.

    Requirements:
    - Track 1-based line numbers.
    - Skip blank lines and lines starting with '//' (comment lines).
    - Each non-blank line must decode to a JSON object (dict).
      * Invalid JSON -> ParseIssue(line, "invalid_json", ...).
      * Valid JSON that is not a dict (e.g. list/int/str) -> ParseIssue(line, "not_object", ...).
    - Optional key normalization: strip surrounding whitespace from string keys; reject empty key
      with ParseIssue(line, "empty_key", ...).
    - In strict mode, return empty records along with collected issues if any issue occurs.
    """
    raise NotImplementedError("implement parse_jsonl")
''',
    "ingest/parsers/kv_parser.py": '''"""Key=Value block parser module (to be implemented)."""

from __future__ import annotations

from ingest.models import ParseResult


def parse_kv(text: str, *, source: str = "memory.kv", strict: bool = False) -> ParseResult:
    """Parse blank-line-separated key=value record blocks.

    Requirements:
    - Blocks are separated by one or more blank lines.
    - Ignore lines starting with '#' or ';'.
    - Each active line must contain '=' splitting into key and value (first '=' only).
      * Missing '=' or empty key -> ParseIssue(line, "malformed_line", ...).
      * Duplicate key in the same block -> ParseIssue(line, "duplicate_key", ...).
    - Record.line is the 1-based line number of the first valid entry in that block.
    - Coerce values using coerce_scalar(), except values wrapped in matching double quotes
      ("...") which must remain unescaped str without scalar coercion.
    - If a block has any issue: in strict mode return empty records + issues;
      in non-strict mode drop only the malformed line/duplicate and keep valid entries
      (drop empty blocks).
    """
    raise NotImplementedError("implement parse_kv")
''',
    "ingest/registry.py": '''"""Unified format registry & batch ingestion pipeline (to be implemented)."""

from __future__ import annotations

from pathlib import Path

from ingest.models import ParseResult


def parse_file(path: Path, *, strict: bool = False) -> ParseResult:
    """Auto-detect parser by file suffix (.csv, .jsonl, .kv) and parse UTF-8 content.

    Unknown suffix should raise ValueError.
    """
    raise NotImplementedError("implement parse_file")


def ingest_directory(
    directory: Path,
    *,
    strict: bool = False,
    dedupe_key: str | None = None,
) -> dict:
    """Parse all supported files (.csv, .jsonl, .kv) in sorted filename order.

    Returns dict:
      {
        "records": list[Record],
        "issues": list[ParseIssue],
        "files_processed": list[str],   # relative posix names in sorted order
        "duplicates_skipped": int,
      }
    When dedupe_key is given, skip records whose data.get(dedupe_key) was already seen
    (only when value is not None), keeping the first occurrence and incrementing duplicates_skipped.
    """
    raise NotImplementedError("implement ingest_directory")
''',
    "tests/test_csv_parser.py": r'''from ingest.parsers.csv_parser import parse_csv


def test_csv_basic_and_coercion():
    text = """# sample export
id, name , active, score, note
1, Alice, true, 98.5, null
2, "Bob, Jr.", false, 0, "hello"
"""
    res = parse_csv(text, source="users.csv")
    assert res.ok
    assert len(res.records) == 2
    assert res.records[0].line == 3
    assert res.records[0].data == {
        "id": 1,
        "name": "Alice",
        "active": True,
        "score": 98.5,
        "note": None,
    }
    assert res.records[1].line == 4
    assert res.records[1].data["name"] == "Bob, Jr."
    assert res.records[1].data["active"] is False


def test_csv_column_mismatch_lenient_vs_strict():
    text = "a,b\n1,2\n3,4,5\n6,7\n"
    lenient = parse_csv(text, strict=False)
    assert len(lenient.records) == 2
    assert len(lenient.issues) == 1
    assert lenient.issues[0].line == 3
    assert lenient.issues[0].code == "column_mismatch"

    strict = parse_csv(text, strict=True)
    assert strict.records == []
    assert len(strict.issues) == 1


def test_csv_duplicate_header_rejected():
    res = parse_csv("id,id\n1,2\n")
    assert not res.ok
    assert res.issues[0].code == "invalid_header"
''',
    "tests/test_jsonl_parser.py": r'''from ingest.parsers.jsonl_parser import parse_jsonl


def test_jsonl_comments_and_valid_objects():
    text = """// header comment
{"id": 1, "name": "alpha"}

{" id ": 2, "ok": true}
"""
    res = parse_jsonl(text, source="events.jsonl")
    assert res.ok
    assert len(res.records) == 2
    assert res.records[0].line == 2
    assert res.records[1].line == 4
    assert res.records[1].data == {"id": 2, "ok": True}


def test_jsonl_errors_lenient_and_strict():
    text = '{"a": 1}\nnot-json\n[1, 2, 3]\n{"": 9}\n{"b": 2}\n'
    lenient = parse_jsonl(text, strict=False)
    assert [r.data for r in lenient.records] == [{"a": 1}, {"b": 2}]
    assert [(i.line, i.code) for i in lenient.issues] == [
        (2, "invalid_json"),
        (3, "not_object"),
        (4, "empty_key"),
    ]

    strict = parse_jsonl(text, strict=True)
    assert strict.records == []
    assert len(strict.issues) == 3
''',
    "tests/test_kv_parser.py": r'''from ingest.parsers.kv_parser import parse_kv


def test_kv_blocks_quotes_and_coercion():
    text = """# config records
id = 007
enabled = true
label = "123"
url = https://example.com?a=1&b=2

; second block
id = 42
ratio = 0.75
empty = none
"""
    res = parse_kv(text, source="items.kv")
    assert res.ok
    assert len(res.records) == 2
    assert res.records[0].line == 2
    assert res.records[0].data == {
        "id": "007",
        "enabled": True,
        "label": "123",
        "url": "https://example.com?a=1&b=2",
    }
    assert res.records[1].line == 8
    assert res.records[1].data == {"id": 42, "ratio": 0.75, "empty": None}


def test_kv_malformed_and_duplicate_keys():
    text = "k1 = 1\nbadline\nk1 = 2\n\nk2 = ok\n"
    lenient = parse_kv(text, strict=False)
    assert len(lenient.records) == 2
    assert lenient.records[0].data == {"k1": 1}
    assert [(i.line, i.code) for i in lenient.issues] == [
        (2, "malformed_line"),
        (3, "duplicate_key"),
    ]

    strict = parse_kv(text, strict=True)
    assert strict.records == []
    assert len(strict.issues) == 2
''',
    "tests/test_integration.py": r'''import pytest

from ingest.registry import ingest_directory, parse_file


def test_ingest_directory_combines_and_dedupes(tmp_path):
    (tmp_path / "01_seed.csv").write_text("id,name\n1,alice\n2,bob\n", encoding="utf-8")
    (tmp_path / "02_extra.jsonl").write_text(
        '{"id": 2, "name": "bob-dup"}\n{"id": 3, "name": "carol"}\n',
        encoding="utf-8",
    )
    (tmp_path / "03_more.kv").write_text("id = 3\nname = carol-dup\n\nid = 4\nname = dave\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("ignored", encoding="utf-8")

    summary = ingest_directory(tmp_path, dedupe_key="id")
    assert summary["files_processed"] == ["01_seed.csv", "02_extra.jsonl", "03_more.kv"]
    assert summary["duplicates_skipped"] == 2
    assert [r.data["id"] for r in summary["records"]] == [1, 2, 3, 4]
    assert summary["records"][1].data["name"] == "bob"
    assert summary["issues"] == []


def test_parse_file_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "data.xml"
    bad.write_text("<x/>", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_file(bad)
''',
}

C1_PARSERS = EvalTask(
    id="c1_multi_format_parsers",
    suite="C",
    title="并行多模块实现：3 个独立格式解析器（CSV/JSONL/KV）+ 统一注册批处理",
    description="3 个互不重叠解析模块 + 1 个集成注册表，非常适合并行委派与统一验收。",
    prompt=(
        "并行补齐 `ingest/parsers/csv_parser.py`、`ingest/parsers/jsonl_parser.py`、"
        "`ingest/parsers/kv_parser.py` 三个独立格式解析模块，并完成 `ingest/registry.py` 的"
        "统一入口与目录批处理去重逻辑。各模块按照函数文档字符串和现有契约 `ingest/models.py` 实现，"
        "运行单元测试与集成测试确保 `tests/` 下全部用例通过。不要修改测试文件。"
    ),
    files=_C1_FILES,
    protected_paths=(
        "ingest/models.py",
        "tests/test_csv_parser.py",
        "tests/test_jsonl_parser.py",
        "tests/test_kv_parser.py",
        "tests/test_integration.py",
    ),
    public_test_targets=(
        "tests/test_csv_parser.py",
        "tests/test_jsonl_parser.py",
        "tests/test_kv_parser.py",
        "tests/test_integration.py",
    ),
    hidden_tests={
        "test_c1_hidden.py": r'''from ingest.parsers.csv_parser import parse_csv
from ingest.parsers.jsonl_parser import parse_jsonl
from ingest.parsers.kv_parser import parse_kv
from ingest.registry import ingest_directory


def test_csv_escaped_quotes_and_blank_lines():
    text = '\n# c\n\nid,bio\n\n1,"He said ""hi"", ok"\n'
    res = parse_csv(text)
    assert res.ok
    assert res.records[0].data["bio"] == 'He said "hi", ok'


def test_kv_escaped_quote_in_quoted_value():
    res = parse_kv('msg = "say \\"hello\\""\n')
    assert res.ok
    assert res.records[0].data["msg"] == 'say "hello"'


def test_jsonl_empty_input():
    res = parse_jsonl("\n// only comment\n")
    assert res.ok
    assert res.records == []


def test_dedupe_ignores_none_keys(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"name":"x"}\n{"name":"y"}\n', encoding="utf-8")
    out = ingest_directory(tmp_path, dedupe_key="id")
    assert len(out["records"]) == 2
    assert out["duplicates_skipped"] == 0
'''
    },
    expected_initial_f2p=(
        "tests/test_csv_parser.py::test_csv_basic_and_coercion",
        "tests/test_jsonl_parser.py::test_jsonl_comments_and_valid_objects",
        "tests/test_kv_parser.py::test_kv_blocks_quotes_and_coercion",
        "tests/test_integration.py::test_ingest_directory_combines_and_dedupes",
    ),
    expected_initial_p2p=(),
    expected_routes=("collaboration", "team"),
    max_turns=28,
    timeout_sec=300.0,
)


# ---------------------------------------------------------------------------
# C2：4 个独立通知通道适配器 + 统一派发器（高并行度功能实现）
# ---------------------------------------------------------------------------

_C2_FILES: dict[str, str] = {
    "notify/__init__.py": "",
    "notify/events.py": '''"""Shared alert event and delivery receipt contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class AlertEvent:
    event_id: str
    service: str
    severity: Severity
    title: str
    body: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliveryReceipt:
    channel: str
    event_id: str
    ok: bool
    payload: str
    headers: dict[str, str] = field(default_factory=dict)
    error: str = ""
''',
    "notify/channels/__init__.py": "",
    "notify/channels/webhook.py": '''"""Signed JSON webhook channel (to be implemented)."""

from __future__ import annotations

from notify.events import AlertEvent, DeliveryReceipt


def build_webhook_delivery(event: AlertEvent, *, secret: str) -> DeliveryReceipt:
    """Serialize AlertEvent as canonical JSON (sort_keys=True, separators=(',', ':'))
    and compute HMAC-SHA256 hex signature over UTF-8 payload using secret.

    Requirements:
    - Reject empty secret with ValueError.
    - Payload JSON fields: event_id, service, severity, title, body, timestamp, metadata.
    - Headers must include:
        "Content-Type": "application/json"
        "X-Alert-Id": event.event_id
        "X-Signature-256": f"sha256={hex_digest}"
    - Return DeliveryReceipt(channel="webhook", event_id=event.event_id, ok=True, payload=..., headers=...).
    """
    raise NotImplementedError("implement build_webhook_delivery")
''',
    "notify/channels/email_mime.py": '''"""RFC-2822 email formatter channel (to be implemented)."""

from __future__ import annotations

from notify.events import AlertEvent, DeliveryReceipt


def build_email_delivery(
    event: AlertEvent,
    *,
    sender: str,
    recipients: list[str],
) -> DeliveryReceipt:
    """Render an email payload and headers for an AlertEvent.

    Requirements:
    - Reject empty recipients list or blank sender with ValueError.
    - Map severity to X-Priority header: critical -> "1", warning -> "3", info -> "5".
    - Subject format: f"[{event.severity.upper()}][{event.service}] {event.title}"
      (strip CR/LF from subject to prevent header injection).
    - Headers dict includes: From, To (comma-space joined), Subject, X-Priority.
    - Payload includes header lines ("Key: Value") sorted or ordered, blank line,
      body text, and sorted metadata lines ("key=value") when metadata is non-empty.
    - Return DeliveryReceipt(channel="email", event_id=event.event_id, ok=True, ...).
    """
    raise NotImplementedError("implement build_email_delivery")
''',
    "notify/channels/markdown_digest.py": '''"""Grouped Markdown digest renderer (to be implemented)."""

from __future__ import annotations

from notify.events import AlertEvent, DeliveryReceipt


def build_markdown_digest(events: list[AlertEvent], *, max_body_chars: int = 80) -> DeliveryReceipt:
    """Render a multi-event Markdown summary digest grouped by severity (critical -> warning -> info).

    Requirements:
    - Reject max_body_chars < 10 with ValueError.
    - Title line: "# Alert Digest (N events)"
    - Only include severity sections ("## CRITICAL", "## WARNING", "## INFO") that have >= 1 event.
    - Within each section, sort events by (timestamp, event_id).
    - Each item line: "- **{service}** `{event_id}`: {title} — {truncated_body}"
      where body newlines are collapsed to single spaces and truncated to max_body_chars
      (ending with "..." if truncated, total truncated body length <= max_body_chars).
    - Return DeliveryReceipt(channel="markdown", event_id="digest", ok=True, payload=...).
    """
    raise NotImplementedError("implement build_markdown_digest")
''',
    "notify/channels/jsonl_audit.py": '''"""Redacting JSONL audit channel (to be implemented)."""

from __future__ import annotations

from notify.events import AlertEvent, DeliveryReceipt

SENSITIVE_KEYS = ("password", "secret", "token", "api_key", "authorization")


def build_audit_line(event: AlertEvent) -> DeliveryReceipt:
    """Render a single compact JSONL line with recursive secret redaction in metadata.

    Requirements:
    - Recursively walk event.metadata (dicts and lists); any dict key containing any
      word in SENSITIVE_KEYS (case-insensitive) must have its value replaced by "[REDACTED]".
    - Do NOT mutate event.metadata in place.
    - Output payload must be single-line compact JSON (no embedded unescaped newlines) with keys:
      event_id, service, severity, title, timestamp, metadata.
    - Return DeliveryReceipt(channel="audit", event_id=event.event_id, ok=True, payload=line).
    """
    raise NotImplementedError("implement build_audit_line")
''',
    "notify/dispatcher.py": '''"""Multi-channel notification dispatcher (to be implemented)."""

from __future__ import annotations

from notify.events import AlertEvent, DeliveryReceipt


class NotificationDispatcher:
    """Route AlertEvents to configured channels with deduplication by event_id.

    Routing rules:
    - "info": ["audit"]
    - "warning": ["audit", "webhook"]
    - "critical": ["audit", "webhook", "email"]
    - Duplicate event_id (already dispatched successfully once) must be skipped, returning [].
    - Also provide flush_digest() -> DeliveryReceipt returning markdown digest of all
      unique dispatched events (or empty digest with 0 events).
    """

    def __init__(self, *, webhook_secret: str, sender: str, recipients: list[str]) -> None:
        self.webhook_secret = webhook_secret
        self.sender = sender
        self.recipients = list(recipients)
        self._seen_ids: set[str] = set()
        self._dispatched_events: list[AlertEvent] = []

    def dispatch(self, event: AlertEvent) -> list[DeliveryReceipt]:
        raise NotImplementedError("implement dispatch")

    def flush_digest(self) -> DeliveryReceipt:
        raise NotImplementedError("implement flush_digest")
''',
    "tests/test_channels_and_dispatcher.py": '''import hashlib
import hmac
import json

import pytest

from notify.channels.email_mime import build_email_delivery
from notify.channels.jsonl_audit import build_audit_line
from notify.channels.markdown_digest import build_markdown_digest
from notify.channels.webhook import build_webhook_delivery
from notify.dispatcher import NotificationDispatcher
from notify.events import AlertEvent


def _evt(eid="e1", severity="critical", title="DB Down", body="Primary timeout", meta=None):
    return AlertEvent(
        event_id=eid,
        service="payments",
        severity=severity,
        title=title,
        body=body,
        timestamp="2026-11-20T10:00:00Z",
        metadata=meta or {},
    )


def test_webhook_hmac_signature():
    e = _evt(meta={"region": "us-east"})
    r = build_webhook_delivery(e, secret="topsecret")
    assert r.ok and r.channel == "webhook"
    expected_sig = hmac.new(b"topsecret", r.payload.encode("utf-8"), hashlib.sha256).hexdigest()
    assert r.headers["X-Signature-256"] == f"sha256={expected_sig}"
    assert r.headers["X-Alert-Id"] == "e1"
    parsed = json.loads(r.payload)
    assert parsed["service"] == "payments" and parsed["metadata"] == {"region": "us-east"}


def test_email_headers_and_injection_guard():
    e = _evt(severity="critical", title="Bad\\r\\nBcc: evil@x.com", meta={"env": "prod"})
    r = build_email_delivery(e, sender="alerts@acme.io", recipients=["a@acme.io", "b@acme.io"])
    assert r.ok and r.channel == "email"
    assert "\\r" not in r.headers["Subject"] and "\\n" not in r.headers["Subject"]
    assert r.headers["X-Priority"] == "1"
    assert r.headers["To"] == "a@acme.io, b@acme.io"
    assert "env=prod" in r.payload


def test_markdown_digest_order_and_truncation():
    e1 = _evt("e1", "info", "Sync", "ok")
    e2 = _evt("e2", "critical", "Outage", "line1\\n" + "x" * 100)
    r = build_markdown_digest([e1, e2], max_body_chars=25)
    assert r.ok and r.channel == "markdown"
    assert r.payload.startswith("# Alert Digest (2 events)")
    assert r.payload.index("## CRITICAL") < r.payload.index("## INFO")
    assert "## WARNING" not in r.payload
    assert "..." in r.payload


def test_audit_line_redacts_nested_secrets_without_mutating():
    meta = {"user": "bob", "nested": {"api_key": "k123", "items": [{"AuthToken": "t999"}]}}
    e = _evt(meta=meta)
    r = build_audit_line(e)
    assert "\\n" not in r.payload
    obj = json.loads(r.payload)
    assert obj["metadata"]["nested"]["api_key"] == "[REDACTED]"
    assert obj["metadata"]["nested"]["items"][0]["AuthToken"] == "[REDACTED]"
    assert obj["metadata"]["user"] == "bob"
    # Original metadata untouched
    assert meta["nested"]["api_key"] == "k123"


def test_dispatcher_routing_dedup_and_digest():
    disp = NotificationDispatcher(
        webhook_secret="sec", sender="bot@acme.io", recipients=["ops@acme.io"]
    )
    r_info = disp.dispatch(_evt("i1", "info", "Ping", "ok"))
    assert [r.channel for r in r_info] == ["audit"]

    r_warn = disp.dispatch(_evt("w1", "warning", "Latency", "high"))
    assert [r.channel for r in r_warn] == ["audit", "webhook"]

    r_crit = disp.dispatch(_evt("c1", "critical", "Crash", "oom"))
    assert [r.channel for r in r_crit] == ["audit", "webhook", "email"]

    # Duplicate skipped
    assert disp.dispatch(_evt("c1", "critical", "Crash", "oom")) == []

    digest = disp.flush_digest()
    assert "(3 events)" in digest.payload


def test_channel_validation_errors():
    with pytest.raises(ValueError):
        build_webhook_delivery(_evt(), secret="")
    with pytest.raises(ValueError):
        build_email_delivery(_evt(), sender="a@b.io", recipients=[])
    with pytest.raises(ValueError):
        build_markdown_digest([], max_body_chars=5)
''',
}

C2_NOTIFIERS = EvalTask(
    id="c2_notification_channels",
    suite="C",
    title="并行多模块实现：4 个独立通知通道适配器 + 路由派发器",
    description="4 个完全解耦通道文件（webhook/email/markdown/audit）+ 1 个集成 dispatcher。",
    prompt=(
        "并行实现 `notify/channels/webhook.py`、`notify/channels/email_mime.py`、"
        "`notify/channels/markdown_digest.py`、`notify/channels/jsonl_audit.py` 四个独立通道模块，"
        "并完成 `notify/dispatcher.py` 的分级路由、按 `event_id` 去重和汇总摘要输出。"
        "严格遵循各模块 docstring 和 `notify/events.py` 契约，确保 `tests/test_channels_and_dispatcher.py` 全部通过。"
        "不要修改测试文件。"
    ),
    files=_C2_FILES,
    protected_paths=("notify/events.py", "tests/test_channels_and_dispatcher.py"),
    public_test_targets=("tests/test_channels_and_dispatcher.py",),
    hidden_tests={
        "test_c2_hidden.py": '''from notify.channels.email_mime import build_email_delivery
from notify.channels.markdown_digest import build_markdown_digest
from notify.dispatcher import NotificationDispatcher
from notify.events import AlertEvent


def test_email_warning_and_info_priorities():
    w = AlertEvent("w", "svc", "warning", "Warn", "b", "2026-01-01T00:00:00Z")
    i = AlertEvent("i", "svc", "info", "Info", "b", "2026-01-01T00:00:00Z")
    assert build_email_delivery(w, sender="s@x.io", recipients=["r@x.io"]).headers["X-Priority"] == "3"
    assert build_email_delivery(i, sender="s@x.io", recipients=["r@x.io"]).headers["X-Priority"] == "5"


def test_markdown_empty_digest():
    r = build_markdown_digest([])
    assert r.payload.startswith("# Alert Digest (0 events)")


def test_dispatcher_empty_flush():
    disp = NotificationDispatcher(webhook_secret="s", sender="a@b.io", recipients=["c@d.io"])
    assert "(0 events)" in disp.flush_digest().payload
'''
    },
    expected_initial_f2p=(
        "tests/test_channels_and_dispatcher.py::test_webhook_hmac_signature",
        "tests/test_channels_and_dispatcher.py::test_email_headers_and_injection_guard",
        "tests/test_channels_and_dispatcher.py::test_markdown_digest_order_and_truncation",
        "tests/test_channels_and_dispatcher.py::test_audit_line_redacts_nested_secrets_without_mutating",
        "tests/test_channels_and_dispatcher.py::test_dispatcher_routing_dedup_and_digest",
    ),
    expected_initial_p2p=(),
    expected_routes=("collaboration", "team"),
    max_turns=30,
    timeout_sec=300.0,
)


# ---------------------------------------------------------------------------
# C3：跨 4 个独立校验模块的批量契约迁移（保持旧版 P2P 兼容）
# ---------------------------------------------------------------------------

_C3_FILES: dict[str, str] = {
    "validators/__init__.py": "",
    "validators/types.py": '''"""Structured validation outcome types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Outcome:
    ok: bool
    code: str
    severity: Severity
    field: str
    message: str
''',
    "validators/schema_rules.py": '''"""Legacy schema validator returning (bool, str). Needs structured migrate."""

from __future__ import annotations

from validators.types import Outcome


def validate_schema(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        return False, "name is required"
    if not isinstance(payload.get("version"), int) or payload["version"] < 1:
        return False, "version must be a positive integer"
    return True, "ok"


def check_schema(payload: dict) -> list[Outcome]:
    """Return structured Outcome list for all schema violations (empty list if valid)."""
    raise NotImplementedError("implement check_schema")
''',
    "validators/auth_rules.py": '''"""Legacy auth validator returning (bool, str). Needs structured migrate."""

from __future__ import annotations

from validators.types import Outcome


def validate_auth(payload: dict) -> tuple[bool, str]:
    role = payload.get("role")
    if role not in {"admin", "editor", "viewer"}:
        return False, "unsupported role"
    if role == "admin" and not payload.get("mfa_verified"):
        return False, "admin requires mfa"
    return True, "ok"


def check_auth(payload: dict) -> list[Outcome]:
    """Return structured Outcome list for auth violations (empty list if valid)."""
    raise NotImplementedError("implement check_auth")
''',
    "validators/quota_rules.py": '''"""Legacy quota validator returning (bool, str). Needs structured migrate."""

from __future__ import annotations

from validators.types import Outcome


def validate_quota(payload: dict) -> tuple[bool, str]:
    used = payload.get("used_mb", 0)
    limit = payload.get("limit_mb", 100)
    if not isinstance(used, (int, float)) or used < 0:
        return False, "used_mb must be non-negative"
    if used > limit:
        return False, "quota exceeded"
    return True, "ok"


def check_quota(payload: dict) -> list[Outcome]:
    """Return structured Outcome list for quota errors and near-limit warnings."""
    raise NotImplementedError("implement check_quota")
''',
    "validators/billing_rules.py": '''"""Legacy billing validator returning (bool, str). Needs structured migrate."""

from __future__ import annotations

from validators.types import Outcome


def validate_billing(payload: dict) -> tuple[bool, str]:
    plan = payload.get("plan", "free")
    if plan not in {"free", "pro", "enterprise"}:
        return False, "unknown billing plan"
    if plan != "free" and not payload.get("payment_method"):
        return False, "paid plan requires payment_method"
    return True, "ok"


def check_billing(payload: dict) -> list[Outcome]:
    """Return structured Outcome list for billing violations (empty list if valid)."""
    raise NotImplementedError("implement check_billing")
''',
    "validators/engine.py": '''"""Rule engine aggregating all 4 validator modules."""

from __future__ import annotations

from validators.auth_rules import check_auth, validate_auth
from validators.billing_rules import check_billing, validate_billing
from validators.quota_rules import check_quota, validate_quota
from validators.schema_rules import check_schema, validate_schema


def run_all_legacy(payload: dict) -> list[tuple[bool, str]]:
    return [
        validate_schema(payload),
        validate_auth(payload),
        validate_quota(payload),
        validate_billing(payload),
    ]


def evaluate_payload(payload: dict) -> dict:
    """Aggregate structured outcomes from all 4 modules into ok/errors/warnings/outcomes."""
    raise NotImplementedError("implement evaluate_payload")
''',
    "tests/test_legacy_compat.py": '''from validators.engine import run_all_legacy


def test_legacy_all_pass():
    payload = {
        "name": "app",
        "version": 2,
        "role": "admin",
        "mfa_verified": True,
        "used_mb": 40,
        "limit_mb": 100,
        "plan": "pro",
        "payment_method": "card_1",
    }
    results = run_all_legacy(payload)
    assert all(ok for ok, _ in results)


def test_legacy_failures_preserved():
    payload = {"name": "", "version": 0, "role": "guest", "used_mb": 200, "limit_mb": 100, "plan": "pro"}
    results = run_all_legacy(payload)
    assert [ok for ok, _ in results] == [False, False, False, False]
''',
    "tests/test_structured_outcomes.py": '''from validators.auth_rules import check_auth
from validators.billing_rules import check_billing
from validators.engine import evaluate_payload
from validators.quota_rules import check_quota
from validators.schema_rules import check_schema


def test_schema_collects_all_outcomes():
    outcomes = check_schema({"name": "", "version": 0})
    assert [(o.code, o.field, o.ok) for o in outcomes] == [
        ("SCHEMA_NAME_REQUIRED", "name", False),
        ("SCHEMA_VERSION_INVALID", "version", False),
    ]


def test_auth_structured_codes():
    assert check_auth({"role": "unknown"})[0].code == "AUTH_ROLE_INVALID"
    out = check_auth({"role": "admin", "mfa_verified": False})
    assert out[0].code == "AUTH_MFA_REQUIRED" and out[0].field == "mfa_verified"


def test_quota_warning_and_error():
    # Warning when used_mb >= 80% of limit_mb (and <= limit_mb)
    warn = check_quota({"used_mb": 85, "limit_mb": 100})
    assert len(warn) == 1
    assert warn[0].ok is True and warn[0].severity == "warning" and warn[0].code == "QUOTA_NEAR_LIMIT"

    err = check_quota({"used_mb": 120, "limit_mb": 100})
    assert err[0].ok is False and err[0].code == "QUOTA_EXCEEDED"


def test_billing_structured_codes():
    assert check_billing({"plan": "gold"})[0].code == "BILLING_PLAN_UNKNOWN"
    out = check_billing({"plan": "enterprise"})
    assert out[0].code == "BILLING_PAYMENT_REQUIRED" and out[0].field == "payment_method"


def test_engine_evaluate_payload_aggregates_errors_and_warnings():
    payload = {
        "name": "svc",
        "version": 1,
        "role": "admin",
        "mfa_verified": False,
        "used_mb": 90,
        "limit_mb": 100,
        "plan": "pro",
    }
    report = evaluate_payload(payload)
    assert report["ok"] is False
    assert [o.code for o in report["errors"]] == ["AUTH_MFA_REQUIRED", "BILLING_PAYMENT_REQUIRED"]
    assert [o.code for o in report["warnings"]] == ["QUOTA_NEAR_LIMIT"]
''',
}

C3_MIGRATION = EvalTask(
    id="c3_validator_migration",
    suite="C",
    title="并行跨模块迁移：4 个校验器升级结构化 Outcome 并保留旧版兼容",
    description="4 个独立校验模块（schema/auth/quota/billing）并行升级结构化错误码与引擎聚合，同时保持旧版 P2P 兼容。",
    prompt=(
        "并行升级 `validators/schema_rules.py`、`validators/auth_rules.py`、`validators/quota_rules.py`、"
        "`validators/billing_rules.py` 四个校验模块，分别新增返回 `list[Outcome]`（使用 `validators/types.py`）的 "
        "`check_schema`、`check_auth`、`check_quota`、`check_billing` 函数，并在 `validators/engine.py` 中新增 "
        "`evaluate_payload(payload) -> dict`（返回 `ok`、`errors`、`warnings`、`outcomes`），"
        "同时保持原有 `validate_*` 和 `run_all_legacy` 完全向后兼容。"
        "确保 `tests/test_legacy_compat.py` 和 `tests/test_structured_outcomes.py` 全部通过。不要修改测试文件。"
    ),
    files=_C3_FILES,
    protected_paths=(
        "validators/types.py",
        "tests/test_legacy_compat.py",
        "tests/test_structured_outcomes.py",
    ),
    public_test_targets=(
        "tests/test_legacy_compat.py",
        "tests/test_structured_outcomes.py",
    ),
    hidden_tests={
        "test_c3_hidden.py": '''from validators.engine import evaluate_payload
from validators.quota_rules import check_quota


def test_engine_clean_payload_passes():
    payload = {
        "name": "ok-svc",
        "version": 3,
        "role": "editor",
        "used_mb": 20,
        "limit_mb": 100,
        "plan": "free",
    }
    report = evaluate_payload(payload)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["warnings"] == []


def test_quota_negative_used_mb():
    out = check_quota({"used_mb": -5, "limit_mb": 100})
    assert out and out[0].ok is False and out[0].field == "used_mb"
'''
    },
    expected_initial_f2p=(
        "tests/test_structured_outcomes.py::test_schema_collects_all_outcomes",
        "tests/test_structured_outcomes.py::test_auth_structured_codes",
        "tests/test_structured_outcomes.py::test_quota_warning_and_error",
        "tests/test_structured_outcomes.py::test_billing_structured_codes",
        "tests/test_structured_outcomes.py::test_engine_evaluate_payload_aggregates_errors_and_warnings",
    ),
    expected_initial_p2p=(
        "tests/test_legacy_compat.py::test_legacy_all_pass",
        "tests/test_legacy_compat.py::test_legacy_failures_preserved",
    ),
    expected_routes=("collaboration", "team"),
    max_turns=28,
    timeout_sec=300.0,
)


# ---------------------------------------------------------------------------
# C4：4 个独立子系统并行只读缺陷审计（验证并行只读调研与零写入约束）
# ---------------------------------------------------------------------------

def _make_subsystem_distractors(sub: str) -> dict[str, str]:
    return {
        f"subsystems/{sub}/__init__.py": "",
        f"subsystems/{sub}/models.py": (
            f'"""Data models for {sub} subsystem."""\n\n'
            f"from dataclasses import dataclass\n\n"
            f"@dataclass\nclass {sub.title()}Record:\n"
            f"    id: str\n    status: str = 'active'\n"
        ),
        f"subsystems/{sub}/metrics.py": (
            f'"""Metrics helpers for {sub}."""\n\n'
            f"def emit_counter(name: str, value: int = 1) -> dict:\n"
            f'    return {{"subsystem": "{sub}", "metric": name, "value": value}}\n'
        ),
        f"subsystems/{sub}/config.py": (
            f'"""Static defaults for {sub}."""\n\n'
            f"DEFAULT_TIMEOUT_MS = 5000\nMAX_RETRIES = 3\n"
        ),
    }


_C4_FILES: dict[str, str] = {
    "subsystems/__init__.py": "",
    **_make_subsystem_distractors("auth"),
    **_make_subsystem_distractors("cache"),
    **_make_subsystem_distractors("queue"),
    **_make_subsystem_distractors("billing"),
    "subsystems/auth/tokens.py": '''"""JWT-like signed token verifier."""

from __future__ import annotations

import time


def verify_session_token(token_record: dict, supplied_sig: str, *, now: float | None = None) -> bool:
    """Verify token signature and expiration timestamp."""
    current = now if now is not None else time.time()
    expected = token_record.get("signature", "")
    # DEFECT: uses non-constant-time == AND compares expires_at backwards (allows expired tokens!)
    if supplied_sig != expected:
        return False
    expires_at = float(token_record.get("expires_at", 0))
    if current < expires_at:
        return False
    return True
''',
    "subsystems/cache/lru.py": '''"""Bounded LRU cache with TTL support."""

from __future__ import annotations

from collections import OrderedDict


class TtlLruCache:
    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._items: OrderedDict[str, tuple[object, float]] = OrderedDict()

    def put(self, key: str, value: object, *, expires_at: float) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = (value, expires_at)
        # DEFECT: evicts the MOST recently used item (last=True) instead of least recently used (last=False)
        if len(self._items) > self.max_size:
            self._items.popitem(last=True)

    def get(self, key: str, *, now: float):
        entry = self._items.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if now >= expires_at:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value
''',
    "subsystems/queue/worker_pool.py": '''"""In-memory job queue with visibility lease."""

from __future__ import annotations


class LeaseQueue:
    def __init__(self) -> None:
        self.ready: list[dict] = []
        self.in_flight: dict[str, dict] = {}

    def enqueue(self, job_id: str, payload: dict) -> None:
        self.ready.append({"id": job_id, "payload": payload, "attempts": 0})

    def lease(self) -> dict | None:
        if not self.ready:
            return None
        job = self.ready.pop(0)
        job["attempts"] += 1
        self.in_flight[job["id"]] = job
        return job

    def nack(self, job_id: str, *, max_attempts: int = 3) -> bool:
        """Requeue failed job if under max_attempts."""
        job = self.in_flight.get(job_id)
        if job is None:
            return False
        # DEFECT: never removes job_id from self.in_flight (leaks in_flight state)
        # AND uses > instead of < so it requeues ONLY AFTER exceeding max_attempts!
        if job["attempts"] > max_attempts:
            self.ready.append(job)
            return True
        return False
''',
    "subsystems/billing/ledger.py": '''"""Idempotent invoice charge processor."""

from __future__ import annotations


class ChargeLedger:
    def __init__(self) -> None:
        self.balances: dict[str, int] = {}
        self.charged_keys: set[str] = set()

    def credit(self, account: str, cents: int) -> None:
        self.balances[account] = self.balances.get(account, 0) + cents

    def charge(self, account: str, cents: int, *, idempotency_key: str) -> int:
        """Deduct cents once per idempotency_key."""
        if cents <= 0:
            raise ValueError("cents must be positive")
        if idempotency_key in self.charged_keys:
            return self.balances.get(account, 0)
        current = self.balances.get(account, 0)
        if current < cents:
            raise ValueError("insufficient funds")
        # DEFECT: adds cents instead of subtracting AND never records idempotency_key in self.charged_keys!
        self.balances[account] = current + cents
        return self.balances[account]
''',
}


def _grade_c4_audit(response_text: str, _files: dict[str, str]) -> tuple[bool, str, dict[str, Any]]:
    """确定性校验只读审计报告是否准确找出 4 个子系统的真实文件、符号与缺陷。"""
    lower = response_text.lower()
    checks = {
        "auth": (
            "tokens.py" in lower
            and "verify_session_token" in lower
            and bool(re.search(r"(expir|过期|反向|current\s*<\s*expires_at|hmac\.compare_digest|constant)", lower))
        ),
        "cache": (
            "lru.py" in lower
            and ("ttllrucache" in lower or "popitem" in lower)
            and bool(re.search(r"(last\s*=\s*true|last\s*=\s*false|最近使用|most recent|淘汰)", lower))
        ),
        "queue": (
            "worker_pool.py" in lower
            and "nack" in lower
            and bool(re.search(r"(in_flight|pop|泄漏|leak|max_attempts|>\s*max_attempts|<\s*max_attempts)", lower))
        ),
        "billing": (
            "ledger.py" in lower
            and "charge" in lower
            and bool(re.search(r"(idempotency_key|charged_keys|current\s*\+\s*cents|加|扣款|重复扣)", lower))
        ),
    }
    found = [name for name, ok in checks.items() if ok]
    missing = [name for name, ok in checks.items() if not ok]
    passed = len(found) == 4
    summary = f"Found {len(found)}/4 subsystem defects; missing={missing}"
    return passed, summary, {"subsystems": checks, "found_count": len(found)}


C4_PARALLEL_AUDIT = EvalTask(
    id="c4_parallel_subsystem_audit",
    suite="C",
    title="并行只读缺陷审计：定位 4 个独立子系统中的生产级逻辑漏洞",
    description="20 文件 4 子系统，要求只读调查并精确输出 4 处缺陷的文件路径、函数符号、行号与根因。",
    prompt=(
        "请**严格只读审计** `subsystems/` 下的 4 个独立子系统（`auth`、`cache`、`queue`、`billing`），"
        "找出每个子系统中隐藏的生产逻辑缺陷。不要修改任何文件。\n"
        "最终报告按子系统逐一列出：文件路径、函数/类名、具体缺陷原因与修复建议。"
    ),
    files=_C4_FILES,
    read_only=True,
    custom_grader=_grade_c4_audit,
    expected_routes=("explore", "collaboration", "team", "react"),
    forbidden_routes=("debug",),
    max_turns=18,
    timeout_sec=180.0,
)

SUITE_C_TASKS: tuple[EvalTask, ...] = (
    C1_PARSERS,
    C2_NOTIFIERS,
    C3_MIGRATION,
    C4_PARALLEL_AUDIT,
)
