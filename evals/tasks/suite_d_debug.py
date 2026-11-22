"""Suite D：强化 Debug 与 Code Harness 专项评测（回归陷阱、高风险多文件审核、Runbook 规范）。"""

from __future__ import annotations

from evals.types import EvalTask

# ---------------------------------------------------------------------------
# D1：回归陷阱（修 F2P 时极易破坏已有 P2P 测试）
# ---------------------------------------------------------------------------

D1_SCHEDULER_TRAP = EvalTask(
    id="d1_scheduler_regression_trap",
    suite="D",
    title="回归陷阱修复：指数退避 Jitter 上限钳位（强 F2P + P2P 保护）",
    description="点名 pytest 失败用例并要求保持全部 P2P 通过，验证 Debug 隔离验收与防回归能力。",
    prompt=(
        "请复现并修复 `tests/test_scheduler.py::test_jitter_respects_max_delay`，"
        "同时保持 `tests/test_scheduler.py` 中其他所有已通过用例继续通过。不要修改测试文件。"
    ),
    files={
        "scheduler/__init__.py": "",
        "scheduler/backoff.py": '''"""Exponential backoff calculator with deterministic jitter."""

from __future__ import annotations

import random


def compute_delay(
    attempt: int,
    *,
    base: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 30.0,
    jitter_ratio: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """Compute retry delay in seconds.

    - attempt must be >= 0 (raises ValueError otherwise).
    - base > 0, factor >= 1.0, max_delay >= base, 0.0 <= jitter_ratio <= 0.5.
    - Raw exponential delay = min(max_delay, base * (factor ** attempt)).
    - When jitter_ratio > 0, apply uniform jitter in [-jitter_ratio, +jitter_ratio]
      relative to raw delay, while ALWAYS clamping the final result into [0.0, max_delay].
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if base <= 0 or factor < 1.0 or max_delay < base:
        raise ValueError("invalid backoff bounds")
    if not (0.0 <= jitter_ratio <= 0.5):
        raise ValueError("jitter_ratio must be between 0.0 and 0.5")

    raw = min(max_delay, base * (factor**attempt))
    if jitter_ratio == 0.0:
        return raw
    generator = rng or random.Random(0)
    delta = raw * jitter_ratio * generator.uniform(-1.0, 1.0)
    # BUG: forgets to clamp raw + delta to [0.0, max_delay], so positive jitter exceeds max_delay!
    return raw + delta
''',
        "tests/test_scheduler.py": '''import random

import pytest

from scheduler.backoff import compute_delay


def test_exponential_growth_without_jitter():
    assert compute_delay(0, base=1.0, factor=2.0, max_delay=10.0) == 1.0
    assert compute_delay(1, base=1.0, factor=2.0, max_delay=10.0) == 2.0
    assert compute_delay(2, base=1.0, factor=2.0, max_delay=10.0) == 4.0
    assert compute_delay(5, base=1.0, factor=2.0, max_delay=10.0) == 10.0


def test_jitter_is_deterministic_for_same_seed():
    a = compute_delay(2, base=1.0, max_delay=30.0, jitter_ratio=0.25, rng=random.Random(42))
    b = compute_delay(2, base=1.0, max_delay=30.0, jitter_ratio=0.25, rng=random.Random(42))
    assert a == b
    assert a != 4.0


def test_jitter_produces_variation_below_cap():
    rng = random.Random(7)
    samples = [compute_delay(1, base=2.0, max_delay=30.0, jitter_ratio=0.2, rng=rng) for _ in range(20)]
    assert min(samples) < 4.0 < max(samples)


def test_jitter_respects_max_delay():
    rng = random.Random(99)
    samples = [compute_delay(6, base=1.0, max_delay=10.0, jitter_ratio=0.3, rng=rng) for _ in range(50)]
    assert all(0.0 <= s <= 10.0 for s in samples)
    assert any(s < 10.0 for s in samples)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        compute_delay(-1)
    with pytest.raises(ValueError):
        compute_delay(0, jitter_ratio=0.8)
    with pytest.raises(ValueError):
        compute_delay(0, base=10.0, max_delay=5.0)
''',
    },
    protected_paths=("tests/test_scheduler.py",),
    public_test_targets=("tests/test_scheduler.py",),
    hidden_tests={
        "test_d1_hidden.py": '''import random

from scheduler.backoff import compute_delay


def test_jitter_never_negative_and_bounded_across_attempts():
    rng = random.Random(1234)
    for attempt in range(8):
        for _ in range(30):
            val = compute_delay(attempt, base=0.5, factor=3.0, max_delay=5.0, jitter_ratio=0.5, rng=rng)
            assert 0.0 <= val <= 5.0
'''
    },
    expected_initial_f2p=("tests/test_scheduler.py::test_jitter_respects_max_delay",),
    expected_initial_p2p=(
        "tests/test_scheduler.py::test_exponential_growth_without_jitter",
        "tests/test_scheduler.py::test_jitter_is_deterministic_for_same_seed",
        "tests/test_scheduler.py::test_jitter_produces_variation_below_cap",
        "tests/test_scheduler.py::test_invalid_parameters_raise",
    ),
    expected_routes=("debug", "react"),
    max_turns=16,
    timeout_sec=180.0,
)


# ---------------------------------------------------------------------------
# D2：双文件联动高风险修复（触发 DebugCoordinator AST 风险信号 -> Auditor 独立审核）
# ---------------------------------------------------------------------------

D2_LEDGER_RISK = EvalTask(
    id="d2_ledger_multi_file_risk",
    suite="D",
    title="多文件事务原子性修复：转账超限回滚（触发高风险独立审核）",
    description="修复需同时修改 accounts.py 与 posting.py，验证多文件风险识别、独立审核与原子性。",
    prompt=(
        "请复现并修复 `tests/test_ledger.py::test_transfer_rollback_on_limit_exceeded`，"
        "确保目标失败用例转为通过，并保持 `tests/test_ledger.py` 中所有原有通过用例不变。"
        "注意保持原子性：当目标账户超过最大余额上限时，转出账户余额与流水记录都不能残留脏状态。"
        "不要修改测试文件。"
    ),
    files={
        "ledger/__init__.py": "",
        "ledger/accounts.py": '''"""Account store with optional maximum balance cap."""

from __future__ import annotations

from dataclasses import dataclass


class AccountError(ValueError):
    pass


class BalanceLimitError(AccountError):
    pass


@dataclass
class Account:
    account_id: str
    balance: int
    max_balance: int | None = None


class AccountBook:
    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}

    def open_account(self, account_id: str, balance: int = 0, *, max_balance: int | None = None) -> Account:
        if balance < 0:
            raise AccountError("initial balance cannot be negative")
        if max_balance is not None and balance > max_balance:
            raise BalanceLimitError("initial balance exceeds max_balance")
        acc = Account(account_id=account_id, balance=balance, max_balance=max_balance)
        self._accounts[account_id] = acc
        return acc

    def get(self, account_id: str) -> Account:
        if account_id not in self._accounts:
            raise AccountError(f"unknown account: {account_id}")
        return self._accounts[account_id]

    def debit(self, account_id: str, amount: int) -> int:
        if amount <= 0:
            raise AccountError("amount must be positive")
        acc = self.get(account_id)
        if acc.balance < amount:
            raise AccountError("insufficient funds")
        acc.balance -= amount
        return acc.balance

    def credit(self, account_id: str, amount: int) -> int:
        if amount <= 0:
            raise AccountError("amount must be positive")
        acc = self.get(account_id)
        # BUG 1: does not check acc.max_balance before crediting!
        acc.balance += amount
        return acc.balance
''',
        "ledger/posting.py": '''"""Double-entry transfer posting service."""

from __future__ import annotations

from dataclasses import dataclass

from ledger.accounts import AccountBook, AccountError


@dataclass(frozen=True)
class TransferRecord:
    source: str
    target: str
    amount: int


class PostingService:
    def __init__(self, book: AccountBook) -> None:
        self.book = book
        self.history: list[TransferRecord] = []

    def transfer(self, source: str, target: str, amount: int) -> TransferRecord:
        if source == target:
            raise AccountError("cannot transfer to same account")
        # BUG 2: debits source first and appends history without rolling back if credit fails!
        self.book.debit(source, amount)
        self.book.credit(target, amount)
        rec = TransferRecord(source=source, target=target, amount=amount)
        self.history.append(rec)
        return rec
''',
        "tests/test_ledger.py": '''import pytest

from ledger.accounts import AccountBook, AccountError, BalanceLimitError
from ledger.posting import PostingService


def test_normal_transfer_updates_both_accounts_and_history():
    book = AccountBook()
    book.open_account("alice", 500)
    book.open_account("bob", 100, max_balance=400)
    svc = PostingService(book)

    rec = svc.transfer("alice", "bob", 200)
    assert rec.amount == 200
    assert book.get("alice").balance == 300
    assert book.get("bob").balance == 300
    assert len(svc.history) == 1


def test_transfer_insufficient_funds_keeps_state_clean():
    book = AccountBook()
    book.open_account("alice", 50)
    book.open_account("bob", 100)
    svc = PostingService(book)

    with pytest.raises(AccountError):
        svc.transfer("alice", "bob", 100)
    assert book.get("alice").balance == 50
    assert book.get("bob").balance == 100
    assert svc.history == []


def test_transfer_rollback_on_limit_exceeded():
    book = AccountBook()
    book.open_account("alice", 500)
    book.open_account("bob", 350, max_balance=400)
    svc = PostingService(book)

    with pytest.raises(BalanceLimitError):
        svc.transfer("alice", "bob", 100)
    # Atomic rollback: alice must still have 500, bob 350, history empty
    assert book.get("alice").balance == 500
    assert book.get("bob").balance == 350
    assert svc.history == []
''',
    },
    protected_paths=("tests/test_ledger.py",),
    public_test_targets=("tests/test_ledger.py",),
    hidden_tests={
        "test_d2_hidden.py": '''import pytest

from ledger.accounts import AccountBook, BalanceLimitError
from ledger.posting import PostingService


def test_direct_credit_enforces_max_balance():
    book = AccountBook()
    book.open_account("vault", 90, max_balance=100)
    with pytest.raises(BalanceLimitError):
        book.credit("vault", 15)
    assert book.get("vault").balance == 90


def test_transfer_exact_max_balance_allowed():
    book = AccountBook()
    book.open_account("a", 100)
    book.open_account("b", 80, max_balance=100)
    svc = PostingService(book)
    svc.transfer("a", "b", 20)
    assert book.get("a").balance == 80
    assert book.get("b").balance == 100
'''
    },
    expected_initial_f2p=("tests/test_ledger.py::test_transfer_rollback_on_limit_exceeded",),
    expected_initial_p2p=(
        "tests/test_ledger.py::test_normal_transfer_updates_both_accounts_and_history",
        "tests/test_ledger.py::test_transfer_insufficient_funds_keeps_state_clean",
    ),
    expected_routes=("debug", "react"),
    max_turns=18,
    timeout_sec=200.0,
)


# ---------------------------------------------------------------------------
# D3：项目规范 Runbook 驱动修复（验证 RAG 资料检索 + Debug 回归验收）
# ---------------------------------------------------------------------------

D3_RUNBOOK_SPEC = EvalTask(
    id="d3_idempotency_runbook_spec",
    suite="D",
    title="规范手册驱动修复：幂等键重放、载荷冲突与过期语义",
    description="预期语义详见 docs/idempotency_runbook.md，验证资料检索与受控调试验收。",
    prompt=(
        "请复现并修复 `tests/test_processor.py::test_duplicate_request_conflict`，"
        "请结合项目文档 `docs/idempotency_runbook.md` 中的幂等处理规范完成修复，"
        "并保持 `tests/test_processor.py` 中所有其他用例通过。不要修改测试文件和文档。"
    ),
    files={
        "docs/idempotency_runbook.md": '''# Idempotency Handling Runbook

## Rules for Duplicate Request Keys

When `JobProcessor.submit(key, payload, now=...)` receives an already-seen `key`:

1. **Expired Entry (`now >= entry.expires_at`)**:
   - The previous record has expired and MUST be evicted.
   - Re-execute the handler with the new `payload`, store a fresh record with `expires_at = now + ttl_seconds`,
     and return `{"status": "executed", "result": result, "replayed": False}`.

2. **Active Entry with Identical Canonical Payload**:
   - Compare payloads via canonical JSON (`json.dumps(payload, sort_keys=True, separators=(",", ":"))`).
   - Do NOT re-run the handler.
   - Return `{"status": "replayed", "result": entry.result, "replayed": True}`.

3. **Active Entry with Different Payload**:
   - Raise `IdempotencyConflictError` without modifying the stored entry or calling the handler.
''',
        "jobs/__init__.py": "",
        "jobs/processor.py": '''"""Idempotent job submission processor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class IdempotencyConflictError(ValueError):
    pass


@dataclass
class _StoredEntry:
    payload: dict[str, Any]
    result: Any
    expires_at: float


class JobProcessor:
    def __init__(self, handler: Callable[[dict[str, Any]], Any], *, ttl_seconds: float = 60.0) -> None:
        self.handler = handler
        self.ttl_seconds = float(ttl_seconds)
        self._records: dict[str, _StoredEntry] = {}

    def submit(self, key: str, payload: dict[str, Any], *, now: float = 0.0) -> dict[str, Any]:
        if not key or not key.strip():
            raise ValueError("idempotency key must be non-empty")
        existing = self._records.get(key)
        if existing is not None:
            # BUG: ignores expiration AND never detects payload conflict!
            return {"status": "replayed", "result": existing.result, "replayed": True}
        result = self.handler(payload)
        self._records[key] = _StoredEntry(
            payload=dict(payload),
            result=result,
            expires_at=float(now) + self.ttl_seconds,
        )
        return {"status": "executed", "result": result, "replayed": False}
''',
        "tests/test_processor.py": '''import pytest

from jobs.processor import IdempotencyConflictError, JobProcessor


def test_first_submission_and_identical_replay():
    calls = []
    proc = JobProcessor(lambda p: (calls.append(p), p["x"] * 2)[1], ttl_seconds=30.0)

    r1 = proc.submit("k1", {"x": 5, "meta": {"a": 1, "b": 2}}, now=10.0)
    r2 = proc.submit("k1", {"meta": {"b": 2, "a": 1}, "x": 5}, now=20.0)

    assert r1 == {"status": "executed", "result": 10, "replayed": False}
    assert r2 == {"status": "replayed", "result": 10, "replayed": True}
    assert len(calls) == 1


def test_duplicate_request_conflict():
    calls = []
    proc = JobProcessor(lambda p: (calls.append(p), p["x"] + 1)[1], ttl_seconds=30.0)

    proc.submit("k2", {"x": 1}, now=0.0)
    with pytest.raises(IdempotencyConflictError):
        proc.submit("k2", {"x": 2}, now=10.0)
    assert len(calls) == 1


def test_expired_key_reexecutes():
    calls = []
    proc = JobProcessor(lambda p: (calls.append(p), p["x"] * 10)[1], ttl_seconds=30.0)

    r1 = proc.submit("k3", {"x": 2}, now=0.0)
    r2 = proc.submit("k3", {"x": 3}, now=30.0)

    assert r1 == {"status": "executed", "result": 20, "replayed": False}
    assert r2 == {"status": "executed", "result": 30, "replayed": False}
    assert len(calls) == 2
''',
    },
    protected_paths=("tests/test_processor.py", "docs/idempotency_runbook.md"),
    public_test_targets=("tests/test_processor.py",),
    hidden_tests={
        "test_d3_hidden.py": '''import pytest

from jobs.processor import IdempotencyConflictError, JobProcessor


def test_conflict_does_not_overwrite_original_entry():
    calls = []
    proc = JobProcessor(lambda p: (calls.append(p), p["v"])[1], ttl_seconds=50.0)
    proc.submit("k", {"v": "orig"}, now=0.0)
    with pytest.raises(IdempotencyConflictError):
        proc.submit("k", {"v": "tamper"}, now=10.0)
    replay = proc.submit("k", {"v": "orig"}, now=20.0)
    assert replay == {"status": "replayed", "result": "orig", "replayed": True}
    assert len(calls) == 1


def test_mutation_of_caller_payload_after_submit_does_not_break_replay():
    payload = {"items": [1, 2]}
    proc = JobProcessor(lambda p: len(p["items"]), ttl_seconds=50.0)
    proc.submit("k-mut", payload, now=0.0)
    payload["items"].append(3)
    # Original stored payload was {"items": [1, 2]}, so {"items": [1, 2, 3]} must conflict!
    with pytest.raises(IdempotencyConflictError):
        proc.submit("k-mut", {"items": [1, 2, 3]}, now=5.0)
'''
    },
    expected_initial_f2p=(
        "tests/test_processor.py::test_duplicate_request_conflict",
        "tests/test_processor.py::test_expired_key_reexecutes",
    ),
    expected_initial_p2p=("tests/test_processor.py::test_first_submission_and_identical_replay",),
    knowledge_paths=("docs",),
    expected_routes=("debug", "react"),
    max_turns=16,
    timeout_sec=180.0,
)

SUITE_D_TASKS: tuple[EvalTask, ...] = (D1_SCHEDULER_TRAP, D2_LEDGER_RISK, D3_RUNBOOK_SPEC)
