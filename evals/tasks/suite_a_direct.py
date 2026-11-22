"""Suite A：简单直达任务（验证轻任务无过度路由、低延迟、低额外开销）。"""

from __future__ import annotations

from evals.types import EvalTask

A1_SLUGIFY = EvalTask(
    id="a1_slugify_fix",
    suite="A",
    title="单文件工具修复：Unicode Slugify 边界与截断",
    description="单文件小修，检验主 Agent 能否直接处理而不误触发 Team/Debug。",
    prompt=(
        "修复 `textutil/slug.py` 中 `slugify` 的边界缺陷，使 `tests/test_slug.py` 全部通过。"
        "不要修改测试文件。"
    ),
    files={
        "textutil/__init__.py": "from textutil.slug import slugify\n\n__all__ = ['slugify']\n",
        "textutil/slug.py": '''"""URL-safe ASCII slug generator."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str, *, max_length: int | None = None, separator: str = "-") -> str:
    """Convert text into a lowercase ASCII slug."""
    if not separator:
        raise ValueError("separator must be non-empty")
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", separator, normalized.lower())
    # BUG 1: does not strip leading/trailing separators properly when max_length truncates
    # BUG 2: does not collapse repeated multi-character or regex-special separators cleanly
    if max_length is not None:
        cleaned = cleaned[:max_length]
    return cleaned.strip(separator)
''',
        "tests/test_slug.py": '''import pytest

from textutil.slug import slugify


def test_basic_ascii():
    assert slugify("Hello, World!") == "hello-world"


def test_unicode_accents():
    assert slugify("Café Crème Brûlée") == "cafe-creme-brulee"


def test_leading_trailing_noise_before_truncation():
    # Truncating BEFORE stripping leading noise wastes max_length budget
    assert slugify("   !!!Hello World!!!", max_length=11) == "hello-world"


def test_truncation_does_not_end_with_separator():
    assert slugify("alpha beta gamma", max_length=6) == "alpha"


def test_custom_separator_regex_special_char():
    assert slugify("a + b + c", separator="+") == "a+b+c"


def test_invalid_max_length():
    with pytest.raises(ValueError):
        slugify("hello", max_length=0)
''',
    },
    protected_paths=("tests/test_slug.py",),
    public_test_targets=("tests/test_slug.py",),
    hidden_tests={
        "test_slug_hidden.py": '''import pytest

from textutil.slug import slugify


def test_negative_max_length():
    with pytest.raises(ValueError):
        slugify("abc", max_length=-3)


def test_dot_separator_and_consecutive_spaces():
    assert slugify("  foo   bar.baz  ", separator=".") == "foo.bar.baz"


def test_all_non_ascii_returns_empty():
    assert slugify("中文测试", max_length=10) == ""
'''
    },
    expected_initial_f2p=(
        "tests/test_slug.py::test_leading_trailing_noise_before_truncation",
        "tests/test_slug.py::test_invalid_max_length",
    ),
    expected_initial_p2p=(
        "tests/test_slug.py::test_basic_ascii",
        "tests/test_slug.py::test_unicode_accents",
        "tests/test_slug.py::test_truncation_does_not_end_with_separator",
        "tests/test_slug.py::test_custom_separator_regex_special_char",
    ),
    expected_routes=("react",),
    forbidden_routes=("team", "collaboration", "debug"),
    max_turns=12,
    timeout_sec=120.0,
)

A2_RATE_BUCKET = EvalTask(
    id="a2_rate_bucket_feature",
    suite="A",
    title="单文件小特性：令牌桶等待时间估算与突发上限",
    description="单文件小特性增量实现，检验轻任务直接修改效率与无误路由。",
    prompt=(
        "完善 `ratelimit/bucket.py` 中的 `TokenBucket`：修复补充令牌时的突发容量钳位，"
        "并实现 `peek_wait_seconds(tokens: float, now: float) -> float`（不消耗令牌，"
        "返回满足请求所需等待秒数；若请求超过 capacity 抛出 ValueError）。"
        "确保 `tests/test_bucket.py` 全部通过，不要改动测试文件。"
    ),
    files={
        "ratelimit/__init__.py": "from ratelimit.bucket import TokenBucket\n\n__all__ = ['TokenBucket']\n",
        "ratelimit/bucket.py": '''"""Deterministic token bucket rate limiter."""

from __future__ import annotations


class TokenBucket:
    def __init__(self, rate: float, capacity: float, *, now: float = 0.0) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_refill = float(now)

    def _refill(self, now: float) -> None:
        if now < self.last_refill:
            return
        elapsed = now - self.last_refill
        # BUG: missing clamp to self.capacity
        self.tokens = self.tokens + elapsed * self.rate
        self.last_refill = float(now)

    def consume(self, tokens: float, *, now: float) -> bool:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.capacity:
            raise ValueError("tokens exceeds bucket capacity")
        self._refill(now)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
''',
        "tests/test_bucket.py": '''import pytest

from ratelimit.bucket import TokenBucket


def test_consume_initial_capacity():
    bucket = TokenBucket(rate=2.0, capacity=5.0, now=0.0)
    assert bucket.consume(5.0, now=0.0) is True
    assert bucket.consume(1.0, now=0.0) is False


def test_refill_caps_at_capacity():
    bucket = TokenBucket(rate=10.0, capacity=5.0, now=0.0)
    assert bucket.consume(3.0, now=0.0) is True
    # After 10 seconds, tokens should cap at 5.0, not 102.0
    assert bucket.consume(5.0, now=10.0) is True
    assert bucket.consume(1.0, now=10.0) is False


def test_peek_wait_seconds_does_not_mutate_tokens():
    bucket = TokenBucket(rate=2.0, capacity=4.0, now=0.0)
    assert bucket.consume(4.0, now=0.0) is True
    assert bucket.peek_wait_seconds(3.0, now=0.5) == pytest.approx(1.0)
    # Peeking should not consume tokens
    assert bucket.consume(1.0, now=0.5) is True


def test_peek_wait_zero_when_available():
    bucket = TokenBucket(rate=5.0, capacity=10.0, now=0.0)
    assert bucket.peek_wait_seconds(4.0, now=0.0) == 0.0


def test_peek_wait_invalid_arguments():
    bucket = TokenBucket(rate=1.0, capacity=5.0, now=0.0)
    with pytest.raises(ValueError):
        bucket.peek_wait_seconds(6.0, now=0.0)
    with pytest.raises(ValueError):
        bucket.peek_wait_seconds(0.0, now=0.0)
''',
    },
    protected_paths=("tests/test_bucket.py",),
    public_test_targets=("tests/test_bucket.py",),
    hidden_tests={
        "test_bucket_hidden.py": '''import pytest

from ratelimit.bucket import TokenBucket


def test_clock_regression_ignored_in_peek():
    bucket = TokenBucket(rate=2.0, capacity=4.0, now=10.0)
    assert bucket.consume(4.0, now=10.0) is True
    assert bucket.peek_wait_seconds(2.0, now=9.0) == pytest.approx(1.0)


def test_peek_then_consume_at_exact_wait_time():
    bucket = TokenBucket(rate=4.0, capacity=8.0, now=1.0)
    assert bucket.consume(7.0, now=1.0) is True
    wait = bucket.peek_wait_seconds(5.0, now=1.0)
    assert wait == pytest.approx(1.0)
    assert bucket.consume(5.0, now=1.0 + wait) is True
'''
    },
    expected_initial_f2p=(
        "tests/test_bucket.py::test_refill_caps_at_capacity",
        "tests/test_bucket.py::test_peek_wait_seconds_does_not_mutate_tokens",
        "tests/test_bucket.py::test_peek_wait_zero_when_available",
        "tests/test_bucket.py::test_peek_wait_invalid_arguments",
    ),
    expected_initial_p2p=("tests/test_bucket.py::test_consume_initial_capacity",),
    expected_routes=("react",),
    forbidden_routes=("team", "collaboration", "debug"),
    max_turns=12,
    timeout_sec=120.0,
)

SUITE_A_TASKS: tuple[EvalTask, ...] = (A1_SLUGIFY, A2_RATE_BUCKET)
