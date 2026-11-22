"""Suite B：广度调研与跨模块耦合重构（验证 code_map / 并行 explore 的影响召回与效率）。"""

from __future__ import annotations

from evals.types import EvalTask


def _distractor_helpers(prefix: str, count: int) -> dict[str, str]:
    """生成逼真的辅助模块，使仓库达到 20+ 文件规模，考验代码感知与调用方召回。"""
    files: dict[str, str] = {f"{prefix}/__init__.py": ""}
    for idx in range(1, count + 1):
        files[f"{prefix}/util_{idx:02d}.py"] = (
            f'"""Utility helper module {idx}."""\n\n'
            f"def format_tag_{idx:02d}(value: str) -> str:\n"
            f'    return f"tag-{idx:02d}:{{value.strip().lower()}}"\n\n'
            f"def clamp_int_{idx:02d}(value: int, low: int = 0, high: int = 100) -> int:\n"
            f"    return max(low, min(high, value))\n"
        )
    return files


# ---------------------------------------------------------------------------
# B1：跨 7+ 耦合模块的上下文契约传播（24 文件仓库）
# ---------------------------------------------------------------------------

_B1_FILES: dict[str, str] = {
    **_distractor_helpers("helpers", 10),
    "core/__init__.py": "",
    "core/context.py": '''"""Request context shared across all services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    actor: str
''',
    "core/audit.py": '''"""In-memory structured audit sink."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AuditEntry:
    tenant_id: str
    actor: str
    action: str
    resource: str


@dataclass
class AuditSink:
    entries: list[AuditEntry] = field(default_factory=list)

    def record(self, *, tenant_id: str, actor: str, action: str, resource: str) -> AuditEntry:
        entry = AuditEntry(
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource=resource,
        )
        self.entries.append(entry)
        return entry
''',
    "services/__init__.py": "",
    "services/orders.py": '''from __future__ import annotations

from core.audit import AuditSink
from core.context import RequestContext
from services.inventory import InventoryService


class OrderService:
    def __init__(self, audit: AuditSink, inventory: InventoryService) -> None:
        self.audit = audit
        self.inventory = inventory

    def create_order(self, ctx: RequestContext, sku: str, qty: int) -> dict:
        reservation = self.inventory.reserve(ctx, sku, qty)
        self.audit.record(
            tenant_id=ctx.tenant_id,
            actor=ctx.actor,
            action="order.create",
            resource=f"{sku}:{qty}",
        )
        return {"sku": sku, "qty": qty, "reserved": reservation["reserved"]}
''',
    "services/inventory.py": '''from __future__ import annotations

from core.audit import AuditSink
from core.context import RequestContext


class InventoryService:
    def __init__(self, audit: AuditSink) -> None:
        self.audit = audit
        self.stock: dict[str, int] = {"widget": 50, "gadget": 20}

    def reserve(self, ctx: RequestContext, sku: str, qty: int) -> dict:
        available = self.stock.get(sku, 0)
        if qty > available:
            raise ValueError(f"insufficient stock for {sku}")
        self.stock[sku] = available - qty
        self.audit.record(
            tenant_id=ctx.tenant_id,
            actor=ctx.actor,
            action="inventory.reserve",
            resource=f"{sku}:{qty}",
        )
        return {"sku": sku, "reserved": qty}
''',
    "services/reports.py": '''from __future__ import annotations

from core.audit import AuditSink
from core.context import RequestContext


class ReportService:
    def __init__(self, audit: AuditSink) -> None:
        self.audit = audit

    def export_summary(self, ctx: RequestContext, period: str) -> dict:
        self.audit.record(
            tenant_id=ctx.tenant_id,
            actor=ctx.actor,
            action="report.export",
            resource=period,
        )
        return {"tenant": ctx.tenant_id, "period": period, "status": "ready"}
''',
    "services/webhooks.py": '''from __future__ import annotations

from core.audit import AuditSink
from core.context import RequestContext


class WebhookDispatcher:
    def __init__(self, audit: AuditSink) -> None:
        self.audit = audit
        self.sent: list[dict] = []

    def emit(self, ctx: RequestContext, event_type: str, payload: dict) -> dict:
        envelope = {
            "tenant_id": ctx.tenant_id,
            "actor": ctx.actor,
            "event_type": event_type,
            "payload": payload,
        }
        self.sent.append(envelope)
        self.audit.record(
            tenant_id=ctx.tenant_id,
            actor=ctx.actor,
            action="webhook.emit",
            resource=event_type,
        )
        return envelope
''',
    "api/__init__.py": "",
    "api/middleware.py": '''from __future__ import annotations

from core.context import RequestContext


def build_context(headers: dict[str, str]) -> RequestContext:
    lowered = {k.lower(): v for k, v in headers.items()}
    return RequestContext(
        tenant_id=lowered["x-tenant-id"],
        actor=lowered.get("x-actor", "anonymous"),
    )
''',
    "api/routes.py": '''from __future__ import annotations

from api.middleware import build_context
from core.audit import AuditSink
from services.inventory import InventoryService
from services.orders import OrderService
from services.reports import ReportService
from services.webhooks import WebhookDispatcher


class AppRouter:
    def __init__(self) -> None:
        self.audit = AuditSink()
        self.inventory = InventoryService(self.audit)
        self.orders = OrderService(self.audit, self.inventory)
        self.reports = ReportService(self.audit)
        self.webhooks = WebhookDispatcher(self.audit)

    def handle_order(self, headers: dict[str, str], sku: str, qty: int) -> dict:
        ctx = build_context(headers)
        result = self.orders.create_order(ctx, sku, qty)
        self.webhooks.emit(ctx, "order.created", result)
        return result

    def handle_report(self, headers: dict[str, str], period: str) -> dict:
        ctx = build_context(headers)
        return self.reports.export_summary(ctx, period)
''',
    "tests/test_context_propagation.py": '''from api.routes import AppRouter
from core.audit import AuditSink
from core.context import RequestContext
from services.inventory import InventoryService


def test_legacy_context_without_correlation_id_still_works():
    ctx = RequestContext(tenant_id="t1", actor="alice")
    sink = AuditSink()
    inv = InventoryService(sink)
    inv.reserve(ctx, "widget", 2)
    assert len(sink.entries) == 1
    assert sink.entries[0].correlation_id is None


def test_correlation_id_propagates_across_order_inventory_and_webhook():
    router = AppRouter()
    headers = {
        "X-Tenant-Id": "acme",
        "X-Actor": "bob",
        "X-Correlation-Id": "corr-9981",
    }
    router.handle_order(headers, "widget", 3)
    actions = [e.action for e in router.audit.entries]
    assert actions == ["inventory.reserve", "order.create", "webhook.emit"]
    assert all(e.correlation_id == "corr-9981" for e in router.audit.entries)
    assert router.webhooks.sent[0]["correlation_id"] == "corr-9981"


def test_correlation_id_propagates_to_reports():
    router = AppRouter()
    headers = {
        "x-tenant-id": "acme",
        "x-actor": "carol",
        "x-correlation-id": "corr-rpt-42",
    }
    router.handle_report(headers, "2026-Q4")
    assert router.audit.entries[0].correlation_id == "corr-rpt-42"
''',
}

B1_TENANT_CONTEXT = EvalTask(
    id="b1_tenant_context_propagation",
    suite="B",
    title="24文件跨层耦合重构：透传 correlation_id 至所有服务与审计日志",
    description="改动核心契约 RequestContext/AuditSink 并同步全部下游服务和中间件。",
    prompt=(
        "在项目中为 `RequestContext` 和 `AuditEntry`/`AuditSink.record` 增加可选字段 "
        "`correlation_id: str | None = None`，并在 HTTP 中间件读取 `X-Correlation-Id`（大小写不敏感），"
        "确保所有下游服务（订单、库存、报表、Webhook 派发器及其发送信封）都完整透传 `correlation_id`，"
        "同时保持未传 `correlation_id` 时的向后兼容。不要修改测试文件。"
    ),
    files=_B1_FILES,
    protected_paths=("tests/test_context_propagation.py",),
    public_test_targets=("tests/test_context_propagation.py",),
    hidden_tests={
        "test_b1_hidden.py": '''from api.middleware import build_context
from api.routes import AppRouter
from core.audit import AuditSink


def test_missing_correlation_header_defaults_to_none():
    ctx = build_context({"X-Tenant-Id": "t2"})
    assert ctx.correlation_id is None


def test_audit_record_legacy_call_without_correlation():
    sink = AuditSink()
    entry = sink.record(tenant_id="t1", actor="sys", action="ping", resource="health")
    assert entry.correlation_id is None


def test_webhook_envelope_contains_none_when_absent():
    router = AppRouter()
    router.handle_order({"X-Tenant-Id": "acme"}, "gadget", 1)
    assert router.webhooks.sent[0]["correlation_id"] is None
    assert all(e.correlation_id is None for e in router.audit.entries)
'''
    },
    expected_initial_f2p=(
        "tests/test_context_propagation.py::test_legacy_context_without_correlation_id_still_works",
        "tests/test_context_propagation.py::test_correlation_id_propagates_across_order_inventory_and_webhook",
        "tests/test_context_propagation.py::test_correlation_id_propagates_to_reports",
    ),
    expected_initial_p2p=(),
    expected_routes=("react", "plan", "explore", "code_map", "collaboration"),
    max_turns=22,
    timeout_sec=180.0,
)


# ---------------------------------------------------------------------------
# B2：22文件电商价格体系从 float 迁移到 Decimal
# ---------------------------------------------------------------------------

_B2_FILES: dict[str, str] = {
    **_distractor_helpers("extras", 10),
    "commerce/__init__.py": "",
    "commerce/money.py": '''"""Monetary helpers (currently float-based, causes rounding drift)."""

from __future__ import annotations


def quantize_amount(value: float) -> float:
    return round(float(value), 2)


def format_currency(amount: float, currency: str = "USD") -> str:
    return f"{currency} {amount:.2f}"
''',
    "commerce/cart.py": '''from __future__ import annotations

from dataclasses import dataclass

from commerce.money import quantize_amount


@dataclass(frozen=True)
class LineItem:
    sku: str
    unit_price: str | float
    quantity: int


def line_total(item: LineItem) -> float:
    return quantize_amount(float(item.unit_price) * item.quantity)


def subtotal(items: list[LineItem]) -> float:
    total = 0.0
    for item in items:
        total += line_total(item)
    return quantize_amount(total)
''',
    "commerce/discounts.py": '''from __future__ import annotations

from commerce.money import quantize_amount


def apply_percent_discount(amount: float, percent: float) -> float:
    discount = quantize_amount(amount * (percent / 100.0))
    return quantize_amount(amount - discount)
''',
    "commerce/tax.py": '''from __future__ import annotations

from commerce.money import quantize_amount


def compute_tax(amount: float, rate_percent: float) -> float:
    return quantize_amount(amount * (rate_percent / 100.0))
''',
    "commerce/invoice.py": '''from __future__ import annotations

from commerce.cart import LineItem, subtotal
from commerce.discounts import apply_percent_discount
from commerce.money import format_currency, quantize_amount
from commerce.tax import compute_tax


def build_invoice(
    items: list[LineItem],
    *,
    discount_percent: float = 0.0,
    tax_percent: float = 0.0,
    currency: str = "USD",
) -> dict:
    sub = subtotal(items)
    after_discount = apply_percent_discount(sub, discount_percent) if discount_percent else sub
    tax = compute_tax(after_discount, tax_percent) if tax_percent else 0.0
    grand = quantize_amount(after_discount + tax)
    return {
        "subtotal": sub,
        "after_discount": after_discount,
        "tax": tax,
        "total": grand,
        "formatted_total": format_currency(grand, currency),
    }
''',
    "commerce/refunds.py": '''from __future__ import annotations

from commerce.money import quantize_amount


def prorate_refund(paid_total: float, refunded_items: int, total_items: int) -> float:
    if total_items <= 0 or refunded_items < 0 or refunded_items > total_items:
        raise ValueError("invalid refund item counts")
    return quantize_amount(paid_total * refunded_items / total_items)
''',
    "commerce/export_csv.py": '''from __future__ import annotations

from commerce.cart import LineItem, line_total
from commerce.money import format_currency


def export_lines_csv(items: list[LineItem], currency: str = "USD") -> str:
    rows = ["sku,qty,total"]
    for item in items:
        total = line_total(item)
        rows.append(f"{item.sku},{item.quantity},{format_currency(total, currency)}")
    return "\\n".join(rows)
''',
    "tests/test_commerce_decimal.py": '''from decimal import Decimal

from commerce.cart import LineItem, line_total, subtotal
from commerce.export_csv import export_lines_csv
from commerce.invoice import build_invoice
from commerce.money import quantize_amount
from commerce.refunds import prorate_refund


def test_quantize_uses_decimal_half_up():
    assert quantize_amount("2.675") == Decimal("2.68")
    assert quantize_amount(Decimal("1.005")) == Decimal("1.01")
    assert isinstance(quantize_amount("10.10"), Decimal)


def test_cart_avoids_binary_float_drift():
    items = [
        LineItem("a", "0.10", 1),
        LineItem("b", "0.20", 1),
        LineItem("c", "19.99", 3),
    ]
    assert line_total(items[2]) == Decimal("59.97")
    assert subtotal(items) == Decimal("60.27")


def test_invoice_exact_decimal_pipeline():
    items = [LineItem("sku1", "10.05", 1)]
    inv = build_invoice(items, discount_percent="10", tax_percent="8.5")
    # 10.05 * 10% = 1.005 -> HALF_UP 1.01 -> after_discount = 9.04
    # 9.04 * 8.5% = 0.7684 -> 0.77 -> total = 9.81
    assert inv["subtotal"] == Decimal("10.05")
    assert inv["after_discount"] == Decimal("9.04")
    assert inv["tax"] == Decimal("0.77")
    assert inv["total"] == Decimal("9.81")
    assert inv["formatted_total"] == "USD 9.81"


def test_refund_proration_half_up():
    assert prorate_refund("10.00", 1, 3) == Decimal("3.33")
    assert prorate_refund(Decimal("1.01"), 1, 2) == Decimal("0.51")


def test_export_csv_compatible():
    out = export_lines_csv([LineItem("pen", "1.25", 2)])
    assert out == "sku,qty,total\\npen,2,USD 2.50"
''',
}

B2_DECIMAL_REFACTOR = EvalTask(
    id="b2_price_decimal_refactor",
    suite="B",
    title="22文件金额精度重构：float 迁移至 Decimal(ROUND_HALF_UP)",
    description="涉及 money/cart/discounts/tax/invoice/refunds/export_csv 7个耦合模块的精度迁移。",
    prompt=(
        "将 `commerce/` 下的金额计算全面从 `float` 迁移为精确的 `Decimal`（保留两位小数，使用 `ROUND_HALF_UP`），"
        "兼容传入 `str` / `int` / `float` / `Decimal` 数值，并保证 `tests/test_commerce_decimal.py` 全部通过。"
        "不要修改测试文件。"
    ),
    files=_B2_FILES,
    protected_paths=("tests/test_commerce_decimal.py",),
    public_test_targets=("tests/test_commerce_decimal.py",),
    hidden_tests={
        "test_b2_hidden.py": '''from decimal import Decimal

from commerce.cart import LineItem
from commerce.invoice import build_invoice
from commerce.money import quantize_amount


def test_zero_tax_returns_decimal_zero():
    inv = build_invoice([LineItem("x", "5.00", 2)])
    assert isinstance(inv["tax"], Decimal)
    assert inv["tax"] == Decimal("0.00")
    assert inv["total"] == Decimal("10.00")


def test_float_input_converted_via_str_not_binary_noise():
    # Decimal(str(2.675)) -> 2.68 under ROUND_HALF_UP
    assert quantize_amount(2.675) == Decimal("2.68")
'''
    },
    expected_initial_f2p=(
        "tests/test_commerce_decimal.py::test_quantize_uses_decimal_half_up",
        "tests/test_commerce_decimal.py::test_cart_avoids_binary_float_drift",
        "tests/test_commerce_decimal.py::test_invoice_exact_decimal_pipeline",
        "tests/test_commerce_decimal.py::test_refund_proration_half_up",
    ),
    expected_initial_p2p=("tests/test_commerce_decimal.py::test_export_csv_compatible",),
    expected_routes=("react", "plan", "explore", "code_map", "collaboration"),
    max_turns=22,
    timeout_sec=180.0,
)


# ---------------------------------------------------------------------------
# B3：25文件跨三层隐蔽缓存失效链路 Bug 排查与修复
# ---------------------------------------------------------------------------

_B3_FILES: dict[str, str] = {
    **_distractor_helpers("support", 12),
    "events/__init__.py": "",
    "events/bus.py": '''"""Simple pub/sub event bus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._handlers[topic.strip()].append(handler)

    def publish(self, topic: str, payload: dict) -> None:
        # BUG 1: topic matching is case-sensitive and does not normalize whitespace/case
        for handler in list(self._handlers.get(topic, ())):
            handler(payload)
''',
    "cache/__init__.py": "",
    "cache/store.py": '''"""Prefix-invalidatable key/value store."""

from __future__ import annotations


class CacheStore:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def get(self, key: str):
        return self._data.get(key.lower())

    def set(self, key: str, value: object) -> None:
        self._data[key.lower()] = value

    def invalidate_prefix(self, prefix: str) -> int:
        # BUG 2: keys are stored lowercase, but prefix check did not lowercase prefix!
        removed = [k for k in self._data if k.startswith(prefix)]
        for k in removed:
            del self._data[k]
        return len(removed)
''',
    "services/__init__.py": "",
    "services/pricing_repo.py": '''from __future__ import annotations


class PricingRepo:
    def __init__(self) -> None:
        self._prices: dict[str, int] = {
            "BOOK-01": 1000,
            "BOOK-02": 2000,
            "TOY-01": 1500,
        }
        self._categories: dict[str, str] = {
            "BOOK-01": "Books",
            "BOOK-02": "Books",
            "TOY-01": "Toys",
        }

    def get_price(self, sku: str) -> int:
        return self._prices[sku]

    def get_category(self, sku: str) -> str:
        return self._categories[sku]

    def apply_category_discount(self, category: str, percent: int) -> list[str]:
        updated = []
        for sku, cat in self._categories.items():
            if cat.lower() == category.lower():
                self._prices[sku] = self._prices[sku] * (100 - percent) // 100
                updated.append(sku)
        return updated
''',
    "services/discount.py": '''from __future__ import annotations

from events.bus import EventBus
from services.pricing_repo import PricingRepo


class DiscountService:
    def __init__(self, repo: PricingRepo, bus: EventBus) -> None:
        self.repo = repo
        self.bus = bus

    def run_bulk_discount(self, category: str, percent: int) -> list[str]:
        updated = self.repo.apply_category_discount(category, percent)
        # BUG 3: publishes uppercase topic "Catalog.Discount.Applied" while listener subscribes lowercase,
        # AND emits key "cat" instead of "category" expected by CatalogApi listener
        self.bus.publish(
            "Catalog.Discount.Applied",
            {"cat": category, "skus": updated},
        )
        return updated
''',
    "api/__init__.py": "",
    "api/catalog.py": '''from __future__ import annotations

from cache.store import CacheStore
from events.bus import EventBus
from services.pricing_repo import PricingRepo


class CatalogApi:
    def __init__(self, repo: PricingRepo, cache: CacheStore, bus: EventBus) -> None:
        self.repo = repo
        self.cache = cache
        self.bus = bus
        self.bus.subscribe("catalog.discount.applied", self._on_discount_applied)

    def _cache_key(self, category: str, sku: str) -> str:
        return f"price:{category}:{sku}"

    def get_sku_price(self, sku: str) -> int:
        category = self.repo.get_category(sku)
        key = self._cache_key(category, sku)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        price = self.repo.get_price(sku)
        self.cache.set(key, price)
        return price

    def _on_discount_applied(self, payload: dict) -> None:
        category = payload["category"]
        self.cache.invalidate_prefix(f"price:{category}:")
''',
    "tests/test_catalog_invalidation.py": '''from api.catalog import CatalogApi
from cache.store import CacheStore
from events.bus import EventBus
from services.discount import DiscountService
from services.pricing_repo import PricingRepo


def test_bulk_discount_invalidates_only_target_category_cache():
    repo = PricingRepo()
    cache = CacheStore()
    bus = EventBus()
    api = CatalogApi(repo, cache, bus)
    discounts = DiscountService(repo, bus)

    # Prime cache for Books and Toys
    assert api.get_sku_price("BOOK-01") == 1000
    assert api.get_sku_price("BOOK-02") == 2000
    assert api.get_sku_price("TOY-01") == 1500

    # Apply 20% discount to Books
    discounts.run_bulk_discount("Books", 20)

    # Books cache must be invalidated and return updated prices
    assert api.get_sku_price("BOOK-01") == 800
    assert api.get_sku_price("BOOK-02") == 1600
    # Toys cache must still remain cached
    assert cache.get("price:Toys:TOY-01") == 1500
''',
}

B3_CACHE_CHAIN = EvalTask(
    id="b3_hidden_cache_invalidation_bug",
    suite="B",
    title="25文件跨三层排障：批量折扣后商品缓存未失效",
    description="症状在 api/catalog.py，真实根因横跨 events/bus.py、cache/store.py、services/discount.py。",
    prompt=(
        "`tests/test_catalog_invalidation.py` 失败：执行批量分类折扣后，通过 `CatalogApi.get_sku_price` "
        "读到的仍然是过期缓存价格。请追踪完整事件与缓存失效链路，修复所有相关模块的缺陷，"
        "确保只失效目标分类缓存且测试全部通过。不要修改测试文件。"
    ),
    files=_B3_FILES,
    protected_paths=("tests/test_catalog_invalidation.py",),
    public_test_targets=("tests/test_catalog_invalidation.py",),
    hidden_tests={
        "test_b3_hidden.py": '''from cache.store import CacheStore
from events.bus import EventBus


def test_cache_invalidate_prefix_case_insensitive():
    store = CacheStore()
    store.set("Price:Books:B1", 10)
    store.set("Price:Toys:T1", 20)
    removed = store.invalidate_prefix("Price:Books:")
    assert removed == 1
    assert store.get("price:books:b1") is None
    assert store.get("price:toys:t1") == 20


def test_event_bus_case_insensitive_topics():
    bus = EventBus()
    seen = []
    bus.subscribe("Orders.Updated", lambda p: seen.append(p["id"]))
    bus.publish("orders.updated", {"id": 7})
    assert seen == [7]
'''
    },
    expected_initial_f2p=(
        "tests/test_catalog_invalidation.py::test_bulk_discount_invalidates_only_target_category_cache",
    ),
    expected_initial_p2p=(),
    expected_routes=("react", "plan", "explore", "code_map", "collaboration"),
    max_turns=22,
    timeout_sec=180.0,
)

SUITE_B_TASKS: tuple[EvalTask, ...] = (B1_TENANT_CONTEXT, B2_DECIMAL_REFACTOR, B3_CACHE_CHAIN)
