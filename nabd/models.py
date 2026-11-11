"""Data models and simulation rules (Sections 3.4 and 3.5).

Key rules implemented here:
  eligible_stock     = total_stock - flagged_stock
  transferable_stock = max(0, eligible_stock - minimum_stock_threshold - reserved_stock)
  request_priority   = High (<25% of minimum), Medium (25-50% inclusive), Low (>50% but below minimum)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def compute_priority(eligible_stock: int, minimum_stock_threshold: int) -> Optional[str]:
    """Return High / Medium / Low, or None when there is no shortage."""
    if minimum_stock_threshold <= 0:
        raise ValueError("minimum_stock_threshold must be positive")
    if eligible_stock >= minimum_stock_threshold:
        return None
    ratio = eligible_stock / minimum_stock_threshold
    if ratio < 0.25:
        return "High"
    if ratio <= 0.50:
        return "Medium"
    return "Low"


@dataclass
class Product:
    product_id: str
    name: str
    unit: str
    temp_min: float  # temperature_limits (deg C)
    temp_max: float

    def in_range(self, reading: float) -> bool:
        return self.temp_min <= reading <= self.temp_max


@dataclass
class Batch:
    batch_id: str
    product_id: str
    quantity: int
    flagged: bool = False
    assessment: str = "OK"  # OK / FLAGGED / RELEASED
    origin: str = ""


@dataclass
class Facility:
    facility_id: str
    name: str
    kind: str  # "hospital" or "warehouse"
    batches: list[Batch] = field(default_factory=list)
    minimum_stock_threshold: dict[str, int] = field(default_factory=dict)
    reserved_stock: dict[str, int] = field(default_factory=dict)
    responsive: bool = True  # False = does not reply to CFP / ACCEPT (simulates timeout)

    # ---- stock calculations -------------------------------------------------
    def products(self) -> list[str]:
        ids = {b.product_id for b in self.batches} | set(self.minimum_stock_threshold)
        return sorted(ids)

    def threshold(self, product_id: str) -> int:
        return self.minimum_stock_threshold.get(product_id, 0)

    def reserved(self, product_id: str) -> int:
        return self.reserved_stock.get(product_id, 0)

    def total_stock(self, product_id: str) -> int:
        return sum(b.quantity for b in self.batches if b.product_id == product_id)

    def flagged_stock(self, product_id: str) -> int:
        return sum(b.quantity for b in self.batches if b.product_id == product_id and b.flagged)

    def eligible_stock(self, product_id: str) -> int:
        return self.total_stock(product_id) - self.flagged_stock(product_id)

    def transferable_stock(self, product_id: str) -> int:
        return max(0, self.eligible_stock(product_id) - self.threshold(product_id) - self.reserved(product_id))

    # ---- stock changes ------------------------------------------------------
    def find_batch(self, batch_id: str) -> Optional[Batch]:
        return next((b for b in self.batches if b.batch_id == batch_id), None)

    def take_eligible(self, product_id: str, quantity: int) -> list[tuple[str, int]]:
        """Remove `quantity` units from unflagged batches (FIFO). Returns [(batch_id, qty)]."""
        taken: list[tuple[str, int]] = []
        remaining = quantity
        for b in self.batches:
            if remaining == 0:
                break
            if b.product_id != product_id or b.flagged or b.quantity == 0:
                continue
            q = min(b.quantity, remaining)
            b.quantity -= q
            remaining -= q
            taken.append((b.batch_id, q))
        self.batches = [b for b in self.batches if b.quantity > 0]
        if remaining:
            raise ValueError(f"{self.facility_id}: not enough eligible {product_id}")
        return taken

    def consume(self, product_id: str, quantity: int) -> int:
        """Demand/usage event: remove up to `quantity` eligible units. Returns amount consumed."""
        q = min(quantity, self.eligible_stock(product_id))
        if q > 0:
            self.take_eligible(product_id, q)
        return q


@dataclass
class Vehicle:
    vehicle_id: str
    name: str
    capacity: int  # vehicle_capacity
    temp_min: float
    temp_max: float
    location: Optional[str]  # facility id, None while travelling
    available: bool = True  # vehicle_availability (in service)
    busy: bool = False

    @property
    def status(self) -> str:
        if not self.available:
            return "unavailable"
        return "busy" if self.busy else "available"

    def compatible_with(self, product: Product) -> bool:
        """The vehicle's controlled range must lie inside the product's permitted range."""
        return product.temp_min <= self.temp_min and self.temp_max <= product.temp_max


@dataclass
class Delivery:
    delivery_id: str
    request_id: str
    allocation_id: str
    product_id: str
    quantity: int
    supplier: str
    receiver: str
    vehicle_id: str
    eta_minutes: int
    assigned_step: int
    status: str = "assigned"  # delivery_status: assigned / in_progress / completed / failed / flagged
    steps_remaining: int = 0
    started_step: Optional[int] = None
    finished_step: Optional[int] = None
    flagged: bool = False
    cargo: list[tuple[str, int]] = field(default_factory=list)
    failure_reason: str = ""

    @property
    def active(self) -> bool:
        return self.status in ("assigned", "in_progress")
