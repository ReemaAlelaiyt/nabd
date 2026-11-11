"""Simulated healthcare logistics environment (Section 3.4).

Holds facilities, products, batches, refrigerated vehicles, routes, active
deliveries and simulated temperature readings. The environment is stochastic:
configurable random events (demand spikes, temperature excursions, vehicle
breakdowns) can occur at each discrete simulation step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .models import Delivery, Facility, Product, Vehicle


@dataclass
class EnvEvent:
    step: int
    kind: str
    text: str


class Environment:
    def __init__(self, products: dict[str, Product], facilities: dict[str, Facility],
                 vehicles: dict[str, Vehicle], travel_times: dict[tuple[str, str], int],
                 config, seed: Optional[int] = None):
        self.products = products
        self.facilities = facilities
        self.vehicles = vehicles
        self._travel = {}
        for (a, b), minutes in travel_times.items():
            self._travel[(a, b)] = minutes
            self._travel[(b, a)] = minutes
        self.config = config
        self.rng = random.Random(seed)
        self.step = 0
        self.deliveries: dict[str, Delivery] = {}
        self.storage_readings: dict[str, float] = {}   # batch_id -> temperature_reading
        self.shipment_readings: dict[str, float] = {}  # delivery_id -> temperature_reading
        self.breakdowns: set[str] = set()              # delivery ids that break down this step
        self.events: list[EnvEvent] = []
        self._forced: dict[str, tuple[float, int]] = {}  # target id -> (reading, last step)
        self._armed_shipment_excursion: Optional[float] = None
        self._forced_breakdowns: set[str] = set()
        self._delivery_counter = 0

    # ---- helpers ------------------------------------------------------------
    def travel_time(self, a: Optional[str], b: str) -> int:
        if a is None or a == b:
            return 0
        return self._travel[(a, b)]

    def batch_location(self, batch_id: str) -> Optional[Facility]:
        for f in self.facilities.values():
            if f.find_batch(batch_id):
                return f
        return None

    def new_delivery_id(self) -> str:
        self._delivery_counter += 1
        return f"SH-{self._delivery_counter:02d}"

    def record(self, kind: str, text: str) -> None:
        self.events.append(EnvEvent(self.step, kind, text))

    # ---- external injections (used by scenarios / UI) -----------------------
    def force_excursion(self, target_id: str, reading: float, steps: int = 1) -> None:
        """Force an out-of-range reading on a batch or shipment for the next `steps` steps."""
        self._forced[target_id] = (reading, self.step + steps)

    def force_breakdown(self, delivery_id: str) -> None:
        """The vehicle carrying this delivery breaks down at the next step."""
        self._forced_breakdowns.add(delivery_id)

    def arm_shipment_excursion(self, reading: float) -> None:
        """The next shipment that is in transit will report this reading."""
        self._armed_shipment_excursion = reading

    # ---- dynamics -----------------------------------------------------------
    def update(self, step: int) -> None:
        self.step = step
        self.breakdowns = {d for d in self._forced_breakdowns
                           if d in self.deliveries and self.deliveries[d].status == "in_progress"}
        self._forced_breakdowns -= self.breakdowns
        self._random_events()
        self._generate_readings()

    def _random_events(self) -> None:
        cfg, rng = self.config, self.rng
        # Demand spike: a facility uses part of its eligible stock.
        if cfg.p_demand_spike > 0 and rng.random() < cfg.p_demand_spike:
            options = [(f, p) for f in self.facilities.values() for p in f.products() if f.eligible_stock(p) > 0]
            if options:
                f, p = rng.choice(options)
                q = max(1, int(f.eligible_stock(p) * rng.uniform(0.2, 0.5)))
                used = f.consume(p, q)
                self.record("demand_spike", f"Demand spike at {f.name}: {used} {self.products[p].unit} of {self.products[p].name} used")
        # Temperature excursion on a random storage batch or shipment.
        if cfg.p_temp_excursion > 0 and rng.random() < cfg.p_temp_excursion:
            targets = [b.batch_id for f in self.facilities.values() for b in f.batches if not b.flagged]
            targets += [d.delivery_id for d in self.deliveries.values() if d.status == "in_progress" and not d.flagged]
            if targets:
                t = rng.choice(targets)
                self._forced[t] = (self._excursion_reading(self._product_of(t)), self.step)
                self.record("temp_excursion", f"Random temperature excursion at {t}")
        # Vehicle breakdown during transit.
        if cfg.p_vehicle_breakdown > 0:
            for d in self.deliveries.values():
                if d.status == "in_progress" and rng.random() < cfg.p_vehicle_breakdown:
                    self.breakdowns.add(d.delivery_id)
                    self.record("breakdown", f"Vehicle {d.vehicle_id} broke down carrying {d.delivery_id}")

    def _product_of(self, target_id: str) -> Product:
        if target_id in self.deliveries:
            return self.products[self.deliveries[target_id].product_id]
        f = self.batch_location(target_id)
        return self.products[f.find_batch(target_id).product_id]

    def _normal_reading(self, p: Product) -> float:
        mid = (p.temp_min + p.temp_max) / 2
        spread = (p.temp_max - p.temp_min) / 8
        return round(min(p.temp_max, max(p.temp_min, self.rng.gauss(mid, spread))), 1)

    def _excursion_reading(self, p: Product) -> float:
        return round(p.temp_max + self.rng.uniform(2.0, 6.0), 1)

    def _reading_for(self, target_id: str, p: Product) -> float:
        forced = self._forced.get(target_id)
        if forced and self.step <= forced[1]:
            return forced[0]
        if forced:
            del self._forced[target_id]
        return self._normal_reading(p)

    def _generate_readings(self) -> None:
        self.storage_readings = {}
        for f in self.facilities.values():
            for b in f.batches:
                self.storage_readings[b.batch_id] = self._reading_for(b.batch_id, self.products[b.product_id])
        self.shipment_readings = {}
        for d in self.deliveries.values():
            if d.status != "in_progress":
                continue
            if self._armed_shipment_excursion is not None and not d.flagged:
                self._forced[d.delivery_id] = (self._armed_shipment_excursion, self.step)
                self._armed_shipment_excursion = None
            self.shipment_readings[d.delivery_id] = self._reading_for(d.delivery_id, self.products[d.product_id])
