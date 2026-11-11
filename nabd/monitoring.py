"""Monitoring: event log, message log, per-step metrics history and exports.

The interface (Implementation pt2) can read everything it needs from
`Simulation.snapshot()`, `Monitor.events`, `Monitor.history` and the message log.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Simulation


@dataclass
class LogEvent:
    step: int
    agent: str
    text: str
    level: str  # info / success / warning / alert

    def __str__(self) -> str:
        icon = {"info": " ", "success": "+", "warning": "!", "alert": "*"}.get(self.level, " ")
        return f"[step {self.step:>3}] {icon} {self.agent:<14} {self.text}"


class Monitor:
    def __init__(self, sim: "Simulation"):
        self.sim = sim
        self.events: list[LogEvent] = []
        self.history: list[dict] = []
        self._listeners = []

    # ---- logging ------------------------------------------------------------
    def log_event(self, step: int, agent: str, text: str, level: str = "info") -> None:
        ev = LogEvent(step, agent, text, level)
        self.events.append(ev)
        for fn in self._listeners:
            fn(ev)

    def subscribe(self, fn) -> None:
        """Register a callback fn(LogEvent) - e.g. to print or push into a GUI widget."""
        self._listeners.append(fn)

    # ---- metrics ------------------------------------------------------------
    def metrics(self) -> dict:
        sim = self.sim
        reqs = sim.coordination.requests.values()
        deliveries = sim.env.deliveries.values()
        perf = Counter(m.performative.value for m in sim.bus.log)
        flagged_batches = [b for f in sim.env.facilities.values() for b in f.batches if b.flagged]
        return {
            "step": sim.step_count,
            "requests_total": len(reqs),
            "requests_open": sum(1 for r in reqs if r.active),
            "requests_fulfilled": sum(1 for r in reqs if r.status == "fulfilled"),
            "requests_unresolved": sum(1 for r in reqs if r.status == "unresolved"),
            "deliveries_active": sum(1 for d in deliveries if d.active),
            "deliveries_completed": sum(1 for d in deliveries if d.status == "completed"),
            "deliveries_failed": sum(1 for d in deliveries if d.status == "failed"),
            "deliveries_flagged": sum(1 for d in deliveries if d.status == "flagged"),
            "flagged_batches": len(flagged_batches),
            "flagged_quantity": sum(b.quantity for b in flagged_batches),
            "excursions_detected": sim.cold_chain.excursions_detected,
            "vehicles_busy": sum(1 for v in sim.env.vehicles.values() if v.busy),
            "vehicles_unavailable": sum(1 for v in sim.env.vehicles.values() if not v.available),
            "messages_total": len(sim.bus.log),
            "messages_this_step": sum(1 for m in sim.bus.log if m.sent_step == sim.step_count),
            "messages_by_performative": dict(perf),
        }

    def record(self) -> dict:
        row = self.metrics()
        self.history.append(row)
        return row

    # ---- exports ------------------------------------------------------------
    def export_messages_csv(self, path: str) -> None:
        rows = [m.to_dict() for m in self.sim.bus.log]
        fields = ["id", "step", "sender", "receiver", "performative", "ontology", "conversation_id", "content"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def export_json(self, path: str) -> None:
        data = {
            "snapshot": self.sim.snapshot(message_limit=None, event_limit=None),
            "history": self.history,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    def events_as_dicts(self, limit: int | None = None) -> list[dict]:
        evs = self.events if limit is None else self.events[-limit:]
        return [asdict(e) for e in evs]
