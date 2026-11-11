"""Hospital Inventory Agent - one per hospital or warehouse (reactive).

Behaviors (Section 3.3): stock check, supply response, reservation, transfer update.
"""
from __future__ import annotations

from ..messages import ACLMessage, Ontology, Performative
from ..models import Batch, compute_priority
from .base import BaseAgent

P, O = Performative, Ontology


class HospitalInventoryAgent(BaseAgent):
    agent_type = "Hospital Inventory Agent"
    architecture = "Reactive"

    def __init__(self, agent_id: str, sim, facility_id: str):
        super().__init__(agent_id, sim)
        self.facility_id = facility_id
        self.open_requests: dict[str, str] = {}     # product -> request id ("pending" until confirmed)
        self.cooldown_until: dict[str, int] = {}    # product -> step before re-alerting after unresolved
        self.last_eligible: dict[str, int] = {}     # baseline to detect unexplained eligible changes
        self.reservations: dict[str, tuple[str, int, str]] = {}  # allocation id -> (product, qty, request id)

    @property
    def facility(self):
        return self.env.facilities[self.facility_id]

    def _pname(self, pid: str) -> str:
        return self.env.products[pid].name

    def _unit(self, pid: str) -> str:
        return self.env.products[pid].unit

    # ---- message handling ---------------------------------------------------
    def handle(self, m: ACLMessage) -> None:
        if m.performative == P.CFP:
            self._on_cfp(m)
        elif m.performative == P.ACCEPT:
            self._on_accept(m)
        elif m.performative == P.REJECT_PROPOSAL:
            self.log(f"Offer for {m.conversation_id} not selected")
        elif m.performative == P.REQUEST and m.topic == "dispatch":
            self._on_dispatch(m)
        elif m.performative == P.INFORM:
            handler = {
                "request_opened": self._on_request_opened,
                "request_closed": self._on_request_closed,
                "release_reservation": self._on_release,
                "receipt": self._on_receipt,
                "return_cargo": self._on_return_cargo,
                "batch_flagged": self._on_cold_chain_notice,
                "assessment": self._on_cold_chain_notice,
            }.get(m.topic)
            if handler:
                handler(m)

    def _on_cfp(self, m: ACLMessage) -> None:
        c, f = m.content, self.facility
        pid = c["product"]
        if not f.responsive:
            self.log(f"Facility offline - no reply to CFP {m.conversation_id}", "warning")
            return
        available = f.transferable_stock(pid) if pid in f.products() else 0
        if available > 0:
            q = min(available, c["quantity"])
            route = self.env.travel_time(self.facility_id, c["requester"])
            self.send(P.PROPOSE, m.sender, O.SUPPLY, m.conversation_id,
                      f"Offer {q} {self._unit(pid)} of {self._pname(pid)}; route {route} minutes",
                      topic="offer", request_id=c["request_id"], round=c["round"], product=pid,
                      quantity=q, travel_time=route)
        else:
            self.send(P.REFUSE, m.sender, O.SUPPLY, m.conversation_id,
                      f"No transferable {self._pname(pid)} stock",
                      topic="cfp_refused", request_id=c["request_id"], round=c["round"])

    def _on_accept(self, m: ACLMessage) -> None:
        c, f = m.content, self.facility
        if not f.responsive:
            self.log(f"Facility offline - no reply to ACCEPT {c['allocation_id']}", "warning")
            return
        pid, q = c["product"], c["quantity"]
        available = f.transferable_stock(pid)  # recheck on acceptance
        if available >= q:
            f.reserved_stock[pid] = f.reserved(pid) + q
            self.reservations[c["allocation_id"]] = (pid, q, c["request_id"])
            self.send(P.INFORM, m.sender, O.SUPPLY, m.conversation_id,
                      f"{q} {self._unit(pid)} reserved for {c['request_id']}",
                      topic="reserved", allocation_id=c["allocation_id"], quantity=q)
            self.log(f"Reserved {q} {self._unit(pid)} of {self._pname(pid)} for {c['request_id']}", "success")
        else:
            self.send(P.REFUSE, m.sender, O.SUPPLY, m.conversation_id,
                      f"Reservation failed: quantity no longer available (only {available})",
                      topic="reservation_failed", allocation_id=c["allocation_id"], available=available)
            self.log(f"Could not reserve {q} {self._unit(pid)} (only {available} transferable)", "warning")

    def _on_dispatch(self, m: ACLMessage) -> None:
        c, f = m.content, self.facility
        res = self.reservations.pop(c["allocation_id"], None)
        if res is None:
            self.send(P.REFUSE, m.sender, O.DELIVERY, m.conversation_id, "Dispatch failed: no reservation",
                      topic="dispatch_failed", delivery_id=c["delivery_id"], reason="no reservation")
            return
        pid, q, _ = res
        f.reserved_stock[pid] = max(0, f.reserved(pid) - q)
        if f.eligible_stock(pid) >= q:
            delivery = self.env.deliveries[c["delivery_id"]]
            delivery.cargo = f.take_eligible(pid, q)
            if pid in self.last_eligible:
                self.last_eligible[pid] -= q  # explained change
            self.send(P.INFORM, m.sender, O.DELIVERY, m.conversation_id,
                      f"{q} {self._unit(pid)} loaded on {delivery.vehicle_id} ({c['delivery_id']})",
                      topic="dispatched", delivery_id=c["delivery_id"], quantity=q)
            self.log(f"Dispatched {q} {self._unit(pid)} of {self._pname(pid)} ({c['delivery_id']})")
        else:
            self.send(P.REFUSE, m.sender, O.DELIVERY, m.conversation_id,
                      "Dispatch failed: reserved stock is no longer eligible (flagged)",
                      topic="dispatch_failed", delivery_id=c["delivery_id"], reason="stock no longer eligible")
            self.log(f"Dispatch of {c['delivery_id']} failed - reserved stock flagged", "warning")

    def _on_request_opened(self, m: ACLMessage) -> None:
        self.open_requests[m.content["product"]] = m.content["request_id"]

    def _on_request_closed(self, m: ACLMessage) -> None:
        pid, status = m.content["product"], m.content["status"]
        self.open_requests.pop(pid, None)
        if status == "unresolved":
            self.cooldown_until[pid] = self.now + self.config.retry_cooldown
        self.log(f"{m.conversation_id} closed as {status}", "success" if status == "fulfilled" else "warning")

    def _on_release(self, m: ACLMessage) -> None:
        res = self.reservations.pop(m.content["allocation_id"], None)
        if res:
            pid, q, _ = res
            self.facility.reserved_stock[pid] = max(0, self.facility.reserved(pid) - q)
            self.log(f"Released reservation of {q} {self._unit(pid)} ({m.content['allocation_id']})")

    def _on_receipt(self, m: ACLMessage) -> None:
        c = m.content
        pid, q, flagged = c["product"], c["quantity"], c["flagged"]
        self.facility.batches.append(Batch(c["batch_id"], pid, q, flagged=flagged,
                                           assessment="FLAGGED" if flagged else "OK", origin=c["supplier"]))
        if flagged:
            self.log(f"Received {q} {self._unit(pid)} in flagged batch {c['batch_id']} - excluded until assessed", "alert")
        else:
            if pid in self.last_eligible:
                self.last_eligible[pid] += q  # explained change
            self.log(f"Received {q} {self._unit(pid)} of {self._pname(pid)} ({c['delivery_id']})", "success")

    def _on_return_cargo(self, m: ACLMessage) -> None:
        c = m.content
        self.facility.batches.append(Batch(c["batch_id"], c["product"], c["quantity"], flagged=True,
                                           assessment="FLAGGED", origin=self.facility_id))
        self.log(f"Returned cargo {c['batch_id']} stored as flagged (needs assessment)", "warning")

    def _on_cold_chain_notice(self, m: ACLMessage) -> None:
        self.log(f"Cold-chain notice: {m.text}", "alert" if m.topic == "batch_flagged" else "info")

    # ---- periodic behavior: stock check -------------------------------------
    def act(self) -> None:
        f = self.facility
        shortages = []
        for pid in f.products():
            eligible, thr = f.eligible_stock(pid), f.threshold(pid)
            baseline = self.last_eligible.get(pid)
            rid = self.open_requests.get(pid)
            # Unexplained change (consumption, flag, release) while a request is open -> update Coordination.
            if baseline is not None and eligible != baseline and rid not in (None, "pending") and thr > 0:
                delta = max(0, thr - eligible) - max(0, thr - baseline)
                if delta:
                    self.send(P.INFORM, "Coordination", O.INVENTORY, rid,
                              f"Shortage {'increased' if delta > 0 else 'reduced'} by {abs(delta)} "
                              f"{self._unit(pid)} (eligible now {eligible})",
                              topic="shortage_update", request_id=rid, product=pid, delta=delta, eligible=eligible)
            self.last_eligible[pid] = eligible

            if thr <= 0 or eligible >= thr:
                continue
            shortages.append(pid)
            if pid in self.open_requests or self.now < self.cooldown_until.get(pid, 0):
                continue
            shortage, priority = thr - eligible, compute_priority(eligible, thr)
            self.send(P.INFORM, "Coordination", O.INVENTORY, f"SHORT-{self.facility_id}-{pid}-{self.now}",
                      f"{f.name}: shortage of {shortage} {self._unit(pid)} of {self._pname(pid)}; "
                      f"{priority} priority",
                      topic="shortage", facility=self.facility_id, product=pid, eligible=eligible,
                      threshold=thr, quantity=shortage, priority=priority)
            self.open_requests[pid] = "pending"
            self.log(f"Shortage detected: {self._pname(pid)} {eligible}/{thr} ({priority})", "alert")

        if self.reservations:
            self.set_status("active", f"Holding {len(self.reservations)} reservation(s) for other facilities")
        elif shortages:
            names = ", ".join(self._pname(p) for p in shortages)
            self.set_status("alert", f"Shortage: {names}")
        elif not f.responsive:
            self.set_status("alert", "Facility offline (not responding)")
        else:
            self.set_status("idle", "Stock above minimum thresholds")
