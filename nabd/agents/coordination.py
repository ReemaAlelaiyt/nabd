"""Emergency Coordination Agent - one shared agent (deliberative, AI-based).

The offer-evaluation decision is made by an LLM (see ../llm_decider.py); the answer is
validated against the rules and falls back to "quickest supplier first" if needed.

Behaviors (Section 3.3): open request, evaluate offers, allocate stock,
confirm and arrange delivery, recover or close.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..llm_decider import decide_allocation
from ..messages import ACLMessage, Ontology, Performative
from .base import BaseAgent

P, O = Performative, Ontology

ACTIVE_ALLOCATION = ("accepted", "reserved", "assigned", "in_progress")


@dataclass
class Allocation:
    allocation_id: str
    supplier: str          # facility id
    quantity: int
    travel_time: int
    accepted_step: int
    status: str = "accepted"  # accepted/reserved/assigned/in_progress/delivered/flagged/failed/refused
    delivery_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    eta_minutes: Optional[int] = None
    note: str = ""


@dataclass
class EmergencyRequest:
    request_id: str
    facility: str
    product: str
    priority: str
    opened_step: int
    initial_shortage: int
    remaining_shortage: int
    confirmed_inbound: int = 0
    received: int = 0
    status: str = "open"           # request_status: open / partial / fulfilled / unresolved
    phase: str = "collecting"      # collecting / allocating / waiting_retry / closed
    round: int = 0
    cfp_step: int = 0
    deadline_step: int = 0
    next_round_step: int = 0
    awaiting: set = field(default_factory=set)
    proposals: dict = field(default_factory=dict)      # supplier -> {"quantity", "travel_time"}
    refusals: list = field(default_factory=list)
    timeouts: list = field(default_factory=list)
    allocations: list = field(default_factory=list)
    closed_step: Optional[int] = None
    history: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.phase != "closed"

    @property
    def total_needed(self) -> int:
        """Received so far plus what is still missing (grows if the shortage worsens)."""
        return self.received + self.remaining_shortage

    def pending_quantity(self) -> int:
        return sum(a.quantity for a in self.allocations if a.status in ACTIVE_ALLOCATION)


class EmergencyCoordinationAgent(BaseAgent):
    agent_type = "Emergency Coordination Agent"
    architecture = "Deliberative (AI)"

    def __init__(self, agent_id: str, sim):
        super().__init__(agent_id, sim)
        self.requests: dict[str, EmergencyRequest] = {}
        self._open_index: dict[tuple[str, str], str] = {}   # (facility, product) -> request id
        self._allocations: dict[str, tuple[EmergencyRequest, Allocation]] = {}
        self._req_counter = 0
        self.decision_log: list[dict] = []   # every AI / fallback decision with its reasoning

    # ---- helpers ------------------------------------------------------------
    def _unit(self, pid):
        return self.env.products[pid].unit

    def _note(self, req: EmergencyRequest, text: str, level: str = "info") -> None:
        req.history.append((self.now, text))
        self.log(f"{req.request_id}: {text}", level)

    def _alloc(self, allocation_id: str):
        return self._allocations.get(allocation_id, (None, None))

    # ---- message handling ---------------------------------------------------
    def handle(self, m: ACLMessage) -> None:
        t, perf = m.topic, m.performative
        if perf == P.INFORM and t == "shortage":
            self._on_shortage(m)
        elif perf == P.INFORM and t == "shortage_update":
            req = self.requests.get(m.content["request_id"])
            if req and req.active:
                req.remaining_shortage = max(0, req.remaining_shortage + m.content["delta"])
                self._note(req, f"Remaining shortage updated to {req.remaining_shortage} ({m.text})")
        elif perf == P.PROPOSE:
            self._on_propose(m)
        elif perf == P.REFUSE and t == "cfp_refused":
            req = self.requests.get(m.content["request_id"])
            supplier = self.sim.facility_of_agent(m.sender)
            if req and m.content["round"] == req.round:
                req.awaiting.discard(supplier)
                req.refusals.append(supplier)
        elif perf == P.INFORM and t == "reserved":
            self._on_reserved(m)
        elif perf == P.REFUSE and t == "reservation_failed":
            req, a = self._alloc(m.content["allocation_id"])
            if a and a.status == "accepted":
                a.status, a.note = "refused", "reservation refused"
                self._note(req, f"{self.sim.facility_name(a.supplier)} refused reservation of {a.quantity}", "warning")
        elif perf == P.INFORM and t == "vehicle_assigned":
            req, a = self._alloc(m.content["allocation_id"])
            if a:
                a.status, a.delivery_id = "assigned", m.content["delivery_id"]
                a.vehicle_id, a.eta_minutes = m.content["vehicle_id"], m.content["eta_minutes"]
                self._note(req, f"{a.vehicle_id} assigned to {a.delivery_id}, ETA {a.eta_minutes} min")
        elif perf == P.REFUSE and t == "no_vehicle":
            self._on_transport_refused(m)
        elif perf == P.INFORM and t == "delivery_started":
            req, a = self._alloc(m.content["allocation_id"])
            if a:
                a.status = "in_progress"
        elif perf == P.INFORM and t in ("delivery_completed", "delivery_flagged", "delivery_failed"):
            self._on_delivery_outcome(m)
        elif perf == P.INFORM and t in ("shipment_flagged", "batch_flagged"):
            self.log(f"Cold-chain alert received: {m.text}", "alert")

    def _on_shortage(self, m: ACLMessage) -> None:
        c = m.content
        key = (c["facility"], c["product"])
        if key in self._open_index:  # one open emergency request per facility/product
            rid = self._open_index[key]
            self.log(f"Duplicate shortage alert ignored; {rid} already open", "warning")
            self.send(P.INFORM, m.sender, O.INVENTORY, rid, f"Request {rid} already open",
                      topic="request_opened", request_id=rid, product=c["product"])
            return
        self._req_counter += 1
        rid = f"REQ-{self._req_counter:03d}"
        req = EmergencyRequest(rid, c["facility"], c["product"], c["priority"], self.now,
                               c["quantity"], c["quantity"])
        self.requests[rid] = req
        self._open_index[key] = rid
        self.send(P.INFORM, m.sender, O.INVENTORY, rid, f"Emergency request {rid} opened",
                  topic="request_opened", request_id=rid, product=c["product"])
        self._note(req, f"Opened for {c['quantity']} {self._unit(c['product'])} at "
                        f"{self.sim.facility_name(c['facility'])} ({c['priority']} priority)", "alert")
        self._start_round(req)

    def _on_propose(self, m: ACLMessage) -> None:
        c = m.content
        req = self.requests.get(c["request_id"])
        supplier = self.sim.facility_of_agent(m.sender)
        if not req or req.phase != "collecting" or c["round"] != req.round:
            self.send(P.REJECT_PROPOSAL, m.sender, O.SUPPLY, m.conversation_id,
                      "Proposal arrived after the response deadline", topic="late_offer")
            if req:
                self._note(req, f"Late offer from {self.sim.facility_name(supplier)} rejected", "warning")
            return
        req.awaiting.discard(supplier)
        req.proposals[supplier] = {"quantity": c["quantity"], "travel_time": c["travel_time"]}

    def _on_reserved(self, m: ACLMessage) -> None:
        req, a = self._alloc(m.content["allocation_id"])
        if not a or a.status != "accepted":
            return
        a.status = "reserved"
        req.confirmed_inbound += a.quantity
        self._note(req, f"{self.sim.facility_name(a.supplier)} confirmed reservation of {a.quantity}", "success")
        # Only after reservation confirmation: request delivery from Transport.
        self.send(P.REQUEST, "Transport", O.DELIVERY, req.request_id,
                  f"Deliver {a.quantity} {self._unit(req.product)} from "
                  f"{self.sim.facility_name(a.supplier)} to {self.sim.facility_name(req.facility)}",
                  topic="deliver", request_id=req.request_id, allocation_id=a.allocation_id,
                  supplier=a.supplier, receiver=req.facility, product=req.product, quantity=a.quantity)

    def _on_transport_refused(self, m: ACLMessage) -> None:
        req, a = self._alloc(m.content["allocation_id"])
        if not a or a.status not in ("reserved", "assigned"):
            return
        a.status, a.note = "failed", m.content.get("reason", "no vehicle")
        req.confirmed_inbound -= a.quantity
        # Release the canceled reservation at the supplier.
        self.send(P.INFORM, self.sim.inventory_agent_id(a.supplier), O.SUPPLY, req.request_id,
                  f"Release reservation {a.allocation_id} ({a.quantity} {self._unit(req.product)}): transport unavailable",
                  topic="release_reservation", allocation_id=a.allocation_id)
        self._note(req, f"Transport refused ({a.note}); reservation released", "warning")

    def _on_delivery_outcome(self, m: ACLMessage) -> None:
        req, a = self._alloc(m.content["allocation_id"])
        if not a or a.status not in ("reserved", "assigned", "in_progress"):
            return
        req.confirmed_inbound -= a.quantity
        if m.topic == "delivery_completed":
            a.status = "delivered"
            req.received += a.quantity
            req.remaining_shortage = max(0, req.remaining_shortage - a.quantity)
            self._note(req, f"{a.delivery_id} delivered {a.quantity}; remaining {req.remaining_shortage}", "success")
        elif m.topic == "delivery_flagged":
            a.status, a.note = "flagged", "shipment temperature-flagged"
            self._note(req, f"{a.delivery_id} arrived flagged - not counted; seeking replacement", "alert")
        else:
            a.status, a.note = "failed", m.content.get("reason", "failed")
            self._note(req, f"{a.delivery_id or a.allocation_id} failed ({a.note})", "warning")

    # ---- deliberation -------------------------------------------------------
    def _start_round(self, req: EmergencyRequest) -> None:
        req.round += 1
        need = max(0, req.remaining_shortage - req.pending_quantity())
        suppliers = [fid for fid in self.env.facilities if fid != req.facility]
        req.phase, req.cfp_step = "collecting", self.now
        req.deadline_step = self.now + self.config.cfp_response_deadline
        req.awaiting, req.proposals, req.refusals, req.timeouts = set(suppliers), {}, [], []
        for fid in suppliers:
            self.send(P.CFP, self.sim.inventory_agent_id(fid), O.SUPPLY, req.request_id,
                      f"Request {self.env.products[req.product].name} supply offer for {req.request_id} "
                      f"({need} {self._unit(req.product)}, {req.priority})",
                      topic="cfp", request_id=req.request_id, round=req.round, product=req.product,
                      quantity=need, requester=req.facility, priority=req.priority)
        self._note(req, f"CFP round {req.round} sent to {len(suppliers)} facilities (deadline step {req.deadline_step})")

    def _evaluate(self, req: EmergencyRequest) -> None:
        for s in sorted(req.awaiting):
            req.timeouts.append(s)
            self._note(req, f"{self.sim.facility_name(s)} did not reply before deadline - timed out", "warning")
        req.awaiting = set()
        need = max(0, req.remaining_shortage - req.pending_quantity())
        offers = sorted(req.proposals.items(), key=lambda kv: (kv[1]["travel_time"], -kv[1]["quantity"]))
        req.proposals = {}
        req.phase = "allocating"
        if not offers:
            self._note(req, "No feasible offers this round", "warning")
            return

        # AI decision: the LLM chooses how to split the shortage (validated, with rule-based fallback).
        chosen: dict[str, int] = {}
        if need > 0:
            offer_info = []
            for supplier, offer in offers:
                f = self.env.facilities[supplier]
                offer_info.append({"supplier": supplier, "supplier_name": f.name,
                                   "quantity": offer["quantity"], "travel_time": offer["travel_time"],
                                   "supplier_eligible_stock": f.eligible_stock(req.product),
                                   "supplier_minimum_threshold": f.threshold(req.product)})
            decision = decide_allocation(
                {"request_id": req.request_id, "facility": req.facility,
                 "facility_name": self.sim.facility_name(req.facility), "product": req.product,
                 "need": need, "priority": req.priority},
                offer_info, use_ai=self.config.use_ai_coordinator)
            chosen = {a["supplier"]: a["quantity"] for a in decision["allocations"]}
            self.decision_log.append({"step": self.now, "request_id": req.request_id, "round": req.round,
                                      "need": need, **decision})
            self._note(req, f"[{decision['method']}] " + ", ".join(
                f"{q} from {self.sim.facility_name(s)}" for s, q in chosen.items())
                + f" - {decision['reasoning']}")

        accepted = 0
        for supplier, offer in offers:
            agent = self.sim.inventory_agent_id(supplier)
            q = chosen.get(supplier, 0)
            if q > 0:
                aid = f"{req.request_id}-A{len(req.allocations) + 1}"
                a = Allocation(aid, supplier, q, offer["travel_time"], self.now)
                req.allocations.append(a)
                self._allocations[aid] = (req, a)
                accepted += q
                self.send(P.ACCEPT, agent, O.SUPPLY, req.request_id,
                          f"Reserve exactly {q} {self._unit(req.product)}",
                          topic="accept", request_id=req.request_id, allocation_id=aid,
                          product=req.product, quantity=q)
            else:
                self.send(P.REJECT_PROPOSAL, agent, O.SUPPLY, req.request_id,
                          "Offer not selected", topic="not_selected", request_id=req.request_id)
        self._note(req, f"Evaluated {len(offers)} offer(s); accepted {accepted} {self._unit(req.product)}")

    def _close(self, req: EmergencyRequest, status: str) -> None:
        req.status, req.phase, req.closed_step = status, "closed", self.now
        self._open_index.pop((req.facility, req.product), None)
        self.send(P.INFORM, self.sim.inventory_agent_id(req.facility), O.INVENTORY, req.request_id,
                  f"Request {req.request_id} closed: {status} ({req.received}/{req.total_needed} received)",
                  topic="request_closed", request_id=req.request_id, product=req.product, status=status)
        self._note(req, f"Closed as {status.upper()} ({req.received} of {req.total_needed} received)",
                   "success" if status == "fulfilled" else "warning")

    def _reconcile(self, req: EmergencyRequest) -> None:
        # Reservation timeout: an ACCEPT that got no answer by the deadline.
        for a in req.allocations:
            if a.status == "accepted" and self.now >= a.accepted_step + self.config.cfp_response_deadline:
                a.status, a.note = "refused", "reservation timed out"
                self._note(req, f"Reservation at {self.sim.facility_name(a.supplier)} timed out", "warning")

        if req.phase == "collecting":
            if not req.awaiting or self.now >= req.deadline_step:
                self._evaluate(req)
            else:
                return
        if req.phase == "waiting_retry":
            if self.now >= req.next_round_step:
                self._start_round(req)
            return

        pending = req.pending_quantity()
        if req.remaining_shortage <= 0 and pending == 0:
            self._close(req, "fulfilled")
            return
        if req.remaining_shortage - pending > 0:
            if req.round < self.config.max_cfp_rounds:
                req.phase = "waiting_retry"
                req.next_round_step = self.now + self.config.cfp_retry_delay
                self._note(req, f"{req.remaining_shortage - pending} still uncovered; "
                                f"new CFP round at step {req.next_round_step}")
            elif pending == 0:
                self._close(req, "unresolved")
                return
        req.status = "partial" if req.received > 0 else "open"

    def act(self) -> None:
        for req in list(self.requests.values()):
            if req.active:
                self._reconcile(req)
                if req.active:
                    req.status = "partial" if req.received > 0 else "open"
        active = [r for r in self.requests.values() if r.active]
        if not active:
            self.set_status("idle", "No open emergency requests")
        else:
            parts = []
            for r in active:
                if r.phase == "collecting":
                    got = len(r.proposals) + len(r.refusals)
                    parts.append(f"{r.request_id} collecting offers ({got}/{got + len(r.awaiting)})")
                elif r.phase == "waiting_retry":
                    parts.append(f"{r.request_id} waiting to retry")
                else:
                    parts.append(f"{r.request_id} {r.received}/{r.total_needed} received, "
                                 f"{r.confirmed_inbound} inbound")
            self.set_status("active", "; ".join(parts))

    # ---- view for UI --------------------------------------------------------
    def requests_table(self) -> list[dict]:
        rows = []
        for r in self.requests.values():
            rows.append({
                "request_id": r.request_id, "facility": r.facility, "product": r.product,
                "priority": r.priority, "initial_shortage": r.initial_shortage,
                "remaining_shortage": r.remaining_shortage, "total_needed": r.total_needed,
                "confirmed_inbound": r.confirmed_inbound,
                "received": r.received, "status": r.status, "phase": r.phase, "round": r.round,
                "opened_step": r.opened_step, "closed_step": r.closed_step,
                "allocations": [vars(a).copy() for a in r.allocations],
            })
        return rows
