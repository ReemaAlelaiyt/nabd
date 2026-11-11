"""Transport and Delivery Agent - one shared agent (hybrid).

Deliberative part: chooses the feasible vehicle with the shortest total travel time.
Reactive part: reacts to dispatch failures, temperature alerts and breakdowns.
Behaviors (Section 3.3): vehicle selection, assignment response, delivery update.
"""
from __future__ import annotations

import math

from ..messages import ACLMessage, Ontology, Performative
from ..models import Delivery
from .base import BaseAgent

P, O = Performative, Ontology


class TransportDeliveryAgent(BaseAgent):
    agent_type = "Transport and Delivery Agent"
    architecture = "Hybrid"

    # ---- vehicle selection ---------------------------------------------------
    def select_vehicle(self, supplier: str, receiver: str, product_id: str, quantity: int):
        """Return (vehicle, eta_minutes) or (None, reason)."""
        product = self.env.products[product_id]
        route = self.env.travel_time(supplier, receiver)
        best, reasons = None, []
        for v in self.env.vehicles.values():
            if not v.available:
                reasons.append(f"{v.vehicle_id} unavailable")
                continue
            if v.busy:
                reasons.append(f"{v.vehicle_id} busy")
                continue
            if v.capacity < quantity:
                reasons.append(f"{v.vehicle_id} capacity {v.capacity} < {quantity}")
                continue
            if not v.compatible_with(product):
                reasons.append(f"{v.vehicle_id} not temperature-compatible")
                continue
            eta = self.env.travel_time(v.location, supplier) + route
            if best is None or eta < best[1]:
                best = (v, eta)
        if best:
            return best
        return None, "; ".join(reasons) or "no vehicles"

    # ---- message handling ---------------------------------------------------
    def handle(self, m: ACLMessage) -> None:
        if m.performative == P.REQUEST and m.topic == "deliver":
            self._on_deliver_request(m)
        elif m.performative == P.INFORM and m.topic == "dispatched":
            self._on_dispatched(m)
        elif m.performative == P.REFUSE and m.topic == "dispatch_failed":
            self._on_dispatch_failed(m)
        elif m.performative == P.INFORM and m.topic == "shipment_flagged":
            self.log(f"Shipment {m.content['delivery_id']} flagged by Cold-Chain; will arrive as flagged", "alert")

    def _on_deliver_request(self, m: ACLMessage) -> None:
        c = m.content
        vehicle, eta = self.select_vehicle(c["supplier"], c["receiver"], c["product"], c["quantity"])
        if vehicle is None:
            self.send(P.REFUSE, m.sender, O.DELIVERY, m.conversation_id,
                      f"No compatible available vehicle ({eta})",
                      topic="no_vehicle", allocation_id=c["allocation_id"], reason=f"no compatible available vehicle")
            self.log(f"Cannot serve {c['allocation_id']}: {eta}", "warning")
            return
        did = self.env.new_delivery_id()
        d = Delivery(did, c["request_id"], c["allocation_id"], c["product"], c["quantity"],
                     c["supplier"], c["receiver"], vehicle.vehicle_id, eta, self.now)
        self.env.deliveries[did] = d
        vehicle.busy = True
        self.send(P.INFORM, m.sender, O.DELIVERY, m.conversation_id,
                  f"Vehicle {vehicle.vehicle_id} assigned ({did}); ETA {eta} minutes",
                  topic="vehicle_assigned", allocation_id=c["allocation_id"], delivery_id=did,
                  vehicle_id=vehicle.vehicle_id, eta_minutes=eta)
        self.send(P.REQUEST, self.sim.inventory_agent_id(c["supplier"]), O.DELIVERY, m.conversation_id,
                  f"Prepare {c['quantity']} {self.env.products[c['product']].unit} for pickup by {vehicle.vehicle_id} ({did})",
                  topic="dispatch", allocation_id=c["allocation_id"], delivery_id=did)
        self.log(f"Assigned {vehicle.name} to {did} ({c['quantity']} units, ETA {eta} min)", "success")

    def _on_dispatched(self, m: ACLMessage) -> None:
        d = self.env.deliveries[m.content["delivery_id"]]
        d.status, d.started_step = "in_progress", self.now
        d.steps_remaining = max(1, math.ceil(d.eta_minutes / self.config.step_minutes))
        self.env.vehicles[d.vehicle_id].location = None
        self.send(P.INFORM, "Coordination", O.DELIVERY, d.request_id,
                  f"{d.delivery_id} in transit; arrives in {d.steps_remaining} step(s)",
                  topic="delivery_started", allocation_id=d.allocation_id, delivery_id=d.delivery_id)

    def _on_dispatch_failed(self, m: ACLMessage) -> None:
        d = self.env.deliveries[m.content["delivery_id"]]
        self._finish(d, "failed", reason=m.content.get("reason", "dispatch failed"), location=d.supplier)

    # ---- periodic behavior: delivery update ---------------------------------
    def act(self) -> None:
        for d in list(self.env.deliveries.values()):
            if d.status != "in_progress" or d.started_step == self.now:
                continue
            if d.delivery_id in self.env.breakdowns:
                self._finish(d, "failed", reason="vehicle breakdown in transit", location=d.supplier, breakdown=True)
                continue
            d.steps_remaining -= 1
            if d.steps_remaining <= 0:
                self._arrive(d)
        active = [d for d in self.env.deliveries.values() if d.active]
        busy = sum(1 for v in self.env.vehicles.values() if v.busy)
        if active:
            self.set_status("active", f"{len(active)} active deliveries; {busy}/{len(self.env.vehicles)} vehicles busy")
        else:
            self.set_status("idle", f"No active deliveries; {busy}/{len(self.env.vehicles)} vehicles busy")

    def _arrive(self, d: Delivery) -> None:
        status = "flagged" if d.flagged else "completed"
        d.status, d.finished_step = status, self.now
        v = self.env.vehicles[d.vehicle_id]
        v.busy, v.location = False, d.receiver
        unit = self.env.products[d.product_id].unit
        self.send(P.INFORM, self.sim.inventory_agent_id(d.receiver), O.DELIVERY, d.request_id,
                  f"{d.delivery_id} arrived: {d.quantity} {unit}{' (FLAGGED)' if d.flagged else ''}",
                  topic="receipt", delivery_id=d.delivery_id, product=d.product_id, quantity=d.quantity,
                  flagged=d.flagged, batch_id=f"{d.delivery_id}-B", supplier=d.supplier)
        topic = "delivery_flagged" if d.flagged else "delivery_completed"
        self.send(P.INFORM, "Coordination", O.DELIVERY, d.request_id,
                  f"{d.delivery_id} {'arrived with temperature flag' if d.flagged else 'completed'}: "
                  f"{d.quantity} {unit} to {self.sim.facility_name(d.receiver)}",
                  topic=topic, allocation_id=d.allocation_id, delivery_id=d.delivery_id, quantity=d.quantity)
        self.log(f"{d.delivery_id} {status} at {self.sim.facility_name(d.receiver)}", "alert" if d.flagged else "success")

    def _finish(self, d: Delivery, status: str, reason: str, location: str, breakdown: bool = False) -> None:
        d.status, d.finished_step, d.failure_reason = status, self.now, reason
        v = self.env.vehicles[d.vehicle_id]
        v.busy = False
        v.location = location
        if breakdown:
            v.available = False
            # Cargo is returned to the supplier and must be assessed (cold chain broken).
            self.send(P.INFORM, self.sim.inventory_agent_id(d.supplier), O.DELIVERY, d.request_id,
                      f"{d.delivery_id} cargo returned after breakdown ({d.quantity} units) - flagged",
                      topic="return_cargo", product=d.product_id, quantity=d.quantity,
                      batch_id=f"{d.delivery_id}-R")
        self.send(P.INFORM, "Coordination", O.DELIVERY, d.request_id,
                  f"{d.delivery_id} failed: {reason}",
                  topic="delivery_failed", allocation_id=d.allocation_id, delivery_id=d.delivery_id, reason=reason)
        self.log(f"{d.delivery_id} failed: {reason}", "warning")
