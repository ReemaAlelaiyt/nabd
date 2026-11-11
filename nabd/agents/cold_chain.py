"""Cold-Chain Monitoring Agent - one shared agent (reactive).

Behaviors (Section 3.3): temperature check, flag and notify, assessment (RELEASE / DISCARD).
"""
from __future__ import annotations

from ..messages import ACLMessage, Ontology, Performative
from .base import BaseAgent

P, O = Performative, Ontology


class ColdChainMonitoringAgent(BaseAgent):
    agent_type = "Cold-Chain Monitoring Agent"
    architecture = "Reactive"

    def __init__(self, agent_id: str, sim):
        super().__init__(agent_id, sim)
        self.excursions_detected = 0
        self.last_alert = ""

    def handle(self, m: ACLMessage) -> None:
        if m.performative == P.REQUEST and m.topic == "assessment":
            self._assess(m.content["batch_id"], m.content["decision"].upper())

    def _assess(self, batch_id: str, decision: str) -> None:
        f = self.env.batch_location(batch_id)
        batch = f.find_batch(batch_id) if f else None
        if batch is None or not batch.flagged:
            self.log(f"Assessment ignored: {batch_id} is not a flagged batch", "warning")
            return
        product = self.env.products[batch.product_id]
        if decision == "RELEASE":
            batch.flagged, batch.assessment = False, "RELEASED"
            text = f"Batch {batch_id} RELEASED: {batch.quantity} {product.unit} eligible again"
        elif decision == "DISCARD":
            f.batches.remove(batch)
            text = f"Batch {batch_id} DISCARDED: {batch.quantity} {product.unit} removed from stock"
        else:
            self.log(f"Unknown assessment decision {decision}", "warning")
            return
        self.send(P.INFORM, self.sim.inventory_agent_id(f.facility_id), O.COLD_CHAIN, batch_id, text,
                  topic="assessment", batch_id=batch_id, decision=decision, product=batch.product_id,
                  quantity=batch.quantity)
        self.log(text, "success" if decision == "RELEASE" else "warning")

    def act(self) -> None:
        env = self.env
        # Storage temperature check
        for f in env.facilities.values():
            for b in list(f.batches):
                reading = env.storage_readings.get(b.batch_id)
                p = env.products[b.product_id]
                if reading is None or b.flagged or p.in_range(reading):
                    continue
                b.flagged, b.assessment = True, "FLAGGED"
                self.excursions_detected += 1
                text = (f"Batch {b.batch_id} at {f.name} flagged: {reading} C outside "
                        f"{p.temp_min:g}..{p.temp_max:g} C ({b.quantity} {p.unit} of {p.name})")
                for receiver in (self.sim.inventory_agent_id(f.facility_id), "Coordination"):
                    self.send(P.INFORM, receiver, O.COLD_CHAIN, b.batch_id, text, topic="batch_flagged",
                              batch_id=b.batch_id, facility=f.facility_id, product=b.product_id,
                              quantity=b.quantity, reading=reading)
                self.last_alert = text
                self.log(text, "alert")
        # Shipment temperature check
        for did, reading in env.shipment_readings.items():
            d = env.deliveries[did]
            p = env.products[d.product_id]
            if d.flagged or p.in_range(reading):
                continue
            d.flagged = True
            self.excursions_detected += 1
            text = (f"Shipment {did} flagged: temperature excursion {reading} C outside "
                    f"{p.temp_min:g}..{p.temp_max:g} C")
            for receiver in ("Transport", "Coordination"):
                self.send(P.INFORM, receiver, O.COLD_CHAIN, d.request_id, text, topic="shipment_flagged",
                          delivery_id=did, allocation_id=d.allocation_id, reading=reading)
            self.last_alert = text
            self.log(text, "alert")

        flagged = sum(1 for f in env.facilities.values() for b in f.batches if b.flagged)
        monitored = len(env.storage_readings) + len(env.shipment_readings)
        if flagged:
            self.set_status("alert", f"{flagged} flagged batch(es) awaiting assessment; monitoring {monitored} readings")
        else:
            self.set_status("active", f"All {monitored} readings within limits")
