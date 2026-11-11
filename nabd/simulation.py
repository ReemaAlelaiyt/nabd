"""Simulation engine - builds the environment and the four agent roles and runs them.

Typical use (e.g. from the interface, Implementation pt2):

    from nabd import Simulation, SimulationConfig
    sim = Simulation(SimulationConfig(seed=7))
    sim.step()                 # one discrete step
    sim.run(10)                # ten steps
    state = sim.snapshot()     # everything the UI needs to draw

Each step:  scheduled events -> environment update (random events, temperature
readings) -> message delivery -> agents act in order
(Cold-Chain, Inventory agents, Coordination, Transport) -> metrics recorded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()  # API keys for the LLM and LangSmith (.env file)

from langsmith import traceable  # noqa: E402

from .agents import (ColdChainMonitoringAgent, EmergencyCoordinationAgent,
                     HospitalInventoryAgent, TransportDeliveryAgent)
from .environment import Environment
from .messages import ACLMessage, MessageBus, Ontology, Performative
from .monitoring import Monitor
from .network import default_network


@dataclass
class SimulationConfig:
    """Adjustable parameters (Section 3.5). All can be bound to UI sliders and changed live."""
    cfp_response_deadline: int = 3   # steps to wait for proposals / reservation replies
    max_cfp_rounds: int = 3          # CFP rounds before a request is closed as unresolved
    cfp_retry_delay: int = 2         # steps between recovery rounds
    retry_cooldown: int = 10         # steps before a facility re-alerts after an unresolved request
    step_minutes: int = 10           # simulated minutes per step (converts travel_time to steps)
    p_demand_spike: float = 0.0      # probability per step of a random demand spike
    p_temp_excursion: float = 0.0    # probability per step of a random temperature excursion
    p_vehicle_breakdown: float = 0.0 # probability per in-transit delivery per step of a breakdown
    seed: Optional[int] = 42
    use_ai_coordinator: bool = True  # Coordination uses the LLM; False = rule-based only


class Simulation:
    def __init__(self, config: Optional[SimulationConfig] = None, network=None):
        self.config = config or SimulationConfig()
        self._network_factory = network or default_network
        self.reset()

    # ---- setup --------------------------------------------------------------
    def reset(self) -> None:
        products, facilities, vehicles, travel = self._network_factory()
        self.step_count = 0
        self.env = Environment(products, facilities, vehicles, travel, self.config, seed=self.config.seed)
        self.bus = MessageBus()
        self.monitor = Monitor(self)
        self._scheduled: list[tuple[int, Callable, str]] = []

        self.cold_chain = ColdChainMonitoringAgent("ColdChain", self)
        self.inventory_agents = {fid: HospitalInventoryAgent(self.inventory_agent_id(fid), self, fid)
                                 for fid in facilities}
        self.coordination = EmergencyCoordinationAgent("Coordination", self)
        self.transport = TransportDeliveryAgent("Transport", self)
        self.agents = [self.cold_chain, *self.inventory_agents.values(), self.coordination, self.transport]
        for a in self.agents:
            self.bus.register(a)
        self.monitor.log_event(0, "System", f"Simulation ready: {len(facilities)} facilities, "
                                            f"{len(vehicles)} vehicles, {len(self.agents)} agents")

    # ---- naming helpers -----------------------------------------------------
    @staticmethod
    def inventory_agent_id(facility_id: str) -> str:
        return f"Inventory-{facility_id}"

    def facility_of_agent(self, agent_id: str) -> str:
        return agent_id.split("-", 1)[1]

    def facility_name(self, facility_id: str) -> str:
        return self.env.facilities[facility_id].name

    # ---- running ------------------------------------------------------------
    def step(self) -> dict:
        """One discrete step, traced in LangSmith with a child run for every agent."""
        self.step_count += 1
        traced = traceable(name=f"Nabd step {self.step_count}", run_type="chain",
                           metadata={"step": self.step_count})(self._run_step)
        return traced(self.step_count)

    def _run_step(self, step: int) -> dict:
        for item in [s for s in self._scheduled if s[0] == self.step_count]:
            self.monitor.log_event(self.step_count, "Scenario", item[2], "warning")
            item[1]()
        self._scheduled = [s for s in self._scheduled if s[0] != self.step_count]
        before = len(self.env.events)
        self.env.update(self.step_count)
        for ev in self.env.events[before:]:
            self.monitor.log_event(self.step_count, "Environment", ev.text, "warning")
        self.bus.deliver(self.step_count)
        for agent in self.agents:
            agent.step()
        return self.monitor.record()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def is_idle(self) -> bool:
        return (self.bus.pending == 0
                and not any(r.active for r in self.coordination.requests.values())
                and not any(d.active for d in self.env.deliveries.values())
                and not any(self._scheduled))

    def run_until_idle(self, max_steps: int = 200) -> int:
        """Run until no messages, requests or deliveries are pending. Returns steps run."""
        start = self.step_count
        self.step()
        while not self.is_idle() and self.step_count - start < max_steps:
            self.step()
        return self.step_count - start

    def schedule(self, step: int, action: Callable[[], None], description: str = "Scheduled event") -> None:
        """Run `action` at the beginning of simulation step `step` (used to script scenarios)."""
        self._scheduled.append((step, action, description))

    # ---- interventions (buttons / scenarios) --------------------------------
    def consume_stock(self, facility_id: str, product_id: str, quantity: int) -> int:
        used = self.env.facilities[facility_id].consume(product_id, quantity)
        self.monitor.log_event(self.step_count, "User", f"{self.facility_name(facility_id)} used {used} "
                                                        f"{self.env.products[product_id].unit} of {product_id}", "warning")
        return used

    def set_threshold(self, facility_id: str, product_id: str, value: int) -> None:
        self.env.facilities[facility_id].minimum_stock_threshold[product_id] = value

    def inject_storage_excursion(self, facility_id: str, product_id: str,
                                 batch_id: Optional[str] = None, reading: Optional[float] = None) -> str:
        f = self.env.facilities[facility_id]
        batch = f.find_batch(batch_id) if batch_id else next(
            (b for b in f.batches if b.product_id == product_id and not b.flagged), None)
        if batch is None:
            raise ValueError("No matching unflagged batch")
        p = self.env.products[batch.product_id]
        value = reading if reading is not None else p.temp_max + 4.0
        self.env.force_excursion(batch.batch_id, value)
        self.monitor.log_event(self.step_count, "User", f"Temperature excursion injected on {batch.batch_id} ({value} C)", "warning")
        return batch.batch_id

    def inject_shipment_excursion(self, delivery_id: Optional[str] = None, reading: Optional[float] = None) -> None:
        """Flag a specific in-transit shipment, or the next shipment that departs if none given."""
        if delivery_id:
            p = self.env.products[self.env.deliveries[delivery_id].product_id]
            self.env.force_excursion(delivery_id, reading if reading is not None else p.temp_max + 4.0)
            target = delivery_id
        else:
            self.env.arm_shipment_excursion(reading if reading is not None else 12.0)
            target = "next shipment in transit"
        self.monitor.log_event(self.step_count, "User", f"Shipment temperature excursion injected on {target}", "warning")

    def set_vehicle_available(self, vehicle_id: str, available: bool) -> None:
        self.env.vehicles[vehicle_id].available = available
        self.monitor.log_event(self.step_count, "User",
                               f"Vehicle {vehicle_id} set {'available' if available else 'unavailable'}", "warning")

    def break_down_vehicle(self, delivery_id: str) -> None:
        """Make the vehicle carrying `delivery_id` break down on the next step (must be in transit)."""
        self.env.force_breakdown(delivery_id)
        self.monitor.log_event(self.step_count, "User", f"Breakdown injected for vehicle carrying {delivery_id}", "warning")

    def set_facility_responsive(self, facility_id: str, responsive: bool) -> None:
        self.env.facilities[facility_id].responsive = responsive
        self.monitor.log_event(self.step_count, "User", f"{self.facility_name(facility_id)} "
                                                        f"{'online' if responsive else 'offline (will not reply)'}", "warning")

    def assess_batch(self, batch_id: str, decision: str) -> None:
        """User assessment of a flagged batch: decision = 'RELEASE' or 'DISCARD' (sent to Cold-Chain)."""
        msg = ACLMessage(Performative.REQUEST, "User", "ColdChain",
                         {"text": f"{decision.upper()} batch {batch_id}", "topic": "assessment",
                          "batch_id": batch_id, "decision": decision.upper()},
                         Ontology.COLD_CHAIN, batch_id)
        self.bus.send(msg, self.step_count)

    # ---- state for the interface --------------------------------------------
    def flagged_batches(self) -> list[dict]:
        return [{"batch_id": b.batch_id, "facility": f.facility_id, "product": b.product_id, "quantity": b.quantity}
                for f in self.env.facilities.values() for b in f.batches if b.flagged]

    def snapshot(self, message_limit: Optional[int] = 50, event_limit: Optional[int] = 50) -> dict:
        env = self.env
        facilities = []
        for f in env.facilities.values():
            rows = []
            for pid in f.products():
                thr = f.threshold(pid)
                rows.append({
                    "product": pid, "product_name": env.products[pid].name, "unit": env.products[pid].unit,
                    "total_stock": f.total_stock(pid), "flagged_stock": f.flagged_stock(pid),
                    "eligible_stock": f.eligible_stock(pid), "minimum_stock_threshold": thr,
                    "reserved_stock": f.reserved(pid), "transferable_stock": f.transferable_stock(pid),
                    "shortage": thr > 0 and f.eligible_stock(pid) < thr,
                })
            facilities.append({"facility_id": f.facility_id, "name": f.name, "kind": f.kind,
                               "responsive": f.responsive, "stock": rows})
        temps = []
        for bid, reading in env.storage_readings.items():
            loc = env.batch_location(bid)
            if loc is None:
                continue
            b = loc.find_batch(bid)
            p = env.products[b.product_id]
            temps.append({"target": bid, "kind": "storage", "facility": loc.facility_id, "product": p.product_id,
                          "reading": reading, "min": p.temp_min, "max": p.temp_max,
                          "in_range": p.in_range(reading), "flagged": b.flagged})
        for did, reading in env.shipment_readings.items():
            d = env.deliveries[did]
            p = env.products[d.product_id]
            temps.append({"target": did, "kind": "shipment", "facility": None, "product": p.product_id,
                          "reading": reading, "min": p.temp_min, "max": p.temp_max,
                          "in_range": p.in_range(reading), "flagged": d.flagged})
        msgs = self.bus.log if message_limit is None else self.bus.log[-message_limit:]
        return {
            "step": self.step_count,
            "simulated_minutes": self.step_count * self.config.step_minutes,
            "config": asdict(self.config),
            "agents": [a.to_dict() for a in self.agents],
            "facilities": facilities,
            "vehicles": [{"vehicle_id": v.vehicle_id, "name": v.name, "status": v.status, "location": v.location,
                          "capacity": v.capacity, "temp_range": f"{v.temp_min:g}..{v.temp_max:g} C"}
                         for v in env.vehicles.values()],
            "requests": self.coordination.requests_table(),
            "deliveries": [{k: v for k, v in vars(d).items() if k != "cargo"} for d in env.deliveries.values()],
            "temperatures": temps,
            "flagged_batches": self.flagged_batches(),
            "messages": [m.to_dict() for m in msgs],
            "events": self.monitor.events_as_dicts(event_limit),
            "metrics": self.monitor.metrics(),
            "ai_decisions": self.coordination.decision_log,
        }
