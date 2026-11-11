# Nabd (نبض) – Multi-Agent Emergency Medical Supply Redistribution

Nabd is a Python multi-agent simulation. Hospitals, a medical warehouse and refrigerated
vehicles work together to resolve vaccine and medicine shortages, and a cold-chain agent
monitors temperatures along the way. All data is synthetic. The system does not connect to real
hospital systems or physical sensors.

## Setup (AI model + LangSmith)

1. Get a free LLM key at <https://console.groq.com/keys> (any OpenAI-compatible API works).
2. Get a LangSmith key at <https://smith.langchain.com> (Settings → API Keys).
3. Copy `.env.example` to `.env` and paste both keys in. `.env` is in `.gitignore`, so never upload it.

Without a key the system still runs, and Coordination uses the rule-based choice instead.

## Run

```bash
pip install -r requirements.txt
python run_demo.py              # baseline: Hospital A influenza vaccine shortage, run until idle
python run_demo.py --messages   # also print every ACL message
python run_demo.py --random     # turn on random demand spikes, temperature excursions, breakdowns
python run_demo.py --export out # save out/messages.csv and out/state.json
python run_demo.py --no-ai      # Coordination without the LLM (rule-based only)
python -m pytest -q             # 19 tests: rules, normal flow, failures, AI decision, LangSmith tracing
```

## Project structure

```
nabd/
  messages.py          ACL-like messages (performative, sender, receiver, content, ontology,
                       conversation ID) and the MessageBus (messages arrive the next step)
  models.py            Product, Batch, Facility, Vehicle, Delivery and the simulation rules
  environment.py       Simulated environment: stock, routes, vehicles, temperature readings,
                       random events (stochastic, dynamic, discrete)
  network.py           Default network: Hospitals A–D, Central Warehouse W, 3 products, 3 vehicles
  agents/
    base.py            Shared agent cycle: read inbox -> handle messages -> act
    inventory.py       Hospital Inventory Agent        (one per facility, reactive)
    cold_chain.py      Cold-Chain Monitoring Agent     (shared, reactive)
    coordination.py    Emergency Coordination Agent    (shared, deliberative, AI / LLM-based)
    transport.py       Transport and Delivery Agent    (shared, hybrid)
  llm_decider.py       AI brain of the Coordination Agent: LLM prompt, rule validation, fallback,
                       LangSmith tracing
  monitoring.py        Event log, message log, metrics per step, CSV/JSON export
  simulation.py        Simulation engine, adjustable parameters, interventions, snapshot()
run_demo.py            Console monitor / demo
tests/test_nabd.py     Automated tests
```

## AI agent and LangSmith

**AI-based agent: the Emergency Coordination Agent.** When the offers for a request are in
(or the CFP deadline passes), Coordination sends the shortage and every supplier's offer to an LLM.
That includes quantity, travel time, and the supplier's own eligible stock and minimum threshold.
The LLM returns how many units to take from each supplier, with its reasoning. The answer is checked
against the hard rules: no more than offered, no more than needed, and only real suppliers.
If the answer breaks a rule or the AI is unavailable, the agent falls back to "quickest supplier
first". The other three agents stay rule-based (reactive / hybrid).
Decisions are listed in `sim.coordination.decision_log` and in `snapshot()["ai_decisions"]`.

**LangSmith tracing.** Every simulation step is one trace, `Nabd step N`. Each agent gets a child
run showing the messages it received, the messages it sent and its status. The AI decision, with
the prompt and the model's answer, is nested under the `Coordination` run.
Open <https://smith.langchain.com>, then the project `nabd`.

## How it matches the design (Sections 3.1–3.6)

| Design item | Where in the code |
|---|---|
| eligible_stock = total − flagged | `Facility.eligible_stock` |
| transferable_stock = max(0, eligible − minimum − reserved) | `Facility.transferable_stock` |
| Priority: High < 25%, Medium 25–50%, Low > 50% of the minimum | `compute_priority` |
| One open request per facility/product | `EmergencyCoordinationAgent._open_index` |
| CFP deadline of 3 steps (adjustable); late replies time out | `cfp_response_deadline`, `_evaluate`, `_on_propose` |
| Offers rechecked when accepted | `HospitalInventoryAgent._on_accept` |
| LLM chooses the allocation (validated; quickest-first fallback); ACCEPT exact quantities, REJECT-PROPOSAL for unused offers | `EmergencyCoordinationAgent._evaluate`, `llm_decider.decide_allocation` |
| Transport requested only after the reservation is confirmed | `_on_reserved` |
| Vehicle check: availability, capacity, temperature compatibility, shortest travel time | `TransportDeliveryAgent.select_vehicle` |
| Flagged shipments are not counted as fulfilled; RELEASE / DISCARD assessment | `_on_delivery_outcome`, `ColdChainMonitoringAgent._assess` |
| Recovery: retry other suppliers, release cancelled reservations, partial/unresolved status | `_reconcile`, `_on_transport_refused` |
| Performatives INFORM, CFP, PROPOSE, REFUSE, ACCEPT, REJECT-PROPOSAL, REQUEST | `Performative` |
| Ontologies Inventory, SupplyNegotiation, Delivery, ColdChain | `Ontology` |

**Order within each step:** scheduled events → environment update (random events, temperature
readings) → messages delivered → Cold-Chain → Inventory agents → Coordination → Transport →
metrics recorded.

**Baseline result (default network):** Hospital A holds 20 of a 100-dose minimum, so it reports an
80-dose High-priority shortage and REQ-001 opens. Hospital B offers 50 doses (20 min), C offers 40
(35 min), D refuses, and W offers 80 (60 min) but is rejected. B reserves 50 and C reserves 30.
V2 and V1 are assigned to the two deliveries, and REQ-001 is **fulfilled at step 16** after
30 messages.

## For the interface (Implementation pt2)

```python
from nabd import Simulation, SimulationConfig

sim = Simulation(SimulationConfig(seed=42))
sim.step()               # "Step" button
sim.run(10)              # "Run N steps" button
sim.run_until_idle()     # "Run until done" button
sim.reset()              # "Reset" button
state = sim.snapshot()   # everything to display, as a plain dict
```

`snapshot()` keys: `step`, `agents` (live status of every agent), `facilities` (stock per product),
`vehicles`, `requests` (with allocations), `deliveries`, `temperatures`, `flagged_batches`,
`messages` (message log), `events` (readable log), `metrics`, `config`.
Use `sim.monitor.history` for charts over time and `sim.monitor.subscribe(fn)` for live log
updates.

**Parameters for sliders.** Change these on `sim.config`; they take effect on the next step.
`cfp_response_deadline`, `max_cfp_rounds`, `cfp_retry_delay`, `retry_cooldown`, `step_minutes`,
`p_demand_spike`, `p_temp_excursion`, `p_vehicle_breakdown`, `use_ai_coordinator` (AI on/off),
`seed` (seed is used on reset).

**Interventions for scenarios and buttons:**

| Method | Effect |
|---|---|
| `sim.consume_stock("D", "FLU", 60)` | Uses up stock, which can create a new shortage or a competing request |
| `sim.inject_storage_excursion("B", "FLU")` | Out-of-range temperature on a storage batch; Cold-Chain flags it |
| `sim.inject_shipment_excursion()` | Next shipment in transit is flagged on arrival |
| `sim.set_vehicle_available("V1", False)` | Vehicle out of service, so Transport may refuse |
| `sim.break_down_vehicle("SH-01")` | Vehicle breaks down in transit; delivery fails and the cargo is returned flagged |
| `sim.set_facility_responsive("B", False)` | Facility stops replying, so its CFP and reservation time out |
| `sim.assess_batch("SH-01-B", "RELEASE")` | User assessment (RELEASE or DISCARD) of a flagged batch |
| `sim.set_threshold("A", "INS", 90)` | Changes a minimum stock threshold |
| `sim.schedule(step, fn, "description")` | Runs an intervention at a given step (for scripted scenarios) |

Failure and conflict paths already checked by the tests:

- a flagged shipment that gets replaced
- no vehicle available (unresolved after 3 rounds)
- an unresponsive supplier that times out
- two hospitals competing for the same supplier (reservation refused, then retry)
- supplier stock flagged after reservation (dispatch fails, then recovery)
- a breakdown in transit
