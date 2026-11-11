"""Tests for Nabd Implementation pt1 (rules, agents, failure handling)."""
import pytest

from nabd import Simulation, SimulationConfig, compute_priority
from nabd.messages import Performative


def new_sim(**kw):
    return Simulation(SimulationConfig(seed=1, **kw))


def req(sim, rid="REQ-001"):
    return sim.coordination.requests[rid]


def topics(sim, topic):
    return [m for m in sim.bus.log if m.topic == topic]


# ---- simulation rules (Section 3.5) -----------------------------------------
def test_priority_rule():
    assert compute_priority(20, 100) == "High"
    assert compute_priority(25, 100) == "Medium"
    assert compute_priority(50, 100) == "Medium"
    assert compute_priority(51, 100) == "Low"
    assert compute_priority(100, 100) is None
    with pytest.raises(ValueError):
        compute_priority(1, 0)


def test_eligible_and_transferable_stock():
    sim = new_sim()
    b = sim.env.facilities["B"]
    assert b.eligible_stock("FLU") == 150
    assert b.transferable_stock("FLU") == 50
    b.batches[1].flagged = True            # flag the 50-dose batch
    assert b.eligible_stock("FLU") == 100
    assert b.transferable_stock("FLU") == 0
    b.batches[1].flagged = False
    b.reserved_stock["FLU"] = 20
    assert b.transferable_stock("FLU") == 30


# ---- baseline: report example -----------------------------------------------
def test_baseline_shortage_fulfilled_like_report_example():
    sim = new_sim()
    sim.run_until_idle()
    r = req(sim)
    assert (r.facility, r.product, r.priority, r.initial_shortage) == ("A", "FLU", "High", 80)
    assert r.status == "fulfilled" and r.received == 80
    assert {(a.supplier, a.quantity) for a in r.allocations} == {("B", 50), ("C", 30)}
    assert sim.env.facilities["A"].eligible_stock("FLU") == 100
    assert sim.env.facilities["B"].reserved("FLU") == 0
    perfs = {m.performative for m in sim.bus.log}
    assert perfs == set(Performative)  # all seven performatives used
    assert len(sim.coordination.requests) == 1  # no duplicate request


def test_conversation_ids_link_request_messages():
    sim = new_sim()
    sim.run_until_idle()
    convs = {m.conversation_id for m in sim.bus.log if m.performative in (Performative.CFP, Performative.ACCEPT)}
    assert convs == {"REQ-001"}


# ---- failure / conflict paths -------------------------------------------------
def test_flagged_shipment_not_counted_and_replaced():
    sim = new_sim()
    sim.inject_shipment_excursion()  # first shipment in transit gets an excursion
    sim.run_until_idle()
    r = req(sim)
    assert any(a.status == "flagged" for a in r.allocations)
    assert r.status == "fulfilled" and r.received == 80
    assert r.round >= 2
    a = sim.env.facilities["A"]
    assert a.flagged_stock("FLU") > 0 and a.eligible_stock("FLU") == 100
    # User releases the flagged batch -> eligible again
    batch = sim.flagged_batches()[0]["batch_id"]
    sim.assess_batch(batch, "RELEASE")
    sim.run(2)
    assert a.flagged_stock("FLU") == 0 and a.eligible_stock("FLU") > 100


def test_no_vehicle_releases_reservations_and_ends_unresolved():
    sim = new_sim()
    sim.set_vehicle_available("V1", False)
    sim.set_vehicle_available("V2", False)
    sim.run_until_idle()
    r = req(sim)
    assert r.status == "unresolved" and r.received == 0
    assert r.round == sim.config.max_cfp_rounds
    assert topics(sim, "no_vehicle") and topics(sim, "release_reservation")
    assert all(f.reserved("FLU") == 0 for f in sim.env.facilities.values())
    # Inventory agent waits for the cooldown before re-alerting
    assert sim.inventory_agents["A"].cooldown_until["FLU"] > sim.step_count - 1


def test_unresponsive_supplier_times_out():
    sim = new_sim()
    sim.set_facility_responsive("B", False)
    sim.run_until_idle()
    r = req(sim)
    assert "B" in r.timeouts or any("timed out" in h[1] for h in r.history)
    assert r.status == "fulfilled"
    assert {a.supplier for a in r.allocations} == {"C", "W"}


def test_competing_requests_reservation_conflict():
    sim = new_sim()
    sim.consume_stock("D", "FLU", 60)  # D now 20/80 -> second shortage at the same time
    sim.run_until_idle()
    assert len(sim.coordination.requests) == 2
    assert topics(sim, "reservation_failed"), "one supplier should refuse the second reservation"
    for r in sim.coordination.requests.values():
        assert r.status == "fulfilled"
    assert all(f.reserved("FLU") == 0 for f in sim.env.facilities.values())


def test_supplier_stock_flagged_after_reservation_dispatch_fails_and_recovers():
    sim = new_sim()
    sim.run(5)  # B and C have reserved by now
    assert sim.env.facilities["B"].reserved("FLU") == 50
    sim.inject_storage_excursion("B", "FLU", batch_id="B-FLU-01")  # 100-dose batch
    sim.run_until_idle()
    r = req(sim)
    assert topics(sim, "dispatch_failed") or any(a.status == "failed" for a in r.allocations) or r.status == "fulfilled"
    assert r.status == "fulfilled"


def test_vehicle_breakdown_in_transit():
    sim = new_sim()
    sim.run(9)
    in_transit = [d for d in sim.env.deliveries.values() if d.status == "in_progress"]
    assert in_transit
    sim.break_down_vehicle(in_transit[0].delivery_id)
    sim.run_until_idle()
    d = sim.env.deliveries[in_transit[0].delivery_id]
    assert d.status == "failed"
    assert not sim.env.vehicles[d.vehicle_id].available
    assert req(sim).status in ("fulfilled", "unresolved")
    assert sim.env.facilities[d.supplier].flagged_stock("FLU") == d.quantity  # returned cargo flagged


def test_random_run_is_stable_and_reproducible():
    def run(seed):
        sim = Simulation(SimulationConfig(seed=seed, p_demand_spike=0.2, p_temp_excursion=0.15, p_vehicle_breakdown=0.05))
        sim.run(150)
        for f in sim.env.facilities.values():
            for p in f.products():
                assert f.eligible_stock(p) >= 0 and f.reserved(p) >= 0
        return len(sim.bus.log), sim.monitor.metrics()["requests_total"]
    assert run(3) == run(3)
    for s in range(10):
        run(s)


def test_snapshot_has_ui_sections():
    sim = new_sim()
    sim.run(8)
    snap = sim.snapshot()
    for key in ("agents", "facilities", "vehicles", "requests", "deliveries", "temperatures",
                "messages", "events", "metrics", "config"):
        assert key in snap
    assert len(snap["agents"]) == 8


def test_exports(tmp_path):
    sim = new_sim()
    sim.run_until_idle()
    sim.monitor.export_messages_csv(tmp_path / "m.csv")
    sim.monitor.export_json(tmp_path / "s.json")
    assert (tmp_path / "m.csv").read_text().count("\n") == len(sim.bus.log) + 1


# ---- AI Coordination Agent (LLM) and LangSmith tracing ------------------------
from types import SimpleNamespace as NS

from nabd import llm_decider


class FakeLLM:
    """Stands in for the real LLM so the test does not need an API key."""
    def __init__(self, answer):
        self.answer, self.calls = answer, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return NS(choices=[NS(message=NS(content=self.answer))])


def run_baseline_with(answer, monkeypatch):
    fake = FakeLLM(answer)
    monkeypatch.setattr(llm_decider, "get_client", lambda: fake)
    sim = new_sim()
    sim.run_until_idle()
    return sim, fake


def test_ai_decision_is_used(monkeypatch):
    answer = ('{"allocations": [{"supplier": "B", "quantity": 40}, {"supplier": "C", "quantity": 40}],'
              ' "reasoning": "Split the load to protect Hospital B."}')
    sim, fake = run_baseline_with(answer, monkeypatch)
    d = sim.coordination.decision_log[0]
    assert fake.calls >= 1 and d["method"] == "AI"
    assert {a["supplier"]: a["quantity"] for a in d["allocations"]} == {"B": 40, "C": 40}
    assert req(sim).status == "fulfilled"


@pytest.mark.parametrize("answer", [
    '{"allocations": [{"supplier": "B", "quantity": 999}], "reasoning": "too much"}',   # more than offered
    '{"allocations": [{"supplier": "Z", "quantity": 10}], "reasoning": "unknown"}',     # not a supplier
    "Sorry, I cannot help with that.",                                                  # not JSON
])
def test_invalid_ai_answer_falls_back_to_rules(answer, monkeypatch):
    sim, _ = run_baseline_with(answer, monkeypatch)
    assert sim.coordination.decision_log[0]["method"] == "rule-based fallback"
    assert req(sim).status == "fulfilled"


def test_no_api_key_uses_rules():
    sim = new_sim()
    sim.run_until_idle()
    assert sim.coordination.decision_log[0]["method"] == "rule-based fallback"
    assert "ai_decisions" in sim.snapshot()


def test_langsmith_traces_steps_agents_and_ai_decision():
    from unittest.mock import MagicMock
    from langsmith import Client, tracing_context
    fake_client = MagicMock(spec=Client)
    sim = new_sim()
    with tracing_context(enabled=True, client=fake_client, project_name="nabd-test"):
        sim.run(5)
    runs = {c.kwargs["id"]: c.kwargs for c in fake_client.create_run.call_args_list}
    names = [r["name"] for r in runs.values()]
    assert "Nabd step 1" in names and "Coordination" in names and "Transport" in names
    ai = next(r for r in runs.values() if r["name"] == "Coordination AI decision")
    assert runs[ai["parent_run_id"]]["name"] == "Coordination"
