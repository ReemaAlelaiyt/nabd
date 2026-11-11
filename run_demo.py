"""Console demo / monitor for Nabd (Implementation pt1).

    python run_demo.py                 # baseline: Hospital A influenza vaccine shortage
    python run_demo.py --messages      # also print every ACL message
    python run_demo.py --random        # turn on random events (stochastic run)
    python run_demo.py --export out    # save message log CSV + full JSON state
    python run_demo.py --no-ai         # Coordination without the LLM (rule-based only)
"""
from __future__ import annotations

import argparse
import os

from nabd import Simulation, SimulationConfig


def print_stock(sim: Simulation) -> None:
    print("\nFacility stock (eligible / minimum, reserved, flagged)")
    for f in sim.snapshot()["facilities"]:
        cells = [f"{s['product']}: {s['eligible_stock']}/{s['minimum_stock_threshold']}"
                 f"{' r' + str(s['reserved_stock']) if s['reserved_stock'] else ''}"
                 f"{' f' + str(s['flagged_stock']) if s['flagged_stock'] else ''}"
                 f"{' !' if s['shortage'] else ''}" for s in f["stock"]]
        print(f"  {f['name']:<27} " + "   ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="fixed number of steps (default: run until idle)")
    ap.add_argument("--messages", action="store_true", help="print ACL messages as they are sent")
    ap.add_argument("--random", action="store_true", help="enable random demand/temperature/breakdown events")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ai", action="store_true", help="Coordination uses the rule-based choice only")
    ap.add_argument("--export", metavar="DIR", help="write messages.csv and state.json to DIR")
    args = ap.parse_args()

    cfg = SimulationConfig(seed=args.seed, use_ai_coordinator=not args.no_ai)
    if args.random:
        cfg.p_demand_spike, cfg.p_temp_excursion, cfg.p_vehicle_breakdown = 0.05, 0.05, 0.02
    sim = Simulation(cfg)
    sim.monitor.subscribe(print)
    print_stock(sim)
    print()

    seen = 0
    steps = args.steps or 60
    for _ in range(steps):
        sim.step()
        if args.messages:
            for m in sim.bus.log[seen:]:
                print("      msg", m)
            seen = len(sim.bus.log)
        if not args.steps and sim.is_idle():
            break

    print_stock(sim)
    print("\nAgent status")
    for a in sim.snapshot()["agents"]:
        print(f"  {a['agent_id']:<14} [{a['architecture']:<17}] {a['state']:<7} {a['status']}")
    print("\nRequests")
    for r in sim.coordination.requests_table():
        print(f"  {r['request_id']} {r['facility']}/{r['product']} {r['priority']:<6} "
              f"{r['received']}/{r['total_needed']} received -> {r['status']}")
    print("\nCoordination decisions (AI)")
    for d in sim.coordination.decision_log:
        chosen = ", ".join(f"{a['quantity']} from {sim.facility_name(a['supplier'])}" for a in d["allocations"])
        print(f"  step {d['step']} {d['request_id']} [{d['method']}] {chosen}\n      reason: {d['reasoning']}")
    m = sim.monitor.metrics()
    print(f"\nSteps: {sim.step_count}   Messages: {m['messages_total']} {m['messages_by_performative']}")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        sim.monitor.export_messages_csv(os.path.join(args.export, "messages.csv"))
        sim.monitor.export_json(os.path.join(args.export, "state.json"))
        print(f"Exported to {args.export}/")


if __name__ == "__main__":
    main()
