"""Default synthetic network: 4 hospitals, 1 central warehouse, 3 products, 3 vehicles.

The numbers reproduce the example used in the report (Sections 2.2 and 3.3):
Hospital A holds 20 influenza vaccine doses against a minimum of 100 (80-dose
shortage, High priority); Hospital B can offer 50 doses (20-minute route),
Hospital C can offer 40 (35-minute route), Hospital D has no transferable stock.
"""
from __future__ import annotations

import copy

from .models import Batch, Facility, Product, Vehicle

PRODUCTS = {
    "FLU": Product("FLU", "Influenza Vaccine", "doses", 2.0, 8.0),
    "INS": Product("INS", "Insulin", "vials", 2.0, 8.0),
    "MRNA": Product("MRNA", "mRNA Vaccine", "doses", -25.0, -15.0),
}


def _facility(fid, name, kind, stock: dict[str, list[int]], thresholds: dict[str, int]) -> Facility:
    batches = []
    for pid, quantities in stock.items():
        for i, q in enumerate(quantities, start=1):
            batches.append(Batch(f"{fid}-{pid}-{i:02d}", pid, q, origin=fid))
    return Facility(fid, name, kind, batches, dict(thresholds), {p: 0 for p in thresholds})


FACILITIES = {
    "A": _facility("A", "Hospital A", "hospital",
                   {"FLU": [20], "INS": [50, 30], "MRNA": [60]}, {"FLU": 100, "INS": 50, "MRNA": 40}),
    "B": _facility("B", "Hospital B", "hospital",
                   {"FLU": [100, 50], "INS": [70], "MRNA": [50]}, {"FLU": 100, "INS": 50, "MRNA": 40}),
    "C": _facility("C", "Hospital C", "hospital",
                   {"FLU": [90, 50], "INS": [60], "MRNA": [30]}, {"FLU": 100, "INS": 50, "MRNA": 30}),
    "D": _facility("D", "Hospital D", "hospital",
                   {"FLU": [80], "INS": [55], "MRNA": [45]}, {"FLU": 80, "INS": 50, "MRNA": 40}),
    "W": _facility("W", "Central Medical Warehouse", "warehouse",
                   {"FLU": [250, 150], "INS": [300], "MRNA": [200]}, {"FLU": 200, "INS": 100, "MRNA": 80}),
}

VEHICLES = {
    "V1": Vehicle("V1", "Reefer Truck 1", 120, 2.0, 8.0, "W"),
    "V2": Vehicle("V2", "Reefer Van 2", 60, 2.0, 8.0, "B"),
    "V3": Vehicle("V3", "Freezer Truck 3", 80, -25.0, -15.0, "W"),
}

# travel_time in minutes (symmetric)
TRAVEL_TIMES = {
    ("A", "B"): 20, ("A", "C"): 35, ("A", "D"): 25, ("A", "W"): 60,
    ("B", "C"): 30, ("B", "D"): 40, ("B", "W"): 45,
    ("C", "D"): 30, ("C", "W"): 25,
    ("D", "W"): 50,
}


def default_network():
    """Return fresh copies (products, facilities, vehicles, travel_times)."""
    return (copy.deepcopy(PRODUCTS), copy.deepcopy(FACILITIES),
            copy.deepcopy(VEHICLES), dict(TRAVEL_TIMES))
