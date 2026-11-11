"""AI decision-making for the Emergency Coordination Agent.

The Coordination Agent asks a Large Language Model (LLM) how to split a shortage
between the suppliers that sent PROPOSE messages. The answer is checked against the
simulation rules. If the AI is unavailable, or its answer breaks a rule, the agent
falls back to the rule-based choice (quickest supplier first, Section 3.3).

Every AI decision is traced in LangSmith (inputs, prompt, model answer, final choice).

Settings come from environment variables (put them in a .env file):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL            -> any OpenAI-compatible API (e.g. Groq)
    LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()   # reads the .env file if it exists

DEFAULT_MODEL = "llama-3.3-70b-versatile"
_client = None


def get_client():
    """Create the LLM client once (wrapped so LangSmith records every call). None if no key."""
    global _client
    if _client is None and os.getenv("LLM_API_KEY"):
        _client = wrap_openai(OpenAI(api_key=os.getenv("LLM_API_KEY"),
                                     base_url=os.getenv("LLM_BASE_URL") or None))
    return _client


SYSTEM_PROMPT = """You are the Emergency Coordination Agent in Nabd, a simulated medical supply network.
A facility has a shortage. Supplier facilities sent offers. Decide how many units to accept from each supplier.

Hard rules (never break them):
- Never accept more from a supplier than it offered.
- The total accepted must not exceed the quantity needed.
- Only use suppliers from the offers list.

Goals, in order:
1. Cover as much of the shortage as possible.
2. For High priority, deliver quickly: prefer a shorter travel_time.
3. Protect suppliers: avoid taking a supplier's whole surplus when another supplier can share the load at a similar travel time.
4. Prefer fewer shipments when travel times are similar.

Answer with JSON only, in this exact format:
{"allocations": [{"supplier": "<supplier id>", "quantity": <integer>}], "reasoning": "<one or two sentences>"}"""


def rule_based_allocation(need: int, offers: list[dict]) -> list[dict]:
    """Fallback: quickest supplier first, then the larger offer (design Section 3.3)."""
    allocations = []
    for offer in sorted(offers, key=lambda o: (o["travel_time"], -o["quantity"])):
        if need <= 0:
            break
        take = min(offer["quantity"], need)
        allocations.append({"supplier": offer["supplier"], "quantity": take})
        need -= take
    return allocations


def validate(allocations: list[dict], need: int, offers: list[dict]) -> list[dict]:
    """Check the AI answer against the hard rules. Returns a clean list or raises ValueError."""
    offered = {o["supplier"]: o["quantity"] for o in offers}
    clean, total = [], 0
    for item in allocations:
        supplier, quantity = str(item["supplier"]), int(item["quantity"])
        if supplier not in offered:
            raise ValueError(f"unknown supplier {supplier}")
        if quantity < 0 or quantity > offered[supplier]:
            raise ValueError(f"{supplier}: {quantity} is more than offered ({offered[supplier]})")
        if quantity > 0:
            clean.append({"supplier": supplier, "quantity": quantity})
            total += quantity
    if total > need:
        raise ValueError(f"total {total} is more than needed {need}")
    if not clean:
        raise ValueError("no allocation chosen")
    return clean


@traceable(name="Coordination AI decision", run_type="chain")
def decide_allocation(request: dict, offers: list[dict], use_ai: bool = True) -> dict:
    """request: request_id, facility, product, need, priority
    offers:  supplier, supplier_name, quantity, travel_time, supplier_eligible_stock, supplier_minimum_threshold
    Returns {"allocations": [{"supplier", "quantity"}], "reasoning": str, "method": "AI" | "rule-based fallback"}"""
    need = request["need"]
    client = get_client() if use_ai else None
    if client is None:
        why = "AI turned off" if not use_ai else "no LLM_API_KEY set"
        return {"allocations": rule_based_allocation(need, offers),
                "reasoning": f"Quickest supplier first ({why}).", "method": "rule-based fallback"}

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": json.dumps({"request": request, "offers": offers}, indent=2)}],
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("no JSON in the AI answer")
        answer = json.loads(match.group(0))
        allocations = validate(answer["allocations"], need, offers)
        return {"allocations": allocations, "reasoning": answer.get("reasoning", ""), "method": "AI"}
    except Exception as error:   # API error, bad JSON or a broken rule -> safe fallback
        return {"allocations": rule_based_allocation(need, offers),
                "reasoning": f"AI answer rejected ({error}); used quickest supplier first.",
                "method": "rule-based fallback"}
