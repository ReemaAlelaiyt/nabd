"""ACL-like messaging for Nabd (Section 3.6 - Communication and Interaction Protocols).

Every message carries a performative, sender, receiver, content, ontology and
conversation ID. Messages sent during simulation step t are delivered to the
receiver's inbox at step t + 1, which makes response deadlines measurable in
simulation steps (e.g. the 3-step CFP deadline).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Performative(str, Enum):
    INFORM = "INFORM"
    CFP = "CFP"
    PROPOSE = "PROPOSE"
    REFUSE = "REFUSE"
    ACCEPT = "ACCEPT"
    REJECT_PROPOSAL = "REJECT-PROPOSAL"
    REQUEST = "REQUEST"


class Ontology(str, Enum):
    INVENTORY = "Inventory"
    SUPPLY = "SupplyNegotiation"
    DELIVERY = "Delivery"
    COLD_CHAIN = "ColdChain"


@dataclass
class ACLMessage:
    performative: Performative
    sender: str
    receiver: str
    content: dict[str, Any]
    ontology: Ontology
    conversation_id: str
    msg_id: int = 0
    sent_step: int = 0
    delivery_step: int = 0

    @property
    def topic(self) -> str:
        return self.content.get("topic", "")

    @property
    def text(self) -> str:
        return self.content.get("text", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.msg_id,
            "step": self.sent_step,
            "sender": self.sender,
            "receiver": self.receiver,
            "performative": self.performative.value,
            "ontology": self.ontology.value,
            "conversation_id": self.conversation_id,
            "content": self.text,
        }

    def __str__(self) -> str:
        return (f"[step {self.sent_step:>3}] {self.sender} -> {self.receiver} "
                f"{self.performative.value} ({self.ontology.value}, {self.conversation_id}): {self.text}")


@dataclass
class MessageBus:
    """Routes messages between registered agents and keeps the full message log."""

    latency: int = 1
    log: list[ACLMessage] = field(default_factory=list)
    _agents: dict[str, Any] = field(default_factory=dict)
    _queue: list[ACLMessage] = field(default_factory=list)
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    def register(self, agent: Any) -> None:
        self._agents[agent.agent_id] = agent

    def send(self, msg: ACLMessage, step: int) -> ACLMessage:
        if msg.receiver not in self._agents:
            raise KeyError(f"Unknown receiver: {msg.receiver}")
        msg.msg_id = next(self._ids)
        msg.sent_step = step
        msg.delivery_step = step + self.latency
        self._queue.append(msg)
        self.log.append(msg)
        return msg

    def deliver(self, step: int) -> int:
        due = [m for m in self._queue if m.delivery_step <= step]
        self._queue = [m for m in self._queue if m.delivery_step > step]
        for m in due:
            self._agents[m.receiver].inbox.append(m)
        return len(due)

    @property
    def pending(self) -> int:
        return len(self._queue)
