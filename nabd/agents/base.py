"""Base class shared by all Nabd agents."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..messages import ACLMessage, Ontology, Performative

if TYPE_CHECKING:
    from ..simulation import Simulation


class BaseAgent:
    agent_type = "Agent"
    architecture = "Reactive"

    def __init__(self, agent_id: str, sim: "Simulation"):
        self.agent_id = agent_id
        self.sim = sim
        self.inbox: list[ACLMessage] = []
        self.state = "idle"  # idle / active / waiting / alert
        self.status = "Waiting for first step"
        self.messages_sent = 0
        self.messages_received = 0

    # ---- shortcuts ----------------------------------------------------------
    @property
    def env(self):
        return self.sim.env

    @property
    def config(self):
        return self.sim.config

    @property
    def now(self) -> int:
        return self.sim.step_count

    # ---- actuators ----------------------------------------------------------
    def send(self, performative: Performative, receiver: str, ontology: Ontology,
             conversation_id: str, text: str, /, **content) -> ACLMessage:
        msg = ACLMessage(performative, self.agent_id, receiver, {"text": text, **content},
                         ontology, conversation_id)
        self.sim.bus.send(msg, self.now)
        self.messages_sent += 1
        return msg

    def log(self, text: str, level: str = "info") -> None:
        self.sim.monitor.log_event(self.now, self.agent_id, text, level)

    def set_status(self, state: str, status: str) -> None:
        self.state, self.status = state, status

    # ---- agent cycle: sense messages -> act ---------------------------------
    def step(self) -> None:
        """Traced in LangSmith as one run per agent per step (messages in, messages out, status)."""
        messages, self.inbox = self.inbox, []
        self.messages_received += len(messages)
        traced = traceable(name=self.agent_id, run_type="chain",
                           tags=[self.agent_type], metadata={"step": self.now},
                           process_inputs=lambda i: {"step": i["step"], "received": i["received"]})(self._run_step)
        traced(self.now, [str(m) for m in messages], _messages=messages)

    def _run_step(self, step: int, received: list[str], _messages: list[ACLMessage]) -> dict:
        sent_before = len(self.sim.bus.log)
        for m in _messages:
            self.handle(m)
        self.act()
        return {"state": self.state, "status": self.status,
                "sent": [str(m) for m in self.sim.bus.log[sent_before:]]}

    def handle(self, msg: ACLMessage) -> None:  # overridden
        pass

    def act(self) -> None:  # overridden
        pass

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "architecture": self.architecture,
            "state": self.state,
            "status": self.status,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }
