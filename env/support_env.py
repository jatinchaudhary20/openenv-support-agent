from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State as EnvState
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from openenv.core.rubrics import Rubric, RubricDict

# --- Rubric Composable Framework ---

class ClassificationRubric(Rubric):
    def forward(self, action: Action, obs: Observation) -> float:
        action_name = getattr(action, "name", getattr(action, "tool_name", ""))
        if action_name == "classify":
            category = obs.metadata["ticket_state"]["category"]
            expected = obs.metadata["expected_category"]
            return 1.0 if category == expected else -0.5
        return 0.0

class PriorityRubric(Rubric):
    def forward(self, action: Action, obs: Observation) -> float:
        action_name = getattr(action, "name", getattr(action, "tool_name", ""))
        if action_name == "classify":
            priority = obs.metadata["ticket_state"]["priority"]
            expected = obs.metadata["expected_priority"]
            return 1.0 if priority == expected else -0.5
        return 0.0

class EmpathyRubric(Rubric):
    def forward(self, action: Action, obs: Observation) -> float:
        action_name = getattr(action, "name", getattr(action, "tool_name", ""))
        if action_name == "respond":
            message = obs.metadata["ticket_state"]["response"].lower()
            empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
            return 1.0 if any(t in message for t in empathy_terms) else 0.2
        return 0.0

class ResolutionRubric(Rubric):
    def forward(self, action: Action, obs: Observation) -> float:
        action_name = getattr(action, "name", getattr(action, "tool_name", ""))
        if action_name == "resolve":
            state = obs.metadata["ticket_state"]
            if state["response"] and state["category"] and state["priority"]:
                return 2.0
            else:
                return -1.0
        return 0.0

class PenaltiesRubric(Rubric):
    def forward(self, action: Action, obs: Observation) -> float:
        state = obs.metadata["ticket_state"]
        frustration = state["frustration_level"]
        penalty = -0.5 * frustration
        if frustration >= 5 and not state["resolved"]:
            penalty -= 5.0 # rage quit
        elif state["turn_count"] >= 6 and not state["resolved"]:
            penalty -= 0.5
        return penalty

class SupportAgentRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.sub_rubrics = RubricDict({
            "classification": ClassificationRubric(),
            "priority": PriorityRubric(),
            "empathy": EmpathyRubric(),
            "resolution": ResolutionRubric(),
            "penalties": PenaltiesRubric(),
        })
        self.last_breakdown = {}

    def forward(self, action: Action, obs: Observation) -> float:
        total = 0.0
        self.last_breakdown = {}
        for name, rubric in self.sub_rubrics.items():
            score = rubric(action, obs)
            self.last_breakdown[name] = score
            total += score
        return total

# --- Main Environment ---

class TicketState(BaseModel):
    ticket: str
    category: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None
    resolved: bool = False
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    turn_count: int = 0
    frustration_level: int = 0

class SupportEnv(MCPEnvironment):
    def __init__(self):
        mcp = FastMCP("support_env")

        @mcp.tool
        def classify(category: str, priority: str) -> str:
            """Classify ticket category and priority."""
            category_norm = category.lower().strip()
            priority_norm = priority.lower().strip()

            self.ticket_state.category = category_norm
            self.ticket_state.priority = priority_norm
            self._advance_turn("assistant", f"classified={category_norm}, priority={priority_norm}")

            expected_category, expected_priority = self._expected_labels(self.ticket_state.ticket)
            if category_norm != expected_category or priority_norm != expected_priority:
                self.ticket_state.frustration_level += 2
                self.ticket_state.conversation.append({"role": "customer", "message": f"[Frustration INCREASED to {self.ticket_state.frustration_level}] You don't understand my issue AT ALL!"})

            return f"Ticket classified as {category_norm} with priority {priority_norm}"

        @mcp.tool
        def respond(message: str) -> str:
            """Respond to the customer."""
            self.ticket_state.response = message
            text = message.lower()
            empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
            has_empathy = any(term in text for term in empathy_terms)
            
            if not has_empathy:
                self.ticket_state.frustration_level += 2
                self.ticket_state.conversation.append({"role": "customer", "message": f"[Frustration INCREASED to {self.ticket_state.frustration_level}] This response sounds like a robot. Extremely unhelpful and unempathetic!"})

            self._advance_turn("assistant", message)
            return f"Responded with: {message}"

        @mcp.tool
        def resolve() -> str:
            """Mark the ticket as resolved."""
            if self.ticket_state.response and self.ticket_state.category and self.ticket_state.priority:
                self.ticket_state.resolved = True
                self.done = True
                return "Ticket successfully resolved."
            return "Cannot resolve without classification, priority, and response."

        super().__init__(mcp)
        self.rubric = SupportAgentRubric()
        self._state = EnvState(episode_id=str(uuid4()), step_count=0)
        self.ticket_state = TicketState(ticket="")
        self.last_reward_breakdown = {}
        self.events: List[Dict[str, Any]] = []
        self.done = False
        self.max_turns = 6

    def reset(self, task: str = "easy", **kwargs: Any) -> Observation:
        if task == "easy":
            ticket = "Payment failed but money deducted"
        elif task == "medium":
            ticket = "Received wrong product, want refund"
        else:
            ticket = "App crashes when I open it"

        self.ticket_state = TicketState(ticket=ticket)
        self.ticket_state.conversation.append({"role": "customer", "message": ticket})
        self._state = EnvState(episode_id=str(uuid4()), step_count=0)
        self.last_reward_breakdown = {}
        self.events = []
        self.done = False

        return Observation(
            done=False,
            reward=0.0,
            metadata={"ticket": self.ticket_state.model_dump()}
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        self.ticket_state.frustration_level += 1
        
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        
        if not obs.metadata:
            obs.metadata = {}
        
        obs.metadata["ticket_state"] = self.ticket_state.model_dump()
        exp_cat, exp_pri = self._expected_labels(self.ticket_state.ticket)
        obs.metadata["expected_category"] = exp_cat
        obs.metadata["expected_priority"] = exp_pri
        
        obs.reward = self.rubric(action, obs)
        self.last_reward_breakdown = self.rubric.last_breakdown
        
        if self.ticket_state.frustration_level >= 5 and not self.ticket_state.resolved:
            self.done = True
            self.ticket_state.conversation.append({"role": "customer", "message": "[RAGE QUIT] I've had enough of this! I'm leaving."})
        elif self.ticket_state.turn_count >= self.max_turns and not self.ticket_state.resolved:
            self.done = True
            
        obs.done = self.done
        action_name = getattr(action, "name", getattr(action, "tool_name", "unknown"))
        event = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "action": action_name,
            "reward": obs.reward,
            "reward_breakdown": self.last_reward_breakdown,
            "done": obs.done,
            "ticket_state": self.ticket_state.model_dump(),
        }
        self.events.append(event)
        obs.metadata.update({
            "event_log": self.events,
            "reward_breakdown": self.last_reward_breakdown
        })
        
        return obs

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(done=self.done, reward=-0.1, metadata={"error": f"Unsupported non-MCP action"})

    @property
    def state(self) -> EnvState:
        return self._state

    def _expected_labels(self, ticket: str) -> tuple[str, str]:
        ticket_lower = ticket.lower()
        if "payment" in ticket_lower:
            return "billing", "medium"
        if "refund" in ticket_lower or "wrong" in ticket_lower:
            return "refund", "medium"
        return "technical", "high"

    def _advance_turn(self, role: str, message: str) -> None:
        self.ticket_state.turn_count += 1
        self.ticket_state.conversation.append({"role": role, "message": message})