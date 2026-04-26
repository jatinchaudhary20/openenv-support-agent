from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State as EnvState
from fastmcp import FastMCP
from pydantic import BaseModel, Field

class TicketState(BaseModel):
    ticket: str
    category: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None
    resolved: bool = False
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    turn_count: int = 0

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
            self.latest_reward += self._classification_reward(category_norm, expected_category)
            self.latest_reward += self._priority_reward(priority_norm, expected_priority)
            self.last_reward_breakdown = {
                "classification": self._classification_reward(category_norm, expected_category),
                "priority": self._priority_reward(priority_norm, expected_priority),
                "empathy": 0.0,
                "resolution": 0.0,
            }
            return f"Ticket classified as {category_norm} with priority {priority_norm}"

        @mcp.tool
        def respond(message: str) -> str:
            """Respond to the customer."""
            self.ticket_state.response = message
            empathy = self._empathy_reward(message)
            self.latest_reward += empathy
            self._advance_turn("assistant", message)
            self.last_reward_breakdown = {
                "classification": 0.0,
                "priority": 0.0,
                "empathy": empathy,
                "resolution": 0.0,
            }
            return f"Responded with: {message}"

        @mcp.tool
        def resolve() -> str:
            """Mark the ticket as resolved."""
            resolution = -1.0
            if self.ticket_state.response and self.ticket_state.category and self.ticket_state.priority:
                self.ticket_state.resolved = True
                resolution = 2.0
                self.latest_reward += resolution
                self.done = True
                self.last_reward_breakdown = {
                    "classification": 0.0,
                    "priority": 0.0,
                    "empathy": 0.0,
                    "resolution": resolution,
                }
                return "Ticket successfully resolved."
            self.latest_reward += resolution
            self.last_reward_breakdown = {
                "classification": 0.0,
                "priority": 0.0,
                "empathy": 0.0,
                "resolution": resolution,
            }
            return "Cannot resolve without classification, priority, and response."

        super().__init__(mcp)
        self._state = EnvState(episode_id=str(uuid4()), step_count=0)
        self.ticket_state = TicketState(ticket="")
        self.latest_reward = 0.0
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
        self.latest_reward = 0.0
        self.last_reward_breakdown = {}
        self.events = []
        self.done = False

        return Observation(
            done=False,
            reward=0.0,
            metadata={"ticket": self.ticket_state.model_dump()}
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self.latest_reward = 0.0
        self._state.step_count += 1
        
        # Super step executes the FastMCP tool
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        
        # Override observation with our environment specifics
        obs.reward = self.latest_reward
        if self.ticket_state.turn_count >= self.max_turns and not self.ticket_state.resolved:
            self.done = True
            obs.reward -= 0.5
            self.last_reward_breakdown["turn_limit_penalty"] = -0.5
        obs.done = self.done
        if not obs.metadata:
            obs.metadata = {}
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
        obs.metadata.update(
            {
                "ticket_state": self.ticket_state.model_dump(),
                "reward_breakdown": self.last_reward_breakdown,
                "event_log": self.events,
            }
        )
        
        return obs

    @property
    def state(self) -> EnvState:
        return self._state

    def _expected_labels(self, ticket: str) -> tuple[str, str]:
        ticket_lower = ticket.lower()
        billing_terms = {
            "payment",
            "bill",
            "billing",
            "charged",
            "charge",
            "invoice",
            "deducted",
            "transaction",
            "upi",
            "card",
            "wallet",
            "subscription",
        }
        refund_terms = {
            "refund",
            "wrong product",
            "return",
            "replacement",
            "cancel order",
            "cancelled order",
            "order issue",
            "defective",
            "damaged",
            "not received",
        }
        technical_terms = {
            "crash",
            "bug",
            "error",
            "issue logging in",
            "login",
            "otp",
            "app not opening",
            "not opening",
            "slow",
            "freeze",
            "stuck",
            "failed to load",
            "server down",
        }
        urgent_terms = {
            "urgent",
            "asap",
            "immediately",
            "furious",
            "angry",
            "frustrated",
            "worst",
            "unacceptable",
        }

        if any(term in ticket_lower for term in refund_terms):
            category = "refund"
        elif any(term in ticket_lower for term in billing_terms):
            category = "billing"
        elif any(term in ticket_lower for term in technical_terms):
            category = "technical"
        else:
            category = "technical"

        priority = "high" if category == "technical" or any(term in ticket_lower for term in urgent_terms) else "medium"
        return category, priority

    def _classification_reward(self, predicted: str, expected: str) -> float:
        return 1.0 if predicted == expected else -0.5

    def _priority_reward(self, predicted: str, expected: str) -> float:
        return 1.0 if predicted == expected else -0.5

    def _empathy_reward(self, message: str) -> float:
        text = message.lower()
        empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
        return 1.0 if any(term in text for term in empathy_terms) else 0.2

    def _advance_turn(self, role: str, message: str) -> None:
        self.ticket_state.turn_count += 1
        self.ticket_state.conversation.append({"role": role, "message": message})

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        # This env only supports MCP tool actions via CallToolAction.
        return Observation(
            done=self.done,
            reward=0.0,
            metadata={
                "error": "Unsupported non-MCP action",
                "action_type": type(action).__name__,
            },
        )