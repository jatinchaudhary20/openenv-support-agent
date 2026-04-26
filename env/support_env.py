from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State as EnvState
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from env.ticket_generator import TicketGenerator
from env.llm_judge import LLMJudge

class TicketState(BaseModel):
    ticket: str
    expected_category: str = ""
    expected_priority: str = ""
    category: Optional[str] = None
    priority: Optional[str] = None
    response: Optional[str] = None
    resolved: bool = False
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    turn_count: int = 0
    user_satisfied: Optional[bool] = None
    emotion: str = "neutral"
    negative_streak: int = 0
    escalated: bool = False

class SupportEnv(MCPEnvironment):
    def __init__(self):
        mcp = FastMCP("support_env")
        
        self.ticket_generator = TicketGenerator()
        self.llm_judge = LLMJudge()

        @mcp.tool
        def classify(category: str, priority: str) -> str:
            """Classify ticket category and priority."""
            category_norm = category.lower().strip()
            priority_norm = priority.lower().strip()

            self.ticket_state.category = category_norm
            self.ticket_state.priority = priority_norm
            self._advance_turn("assistant", f"classified={category_norm}, priority={priority_norm}")

            expected_category = self.ticket_state.expected_category
            expected_priority = self.ticket_state.expected_priority
            
            self.latest_reward += self._classification_reward(category_norm, expected_category)
            self.latest_reward += self._priority_reward(priority_norm, expected_priority)
            self.last_reward_breakdown = {
                "classification": self._classification_reward(category_norm, expected_category),
                "priority": self._priority_reward(priority_norm, expected_priority),
                "empathy": 0.0,
                "resolution": 0.0,
            }
            return str(self._tool_payload(f"Ticket classified as {category_norm} with priority {priority_norm}"))

        @mcp.tool
        def respond(message: str) -> str:
            """Respond to the customer."""
            self.ticket_state.response = message
            empathy = self.llm_judge.evaluate_empathy(self.ticket_state.ticket, message)
            self.latest_reward += empathy
            
            # --- DYNAMIC AGGRAVATION MECHANISM ---
            if empathy < 0.3:
                self.ticket_state.ticket += " (FURIOUS UPDATE: Your agent was extremely unhelpful and rude! I want to speak to a manager immediately!)"
                self.ticket_state.expected_priority = "high"
                self.ticket_state.priority = "high"
                self._advance_turn("customer", "Your response was unhelpful. Escalate this issue now.")
            # -------------------------------------
            
            self._advance_turn("assistant", message)
            self.last_reward_breakdown = {
                "classification": 0.0,
                "priority": 0.0,
                "empathy": empathy,
                "resolution": 0.0,
            }
            return str(self._tool_payload(f"Responded with: {message}"))

        @mcp.tool
        def feedback(message: str) -> str:
            """Capture customer feedback sentiment/satisfaction for next decision."""
            user_text = message.strip()
            lowered = user_text.lower()
            negative_terms = ["not satisfied", "no", "still", "issue", "problem", "doesn't work", "dont work", "didn't help", "didnt help", "bad"]
            positive_terms = ["satisfied", "yes", "resolved", "works", "fixed", "thank you", "thanks"]
            high_negative_terms = ["furious", "angry", "frustrated", "worst", "unacceptable", "terrible"]

            if any(term in lowered for term in high_negative_terms):
                self.ticket_state.emotion = "high_negative"
            elif any(term in lowered for term in negative_terms):
                self.ticket_state.emotion = "negative"
            elif any(term in lowered for term in positive_terms):
                self.ticket_state.emotion = "positive"
            else:
                self.ticket_state.emotion = "neutral"

            if any(term in lowered for term in positive_terms) and not any(term in lowered for term in negative_terms):
                self.ticket_state.user_satisfied = True
                self.ticket_state.negative_streak = 0
            else:
                self.ticket_state.user_satisfied = False
                self.ticket_state.negative_streak += 1

            if self.ticket_state.negative_streak >= 3 and not self.ticket_state.resolved:
                self.ticket_state.escalated = True
                self._advance_turn("assistant", "We have raised your issue and a support person will contact you further.")
                self.done = True

            self._advance_turn("customer", user_text)
            self.last_reward_breakdown = {
                "classification": 0.0,
                "priority": 0.0,
                "empathy": 0.0,
                "resolution": 0.0,
            }
            return str(self._tool_payload("Feedback recorded."))

        @mcp.tool
        def resolve() -> str:
            """Mark the ticket as resolved."""
            resolution = -1.0
            if self.ticket_state.escalated:
                self.done = True
                self.last_reward_breakdown = {
                    "classification": 0.0,
                    "priority": 0.0,
                    "empathy": 0.0,
                    "resolution": 0.0,
                }
                return str(self._tool_payload("Issue already escalated to human support."))
            if (
                self.ticket_state.user_satisfied is True
                and self.ticket_state.response
                and self.ticket_state.category
                and self.ticket_state.priority
            ):
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
                return str(self._tool_payload("Ticket successfully resolved."))
            self.latest_reward += resolution
            self.last_reward_breakdown = {
                "classification": 0.0,
                "priority": 0.0,
                "empathy": 0.0,
                "resolution": resolution,
            }
            return str(self._tool_payload("Cannot resolve without satisfaction, classification, priority, and response."))

        super().__init__(mcp)
        self._state = EnvState(episode_id=str(uuid4()), step_count=0)
        self.ticket_state = TicketState(ticket="")
        self.latest_reward = 0.0
        self.last_reward_breakdown = {}
        self.events: List[Dict[str, Any]] = []
        self.done = False
        self.max_turns = 6

    def reset(self, task: str = "easy", **kwargs: Any) -> Observation:
        ticket_data = self.ticket_generator.generate(task=task)

        self.ticket_state = TicketState(
            ticket=ticket_data["ticket"],
            expected_category=ticket_data["expected_category"],
            expected_priority=ticket_data["expected_priority"]
        )
        self.ticket_state.conversation.append({"role": "customer", "message": ticket_data["ticket"]})
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

    def _classification_reward(self, predicted: str, expected: str) -> float:
        return 1.0 if predicted == expected else -0.5

    def _priority_reward(self, predicted: str, expected: str) -> float:
        return 1.0 if predicted == expected else -0.5

    def _advance_turn(self, role: str, message: str) -> None:
        self.ticket_state.turn_count += 1
        self.ticket_state.conversation.append({"role": role, "message": message})

    def _tool_payload(self, message: str) -> Dict[str, Any]:
        return {
            "message": message,
            "ticket_state": self.ticket_state.model_dump(),
            "step_reward": self.latest_reward,
            "reward_breakdown": self.last_reward_breakdown,
            "done": self.done,
        }

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=self.done,
            reward=0.0,
            metadata={
                "error": "Unsupported non-MCP action",
                "action_type": type(action).__name__,
            },
        )