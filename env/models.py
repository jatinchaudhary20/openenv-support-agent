from pydantic import BaseModel
from typing import Optional


class State(BaseModel):
    ticket: str
    category: Optional[str] = None
    response: Optional[str] = None
    resolved: bool = False


class Action(BaseModel):
    action_str: str