from fastapi import FastAPI
from env.support_env import SupportEnv
from env.models import Action

app = FastAPI()

env = SupportEnv()


@app.post("/reset")
def reset(task: str = "easy"):
    state = env.reset(task)
    return state.dict()


@app.post("/step")
def step(action: str):
    act = Action(action_str=action)
    state, reward, done, _ = env.step(act)
    return {
        "state": state.dict(),
        "reward": reward,
        "done": done
    }


@app.get("/state")
def get_state():
    return env.state.dict()