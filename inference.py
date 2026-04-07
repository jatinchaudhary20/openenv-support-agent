import os
import re
from openai import OpenAI

from env.support_env import SupportEnv
from env.models import Action

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

ACTION_PATTERN = re.compile(r"[a-zA-Z_]+\([^)]*\)")


def parse_action(text):
    if not text:
        return "escalate()"
    match = ACTION_PATTERN.search(text)
    if match:
        return match.group(0)
    return "escalate()"


def build_prompt(state):
    return f"""
You are a customer support agent.

Ticket:
{state.ticket}

State:
category={state.category}
response={state.response}
resolved={state.resolved}

Choose ONE action:
- classify('billing')
- classify('refund')
- classify('technical')
- respond('message')
- resolve()
- escalate()

Return ONLY the action.
"""


def run_baseline():
    env = SupportEnv()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        state = env.reset(task)
        total_reward = 0

        print(f"[START] task={task}", flush=True)

        for step in range(1, 6):
            prompt = build_prompt(state)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                text = response.choices[0].message.content
            except Exception as e:
                text = "escalate()"

            action_str = parse_action(text)

            # fallback logic (safety)
            if state.category is None:
                if "payment" in state.ticket.lower():
                    action_str = "classify('billing')"
                elif "refund" in state.ticket.lower() or "wrong" in state.ticket.lower():
                    action_str = "classify('refund')"
                else:
                    action_str = "classify('technical')"
            elif state.response is None:
                action_str = "respond('ok')"
            else:
                action_str = "resolve()"

            state, reward, done, _ = env.step(Action(action_str=action_str))
            total_reward += reward

            print(f"[STEP] step={step} reward={reward}", flush=True)

            if done:
                break

        print(f"[END] task={task} score={total_reward} steps={step}", flush=True)


if __name__ == "__main__":
    run_baseline()