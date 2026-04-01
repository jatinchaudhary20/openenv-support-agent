import os
import re
from openai import OpenAI

from env.support_env import SupportEnv
from env.models import Action

# Initialize OpenAI-compatible client (HF router)
client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")

# Regex for extracting action
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
You are a STRICT rule-based customer support agent.

Ticket:
{state.ticket}

State:
- category: {state.category}
- response: {state.response}
- resolved: {state.resolved}

RULES:
- If category is NONE → classify
- If category is SET and no response → respond
- If response exists → resolve
- Never repeat classify if category already exists

Actions:
- classify('billing' / 'refund' / 'technical')
- respond('your message')
- resolve()
- escalate()

Return ONLY ONE action.
"""


def smart_action_correction(state, raw_action):
    ticket = state.ticket.lower()

    # STEP 1: CLASSIFICATION (FIXED LOGIC)
    if state.category is None:

        if "refund" in ticket or "wrong product" in ticket:
            return "classify('refund')"

        elif "crash" in ticket or "error" in ticket or "app" in ticket:
            return "classify('technical')"

        elif "payment" in ticket or "charged" in ticket:
            return "classify('billing')"

        else:
            return "classify('technical')"

    # STEP 2: RESPONSE
    elif state.category is not None and state.response is None:
        if "respond" not in raw_action:
            return "respond('sorry for the inconvenience, we are resolving your issue')"
        return raw_action

    # STEP 3: RESOLVE
    elif state.response is not None and not state.resolved:
        return "resolve()"

    return raw_action


def main():
    env = SupportEnv()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"\nRunning task: {task}")

        state = env.reset(task)
        total_reward = 0

        for step in range(4):  # prevent loops
            prompt = build_prompt(state)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                text = response.choices[0].message.content
            except Exception as e:
                print("LLM Error:", e)
                text = "escalate()"

            raw_action = parse_action(text)

            # 🔥 CONTROL LOGIC (KEY)
            action_str = smart_action_correction(state, raw_action)

            print(f"Step {step}: {action_str}")

            action = Action(action_str=action_str)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

        print(f"{task} score: {total_reward}")


if __name__ == "__main__":
    main()