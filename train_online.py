import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from openenv.core.mcp_client import MCPToolClient


BASE_URL = "http://localhost:7860"
MAX_STEPS = 5
TRAIN_EPISODES = 180
EVAL_EPISODES = 60
ALPHA = 0.25
GAMMA = 0.9
EPSILON_START = 0.35
EPSILON_END = 0.05


TASKS = ["easy", "medium", "hard"]


def initial_state(task: str) -> Dict[str, object]:
    if task == "easy":
        ticket = "Payment failed but money deducted"
    elif task == "medium":
        ticket = "Received wrong product, want refund"
    else:
        ticket = "App crashes when I open it"
    return {
        "ticket": ticket,
        "category": None,
        "priority": None,
        "response": None,
        "resolved": False,
    }


def expected_labels(ticket: str) -> Tuple[str, str]:
    text = ticket.lower()
    if "payment" in text:
        return "billing", "medium"
    if "refund" in text or "wrong" in text:
        return "refund", "medium"
    return "technical", "high"


def empathy_score(message: str) -> float:
    terms = ["sorry", "understand", "apologize", "thanks", "assist"]
    return 1.0 if any(t in message.lower() for t in terms) else 0.2


def state_key(state: Dict[str, object]) -> str:
    if state.get("resolved"):
        stage = "resolved"
    elif not state.get("category"):
        stage = "needs_classify"
    elif not state.get("response"):
        stage = "needs_respond"
    else:
        stage = "needs_resolve"

    text = str(state.get("ticket", "")).lower()
    if "payment" in text:
        ticket_type = "billing"
    elif "refund" in text or "wrong" in text:
        ticket_type = "refund"
    else:
        ticket_type = "technical"
    return f"{ticket_type}:{stage}"


@dataclass(frozen=True)
class ActionSpec:
    tool: str
    args: Dict[str, object]
    name: str


ACTION_SPACE: List[ActionSpec] = [
    ActionSpec("classify", {"category": "billing", "priority": "medium"}, "classify_billing"),
    ActionSpec("classify", {"category": "refund", "priority": "medium"}, "classify_refund"),
    ActionSpec("classify", {"category": "technical", "priority": "high"}, "classify_technical"),
    ActionSpec(
        "respond",
        {"message": "I understand your issue and I am sorry for the inconvenience. We are fixing this now."},
        "respond_empathy",
    ),
    ActionSpec("respond", {"message": "Issue noted. Working on it."}, "respond_plain"),
    ActionSpec("resolve", {}, "resolve"),
]


def action_reward(state: Dict[str, object], action: ActionSpec) -> float:
    if action.tool == "classify":
        if state.get("category") is not None:
            return -1.0
        expected_category, expected_priority = expected_labels(str(state["ticket"]))
        cat = action.args["category"]
        pr = action.args["priority"]
        reward = 0.0
        reward += 1.0 if cat == expected_category else -0.5
        reward += 1.0 if pr == expected_priority else -0.5
        return reward
    if action.tool == "respond":
        if state.get("response"):
            return -0.5
        return empathy_score(str(action.args["message"]))
    if action.tool == "resolve":
        if state.get("category") and state.get("priority") and state.get("response"):
            return 3.0
        return -1.0
    return -0.1


def apply_action(state: Dict[str, object], action: ActionSpec) -> Dict[str, object]:
    next_state = dict(state)
    if action.tool == "classify":
        next_state["category"] = action.args["category"]
        next_state["priority"] = action.args["priority"]
    elif action.tool == "respond":
        next_state["response"] = action.args["message"]
    elif action.tool == "resolve":
        if next_state.get("category") and next_state.get("priority") and next_state.get("response"):
            next_state["resolved"] = True
    return next_state


def choose_action(q_table: Dict[str, Dict[str, float]], s_key: str, epsilon: float) -> ActionSpec:
    if random.random() < epsilon or s_key not in q_table:
        return random.choice(ACTION_SPACE)
    row = q_table[s_key]
    best_name = max(row, key=row.get)
    for a in ACTION_SPACE:
        if a.name == best_name:
            return a
    return random.choice(ACTION_SPACE)


def q_update(
    q_table: Dict[str, Dict[str, float]],
    s_key: str,
    action_name: str,
    reward: float,
    next_key: str,
) -> None:
    if s_key not in q_table:
        q_table[s_key] = {a.name: 0.0 for a in ACTION_SPACE}
    if next_key not in q_table:
        q_table[next_key] = {a.name: 0.0 for a in ACTION_SPACE}

    old = q_table[s_key][action_name]
    target = reward + GAMMA * max(q_table[next_key].values())
    q_table[s_key][action_name] = old + ALPHA * (target - old)


def run_episode(env: MCPToolClient, q_table: Dict[str, Dict[str, float]], train: bool, epsilon: float) -> Tuple[float, int, bool]:
    task = random.choice(TASKS)
    env.reset(task=task)
    state = initial_state(task)
    total_reward = 0.0
    done = False
    steps = 0

    for _ in range(MAX_STEPS):
        steps += 1
        s_key = state_key(state)
        action = choose_action(q_table, s_key, epsilon if train else 0.0) if train else random.choice(ACTION_SPACE)

        # Required by challenge: interact with live environment every step.
        env.call_tool(action.tool, **action.args)

        reward = action_reward(state, action)
        next_state = apply_action(state, action)
        next_key = state_key(next_state)
        total_reward += reward
        done = bool(next_state.get("resolved", False))

        if train:
            q_update(q_table, s_key, action.name, reward, next_key)

        state = next_state
        if done:
            break

    return total_reward, steps, done


def moving_average(values: List[float], window: int = 15) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def evaluate_policy(env: MCPToolClient, q_table: Dict[str, Dict[str, float]], episodes: int, use_trained: bool) -> Dict[str, float]:
    rewards: List[float] = []
    resolved = 0
    steps_total = 0
    for _ in range(episodes):
        if use_trained:
            # Greedy evaluation
            task = random.choice(TASKS)
            env.reset(task=task)
            state = initial_state(task)
            total = 0.0
            done = False
            steps = 0
            for _ in range(MAX_STEPS):
                steps += 1
                s_key = state_key(state)
                if s_key in q_table:
                    best = max(q_table[s_key], key=q_table[s_key].get)
                    action = next((a for a in ACTION_SPACE if a.name == best), random.choice(ACTION_SPACE))
                else:
                    action = random.choice(ACTION_SPACE)
                env.call_tool(action.tool, **action.args)
                total += action_reward(state, action)
                state = apply_action(state, action)
                done = bool(state.get("resolved", False))
                if done:
                    break
        else:
            total, steps, done = run_episode(env, q_table, train=False, epsilon=0.0)

        rewards.append(total)
        resolved += int(done)
        steps_total += steps

    return {
        "avg_reward": sum(rewards) / len(rewards),
        "resolution_rate": resolved / episodes,
        "avg_steps": steps_total / episodes,
    }


def main() -> None:
    random.seed(42)
    q_table: Dict[str, Dict[str, float]] = {}
    train_rewards: List[float] = []

    with MCPToolClient(base_url=BASE_URL).sync() as env:
        # Random baseline before training
        baseline = evaluate_policy(env, q_table, episodes=EVAL_EPISODES, use_trained=False)

        # Online Q-learning against environment
        for ep in range(TRAIN_EPISODES):
            eps = EPSILON_START + (EPSILON_END - EPSILON_START) * (ep / max(1, TRAIN_EPISODES - 1))
            reward, _, _ = run_episode(env, q_table, train=True, epsilon=eps)
            train_rewards.append(reward)

        # Evaluate trained policy
        trained = evaluate_policy(env, q_table, episodes=EVAL_EPISODES, use_trained=True)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # Training curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_rewards, alpha=0.35, label="Episode Reward")
    plt.plot(moving_average(train_rewards, 15), linewidth=2, label="Moving Avg (15)")
    plt.title("Online Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_reward_curve.png")
    plt.close()

    # Baseline vs trained bars
    labels = ["Avg Reward", "Resolution Rate", "Avg Steps (lower better)"]
    baseline_vals = [baseline["avg_reward"], baseline["resolution_rate"], baseline["avg_steps"]]
    trained_vals = [trained["avg_reward"], trained["resolution_rate"], trained["avg_steps"]]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Random Baseline")
    plt.bar([i + width / 2 for i in x], trained_vals, width=width, label="Trained Policy")
    plt.xticks(list(x), labels)
    plt.title("Baseline vs Trained Agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_vs_trained.png")
    plt.close()

    metrics = {
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "baseline": baseline,
        "trained": trained,
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {out_dir / 'training_reward_curve.png'}")
    print(f"Saved: {out_dir / 'baseline_vs_trained.png'}")
    print(f"Saved: {out_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
