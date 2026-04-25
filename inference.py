import os
import json
from datetime import datetime, timezone

from huggingface_hub import InferenceClient

try:
    from openenv.core.mcp_client import MCPToolClient
except ImportError:
    print("openenv-core not installed. Cannot run MCPToolClient inference.")
    exit(1)

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOG_PATH = os.getenv("EVAL_LOG_PATH", "evaluation_logs.jsonl")
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))


def _initial_ticket_state(task: str):
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


def _rule_based_decision(ticket_state):
    ticket = ticket_state.get("ticket", "").lower()
    if not ticket_state.get("category"):
        if "payment" in ticket:
            return {"tool": "classify", "args": {"category": "billing", "priority": "medium"}}
        if "refund" in ticket or "wrong" in ticket:
            return {"tool": "classify", "args": {"category": "refund", "priority": "medium"}}
        return {"tool": "classify", "args": {"category": "technical", "priority": "high"}}
    if not ticket_state.get("response"):
        return {
            "tool": "respond",
            "args": {"message": "I understand the issue and I am sorry for the inconvenience. We are fixing this now."},
        }
    return {"tool": "resolve", "args": {}}


def _hf_decision(ticket_state):
    prompt = (
        "You are a support agent. Decide next action as strict JSON with keys: "
        'tool and args. Allowed tools: classify, respond, resolve. '
        "If classifying, include category and priority. "
        "IMPORTANT: Manage the customer's frustration_level! If it reaches 5, they will rage quit. Always be empathetic. "
        f"Ticket state: {json.dumps(ticket_state)}"
    )
    completion = hf_client.chat_completion(
        model=HF_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.1,
    )
    content = completion.choices[0].message.content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON action returned by HF model")
    parsed = json.loads(content[start : end + 1])
    if "tool" not in parsed or "args" not in parsed:
        raise ValueError("Missing tool/args in HF action")
    return parsed


def _expected_labels(ticket: str):
    text = (ticket or "").lower()
    if "payment" in text:
        return "billing", "medium"
    if "refund" in text or "wrong" in text:
        return "refund", "medium"
    return "technical", "high"


def _reward_for_action(ticket_state, action_name, args):
    classification = 0.0
    priority = 0.0
    empathy = 0.0
    resolution = 0.0
    if action_name == "classify":
        expected_category, expected_priority = _expected_labels(ticket_state.get("ticket", ""))
        classification = 1.0 if args.get("category") == expected_category else -0.5
        priority = 1.0 if args.get("priority") == expected_priority else -0.5
    elif action_name == "respond":
        msg = (args.get("message") or "").lower()
        empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
        empathy = 1.0 if any(term in msg for term in empathy_terms) else 0.2
    elif action_name == "resolve":
        if ticket_state.get("category") and ticket_state.get("priority") and ticket_state.get("response"):
            resolution = 2.0
        else:
            resolution = -1.0
    total = classification + priority + empathy + resolution
    return total, {
        "classification": classification,
        "priority": priority,
        "empathy": empathy,
        "resolution": resolution,
    }


def _apply_action(ticket_state, action_name, args):
    next_state = dict(ticket_state)
    if action_name == "classify":
        next_state["category"] = args.get("category")
        next_state["priority"] = args.get("priority")
    elif action_name == "respond":
        next_state["response"] = args.get("message")
    elif action_name == "resolve":
        if next_state.get("category") and next_state.get("priority") and next_state.get("response"):
            next_state["resolved"] = True
    return next_state

def run_baseline():
    tasks = ["easy", "medium", "hard"]
    all_records = []

    print("Connecting to local support_env at http://localhost:7860...")
    with MCPToolClient(base_url="http://localhost:7860").sync() as env:
        for task in tasks:
            print(f"---\n[START] task={task}")
            env.reset(task=task)
            
            total_reward = 0
            ticket_state = _initial_ticket_state(task)
            done = False

            for step in range(1, 6):
                print(f"[STEP {step}] AI Thinking...")

                try:
                    decision = _hf_decision(ticket_state)
                    action_name = decision["tool"]
                    args = decision["args"]
                    decision_source = "hf_model"
                except Exception:
                    decision = _rule_based_decision(ticket_state)
                    action_name = decision["tool"]
                    args = decision["args"]
                    decision_source = "rule_fallback"

                print(f"Agent Action chosen: {action_name}{args}")
                env.call_tool(action_name, **args)
                reward, reward_breakdown = _reward_for_action(ticket_state, action_name, args)
                ticket_state = _apply_action(ticket_state, action_name, args)
                done = bool(ticket_state.get("resolved", False))
                total_reward += reward

                log_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": task,
                    "step": step,
                    "decision_source": decision_source,
                    "action": action_name,
                    "args": args,
                    "reward": reward,
                    "reward_breakdown": reward_breakdown,
                    "done": done,
                    "ticket_state": ticket_state,
                }
                all_records.append(log_record)
                print(json.dumps(log_record))
                print(f"Ticket State: {ticket_state} | Reward: {reward}")

                if done:
                    print(f"[END] Task {task} resolved successfully in {step} steps. Score: {total_reward}")
                    break
    with open(LOG_PATH, "w", encoding="utf-8") as fp:
        for record in all_records:
            fp.write(json.dumps(record) + "\n")
    print(f"Saved structured logs to {LOG_PATH}")

if __name__ == "__main__":
    run_baseline()