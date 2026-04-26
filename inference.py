import os
import json
from datetime import datetime, timezone

from huggingface_hub import InferenceClient

try:
    from openenv.core.mcp_client import MCPToolClient
except ImportError:
    print("openenv-core not installed. Cannot run MCPToolClient inference.")
    exit(1)

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
LOG_PATH = os.getenv("EVAL_LOG_PATH", "evaluation_logs.jsonl")
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))


def _result_parts(result):
    """Handle both OpenEnv result shapes: StepResult or direct observation."""
    observation = getattr(result, "observation", result)
    reward = getattr(result, "reward", None)
    done = getattr(result, "done", False)
    metadata = getattr(observation, "metadata", None) or {}
    return observation, reward, done, metadata


def _initial_ticket_state(task: str) -> dict:
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


def _expected_labels(ticket: str) -> tuple[str, str]:
    text = ticket.lower()
    if any(term in text for term in ["refund", "wrong product", "return", "replacement", "defective", "damaged"]):
        category = "refund"
    elif any(term in text for term in ["payment", "billing", "charged", "invoice", "deducted", "subscription"]):
        category = "billing"
    else:
        category = "technical"
    priority = "high" if category == "technical" or any(term in text for term in ["urgent", "asap", "furious", "angry"]) else "medium"
    return category, priority


def _reward_for_action(tool: str, args: dict, st: dict) -> float:
    if tool == "classify":
        exp_category, exp_priority = _expected_labels(st.get("ticket", ""))
        c = 1.0 if (args.get("category") or "").lower() == exp_category else -0.5
        p = 1.0 if (args.get("priority") or "").lower() == exp_priority else -0.5
        return c + p
    if tool == "respond":
        message = (args.get("message") or "").lower()
        empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
        return 1.0 if any(term in message for term in empathy_terms) else 0.2
    if tool == "resolve":
        return 2.0 if st.get("category") and st.get("priority") and st.get("response") else -1.0
    return 0.0


def _apply_action(tool: str, args: dict, st: dict) -> dict:
    next_state = dict(st)
    if tool == "classify":
        next_state["category"] = (args.get("category") or "").lower()
        next_state["priority"] = (args.get("priority") or "").lower()
    elif tool == "respond":
        next_state["response"] = args.get("message", "")
    elif tool == "resolve" and next_state.get("category") and next_state.get("priority") and next_state.get("response"):
        next_state["resolved"] = True
    return next_state


def _rule_based_decision(ticket_state):
    ticket = ticket_state.get("ticket", "").lower()
    billing_terms = [
        "payment", "bill", "billing", "charged", "charge", "invoice",
        "deducted", "transaction", "upi", "card", "wallet", "subscription",
    ]
    refund_terms = [
        "refund", "wrong product", "return", "replacement", "cancel order",
        "cancelled order", "order issue", "defective", "damaged", "not received",
    ]
    technical_terms = [
        "crash", "bug", "error", "issue logging in", "login", "otp",
        "app not opening", "not opening", "slow", "freeze", "stuck",
        "failed to load", "server down",
    ]
    urgent_terms = ["urgent", "asap", "immediately", "furious", "angry", "frustrated", "worst", "unacceptable"]

    def infer_labels(text):
        if any(t in text for t in refund_terms):
            category = "refund"
        elif any(t in text for t in billing_terms):
            category = "billing"
        elif any(t in text for t in technical_terms):
            category = "technical"
        else:
            category = "technical"
        priority = "high" if category == "technical" or any(t in text for t in urgent_terms) else "medium"
        return category, priority

    if not ticket_state.get("category"):
        category, priority = infer_labels(ticket)
        return {"tool": "classify", "args": {"category": category, "priority": priority}}
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

def run_baseline():
    tasks = ["easy", "medium", "hard"]
    all_records = []

    print("Connecting to local support_env at http://localhost:7860...")
    with MCPToolClient(base_url="http://localhost:7860").sync() as env:
        for task in tasks:
            print(f"---\n[START] task={task}")
            obs = env.reset(task=task)
            
            # Fetch tools
            tools = env.list_tools()
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

            total_reward = 0.0
            _, _, done, metadata = _result_parts(obs)
            ticket_state = metadata.get("ticket", {}) or _initial_ticket_state(task)

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
                obs = env.call_tool(action_name, **args)
                _, obs_reward, done, obs_metadata = _result_parts(obs)

                local_reward = _reward_for_action(action_name, args, ticket_state)
                step_reward = obs_reward if isinstance(obs_reward, (int, float)) else local_reward
                total_reward += step_reward

                ticket_state = obs_metadata.get("ticket_state", {}) or _apply_action(action_name, args, ticket_state)
                done = bool(done) or bool(ticket_state.get("resolved"))

                log_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": task,
                    "step": step,
                    "decision_source": decision_source,
                    "action": action_name,
                    "args": args,
                    "reward": step_reward,
                    "reward_breakdown": obs_metadata.get("reward_breakdown", {}),
                    "done": done,
                }
                all_records.append(log_record)
                print(json.dumps(log_record))
                print(f"Observation: {obs_metadata} | Reward: {step_reward}")

                if done:
                    print(f"[END] Task {task} resolved successfully in {step} steps. Score: {total_reward}")
                    break
    with open(LOG_PATH, "w", encoding="utf-8") as fp:
        for record in all_records:
            fp.write(json.dumps(record) + "\n")
    print(f"Saved structured logs to {LOG_PATH}")

if __name__ == "__main__":
    run_baseline()