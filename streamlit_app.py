import json
import random
from typing import Any, Dict, Tuple

import requests
import streamlit as st


API_BASE_URL = "http://localhost:7860"
MAX_STEPS = 5


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """POST helper with graceful error handling."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return True, data, ""
        return False, {}, f"Invalid response format from {path}"
    except requests.RequestException as exc:
        return False, {}, f"API request failed for {path}: {exc}"
    except json.JSONDecodeError:
        return False, {}, f"Failed to decode JSON response from {path}"


def _expected_labels(ticket: str) -> Tuple[str, str]:
    text = (ticket or "").lower()
    if "payment" in text:
        return "billing", "medium"
    if "refund" in text or "wrong" in text:
        return "refund", "medium"
    return "technical", "high"


def _reward_for_step(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, float]:
    breakdown = {"classification": 0.0, "priority": 0.0, "empathy": 0.0, "penalty": 0.0}
    tool = action["tool_name"]
    args = action["arguments"]

    if tool == "classify":
        expected_category, expected_priority = _expected_labels(state.get("ticket", ""))
        breakdown["classification"] = 1.0 if args.get("category") == expected_category else -0.5
        breakdown["priority"] = 1.0 if args.get("priority") == expected_priority else -0.5
    elif tool == "respond":
        msg = (args.get("message") or "").lower()
        empathy_terms = ["sorry", "understand", "apologize", "thanks", "assist"]
        breakdown["empathy"] = 1.0 if any(t in msg for t in empathy_terms) else 0.2
    elif tool == "resolve":
        if state.get("category") and state.get("priority") and state.get("response"):
            pass
        else:
            breakdown["penalty"] = -1.0

    return breakdown


def _infer_sentiment(ticket: str) -> str:
    text = (ticket or "").lower()
    angry_terms = ["angry", "frustrated", "worst", "terrible", "refund now", "not happy"]
    if any(term in text for term in angry_terms):
        return "angry"
    return "neutral"


def _build_fallback_response(state: Dict[str, Any]) -> str:
    ticket = state.get("ticket", "your issue")
    templates = [
        f"I understand your concern about '{ticket}' and I am sorry for the inconvenience. We are resolving this now.",
        f"Thanks for reporting this. I understand how frustrating '{ticket}' can be, and we are working on a fix.",
        f"I apologize for the trouble with '{ticket}'. We are on it and will update you shortly.",
    ]
    if st.session_state.get("deterministic_mode", False):
        return templates[0]
    return random.choice(templates)


def _choose_action_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
    ticket = (state.get("ticket") or "").lower()
    if not state.get("category"):
        if "payment" in ticket:
            return {"tool_name": "classify", "arguments": {"category": "billing", "priority": "medium"}}
        if "refund" in ticket or "wrong" in ticket:
            return {"tool_name": "classify", "arguments": {"category": "refund", "priority": "medium"}}
        return {"tool_name": "classify", "arguments": {"category": "technical", "priority": "high"}}
    if not state.get("response"):
        return {
            "tool_name": "respond",
            "arguments": {"message": _build_fallback_response(state)},
        }
    return {"tool_name": "resolve", "arguments": {}}


def _choose_action_hf(state: Dict[str, Any]) -> Dict[str, Any]:
    token = st.session_state.get("hf_token", "").strip()
    model = st.session_state.get("hf_model_name", "").strip()
    if not token or not model:
        raise ValueError("HF token/model missing")

    prompt = (
        "You are a customer support agent. "
        "Return ONLY strict JSON with keys tool_name and arguments. "
        "Allowed tool_name values: classify, respond, resolve. "
        "For classify, arguments must include category (billing/refund/technical) and priority (low/medium/high). "
        "For respond, arguments must include message with empathetic language. "
        "For resolve, arguments must be an empty object. "
        f"Current state: {json.dumps(state)}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(st.session_state.get("model_temperature", 0.2)),
        "max_tokens": 180,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON action in HF response")
    action = json.loads(content[start : end + 1])
    if action.get("tool_name") not in {"classify", "respond", "resolve"}:
        raise ValueError("Invalid tool_name from HF")
    if "arguments" not in action or not isinstance(action["arguments"], dict):
        raise ValueError("Invalid arguments from HF")
    return action


def _apply_action_to_state(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    next_state = dict(state)
    tool = action["tool_name"]
    args = action["arguments"]
    if tool == "classify":
        next_state["category"] = args.get("category")
        next_state["priority"] = args.get("priority")
    elif tool == "respond":
        next_state["response"] = args.get("message")
    elif tool == "resolve":
        if state.get("category") and state.get("priority") and state.get("response"):
            next_state["resolved"] = True
    return next_state


def _step_once() -> Tuple[bool, str]:
    state = st.session_state.current_state or {}
    try:
        action = _choose_action_hf(state)
        decision_source = "hf_model"
    except Exception:
        action = _choose_action_fallback(state)
        decision_source = "rule_fallback"
    payload = {"action": {"type": "call_tool", "tool_name": action["tool_name"], "arguments": action["arguments"]}}
    ok, data, err = _post_json("/step", payload)
    if not ok:
        return False, err

    breakdown = _reward_for_step(state, action)
    reward = breakdown["classification"] + breakdown["priority"] + breakdown["empathy"] + breakdown["penalty"]
    next_state = _apply_action_to_state(state, action)
    done = bool(next_state.get("resolved", False))

    st.session_state.total_score += reward
    st.session_state.current_state = next_state
    st.session_state.last_breakdown = breakdown
    st.session_state.done = done
    st.session_state.step_logs.append(
        {
            "step": len(st.session_state.step_logs) + 1,
            "action": payload["action"],
            "decision_source": decision_source,
            "reward": reward,
            "state": next_state,
            "done": done,
        }
    )
    return True, ""


st.set_page_config(page_title="Support Agent Visualizer", layout="wide")
st.title("OpenEnv Support Agent Visualizer")

if "current_state" not in st.session_state:
    st.session_state.current_state = {}
if "step_logs" not in st.session_state:
    st.session_state.step_logs = []
if "total_score" not in st.session_state:
    st.session_state.total_score = 0.0
if "done" not in st.session_state:
    st.session_state.done = False
if "last_breakdown" not in st.session_state:
    st.session_state.last_breakdown = {"classification": 0.0, "priority": 0.0, "empathy": 0.0, "penalty": 0.0}
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""
if "hf_model_name" not in st.session_state:
    st.session_state.hf_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
if "model_temperature" not in st.session_state:
    st.session_state.model_temperature = 0.2
if "deterministic_mode" not in st.session_state:
    st.session_state.deterministic_mode = False


left, right = st.columns([1, 2])

with left:
    st.subheader("Controls")
    st.text_input("HF Token (optional, model-first if set)", key="hf_token", type="password")
    st.text_input("HF Model", key="hf_model_name")
    st.slider("Model Temperature", min_value=0.0, max_value=1.0, step=0.1, key="model_temperature")
    st.checkbox("Deterministic fallback mode", key="deterministic_mode")
    st.caption("If HF fails or token/model is missing, app uses robust rule fallback.")
    ticket_text = st.text_area("Ticket Input", value=st.session_state.ticket_text, height=140)
    st.session_state.ticket_text = ticket_text

    if st.button("Reset Environment", use_container_width=True):
        if not ticket_text.strip():
            st.error("Please enter a support ticket first.")
        else:
            ok, data, err = _post_json("/reset", {"ticket": ticket_text.strip()})
            if not ok:
                st.error(err)
            else:
                st.session_state.current_state = {
                    "ticket": ticket_text.strip(),
                    "category": None,
                    "priority": None,
                    "response": None,
                    "resolved": False,
                    "escalated": False,
                }
                st.session_state.step_logs = []
                st.session_state.total_score = 0.0
                st.session_state.done = False
                st.session_state.last_breakdown = {
                    "classification": 0.0,
                    "priority": 0.0,
                    "empathy": 0.0,
                    "penalty": 0.0,
                }
                st.success("Environment reset.")

    if st.button("Run Agent", use_container_width=True):
        if not st.session_state.current_state:
            st.error("Reset the environment first.")
        else:
            error = ""
            for _ in range(MAX_STEPS):
                if st.session_state.done:
                    break
                ok, error = _step_once()
                if not ok:
                    break
            if error:
                st.error(error)
            else:
                st.success("Agent loop finished.")

    st.markdown("---")
    st.subheader("Reward Breakdown")
    rb = st.session_state.last_breakdown
    st.write(f"classification reward: `{rb['classification']:.2f}`")
    st.write(f"priority reward: `{rb['priority']:.2f}`")
    st.write(f"empathy reward: `{rb['empathy']:.2f}`")
    st.write(f"penalty: `{rb['penalty']:.2f}`")
    st.write(f"total score: `{st.session_state.total_score:.2f}`")

    st.markdown("---")
    st.subheader("Final Output")
    resolved = bool(st.session_state.current_state.get("resolved", False))
    st.write(f"resolved: `{'yes' if resolved else 'no'}`")
    st.write(f"total score: `{st.session_state.total_score:.2f}`")
    st.write(f"steps taken: `{len(st.session_state.step_logs)}`")

    # Optional, lightweight fields
    st.markdown("---")
    st.subheader("Signals")
    st.write(f"sentiment: `{_infer_sentiment(st.session_state.ticket_text)}`")
    escalated = st.session_state.current_state.get("escalated", False)
    st.write(f"escalation status: `{'yes' if escalated else 'no'}`")

with right:
    st.subheader("Step-by-Step Logs")
    if not st.session_state.step_logs:
        st.info("No steps yet. Reset environment, then run agent.")
    else:
        for log in st.session_state.step_logs:
            st.markdown(f"**Step {log['step']}**")
            st.write("Decision Source:", log.get("decision_source", "unknown"))
            st.write("Action:", log["action"])
            st.write("Reward:", f"{log['reward']:.2f}")
            st.write("Updated State:", log["state"])
            st.write("Done:", log["done"])
            st.markdown("---")
