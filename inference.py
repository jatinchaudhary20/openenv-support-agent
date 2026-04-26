import os
import json
import re
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
    observation = getattr(result, "observation", result)
    reward = getattr(result, "reward", None)
    done = getattr(result, "done", False)
    metadata = getattr(observation, "metadata", None) or {}
    return observation, reward, done, metadata


def _hf_decision(ticket_state, env_tools, retries=3):
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in env_tools])
    
    prompt = (
        "You are a helpful customer support agent. Base your decision ONLY on the provided ticket state.\n"
        "You MUST respond with a strict JSON object containing 'tool' and 'args' keys.\n\n"
        "Available Tools:\n"
        f"{tool_descriptions}\n\n"
        f"Current Ticket State: {json.dumps(ticket_state)}\n\n"
        "Output ONLY valid JSON. Examples:\n"
        "{\"tool\": \"classify\", \"args\": {\"category\": \"refund\", \"priority\": \"medium\"}}\n"
        "{\"tool\": \"respond\", \"args\": {\"message\": \"I apologize for the issue.\"}}\n"
        "{\"tool\": \"resolve\", \"args\": {}}"
    )
    
    for attempt in range(retries):
        try:
            completion = hf_client.chat_completion(
                model=HF_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.1 + (0.2 * attempt), # Increase temperature slightly on retries
            )
            content = completion.choices[0].message.content.strip()
            
            # Robust parser
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in output.")
                
            json_str = match.group(0)
            parsed = json.loads(json_str)
            
            if "tool" not in parsed or "args" not in parsed:
                raise ValueError("JSON missing 'tool' or 'args' keys.")
                
            # If tool doesn't match available ones, it's a hallucination
            if parsed["tool"] not in [t.name for t in env_tools]:
                raise ValueError(f"Hallucinated tool: {parsed['tool']}")
                
            return parsed
        except Exception as e:
            if attempt == retries - 1:
                print(f"[FAIL] Final LLM attempt failed: {str(e)}")
                raise
            print(f"[WARN] LLM parse fail, retrying... ({str(e)})")


def run_baseline():
    tasks = ["easy", "medium", "hard"]
    all_records = []

    print("Connecting to local support_env at http://localhost:7860...")
    with MCPToolClient(base_url="http://localhost:7860").sync() as env:
        for task in tasks:
            print(f"---\n[START] task={task}")
            obs = env.reset(task=task)
            
            tools = env.list_tools()

            total_reward = 0.0
            _, _, done, metadata = _result_parts(obs)
            ticket_state = metadata.get("ticket", {})

            for step in range(1, 7):
                print(f"[STEP {step}] AI Thinking...")

                try:
                    decision = _hf_decision(ticket_state, tools)
                    action_name = decision["tool"]
                    args = decision["args"]
                    
                    print(f"Agent Action chosen: {action_name}{args}")
                    
                    obs = env.call_tool(action_name, **args)
                    _, obs_reward, done, obs_metadata = _result_parts(obs)
                    
                    # Environment is now the sole source of truth for rewards and state
                    step_reward = obs_reward if isinstance(obs_reward, (int, float)) else 0.0
                    total_reward += step_reward
                    ticket_state = obs_metadata.get("ticket_state", ticket_state)
                    
                    log_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task": task,
                        "step": step,
                        "action": action_name,
                        "args": args,
                        "reward": step_reward,
                        "reward_breakdown": obs_metadata.get("reward_breakdown", {}),
                        "done": done,
                    }
                    
                except Exception as e:
                    print(f"Agent crashed during step: {e}")
                    step_reward = -1.0
                    total_reward += step_reward
                    done = True
                    log_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task": task,
                        "step": step,
                        "error": str(e),
                        "reward": step_reward,
                        "done": done
                    }

                all_records.append(log_record)
                print(json.dumps(log_record))
                print(f"Reward: {step_reward}")

                if done or ticket_state.get("resolved"):
                    print(f"[END] Task {task} finished in {step} steps. Total Score: {total_reward}")
                    break
                    
    with open(LOG_PATH, "w", encoding="utf-8") as fp:
        for record in all_records:
            fp.write(json.dumps(record) + "\n")
    print(f"Saved structured logs to {LOG_PATH}")

if __name__ == "__main__":
    run_baseline()