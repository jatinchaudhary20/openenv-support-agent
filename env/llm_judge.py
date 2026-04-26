import os
import json
from huggingface_hub import InferenceClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class LLMJudge:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct"):
        self.provider = os.getenv("EVAL_PROVIDER", "huggingface").lower().strip()
        self.hf_client = None
        self.openai_client = None

        hf_token = os.getenv("HF_TOKEN")
        openai_key = os.getenv("OPENAI_API_KEY")

        if self.provider == "openai":
            if OpenAI is not None and openai_key:
                self.openai_client = OpenAI(api_key=openai_key)
            else:
                print("WARNING: OPENAI_API_KEY missing (or openai package unavailable). Falling back to rule-based judge.")
        else:
            if hf_token:
                self.hf_client = InferenceClient(token=hf_token)
            else:
                print("WARNING: No HF_TOKEN. Judge will use rule-based fallback locally.")

        self.model_name = os.getenv("EVAL_MODEL_NAME", model_name)

    def evaluate_empathy(self, ticket: str, response: str) -> float:
        """Uses LLM to score empathy on a scale from 0.0 to 1.0"""
        if not self.hf_client and not self.openai_client:
            text = response.lower()
            return 1.0 if any(t in text for t in ["sorry", "apologize", "understand", "assist"]) else 0.2
            
        prompt = (
            "You are evaluating a customer support agent. "
            "Score the agent's empathy on a scale from 0.0 to 1.0 based on the response. "
            "Return ONLY a JSON dictionary with a single key 'score' mapping to the float value.\n\n"
            f"Customer Ticket: {ticket}\n"
            f"Agent Response: {response}"
        )
        try:
            if self.openai_client:
                completion = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1,
                )
                content = (completion.choices[0].message.content or "").strip()
            else:
                completion = self.hf_client.chat_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1,
                )
                content = completion.choices[0].message.content.strip()

            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(content[start:end+1])
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"LLM Judge failed: {e}")
            
        return 0.5
