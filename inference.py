"""
inference.py - FinDocAgent-Env Baseline Inference Script

STDOUT FORMAT (mandatory):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import sys
import json
import requests
from openai import OpenAI
from typing import List, Optional

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_URL      = os.getenv("ENV_URL", "https://niyasingh-findocagentenv.hf.space")
BENCHMARK    = "findocagent"
MAX_STEPS    = 15

TASKS = [
    "task1_invoice_parser",
    "task2_anomaly_detector",
    "task3_three_way_reconciler",
]

SYSTEM_PROMPT = """You are a financial document processing agent.
Respond ONLY with a single valid JSON object — no markdown, no explanation.

For task1_invoice_parser, extract fields one at a time:
  {"action_type": "extract", "field": "<field_name>", "value": "<value>"}
  Fields: vendor_name, buyer_name, invoice_number, po_reference, issue_date,
          due_date, subtotal, tax_amount, total_amount, currency
  When all fields extracted: {"action_type": "skip"}

For task2_anomaly_detector, flag anomalies then finalize:
  {"action_type": "flag", "field": "<anomaly_type>", "value": "<desc>",
   "doc_ref": "<invoice_id>", "severity": "high", "reason": "<explanation>"}
  Then: {"action_type": "reconcile", "field": "done", "value": "",
         "decision": "escalate", "reason": "Review complete"}

For task3_three_way_reconciler, match discrepancies then decide:
  {"action_type": "match", "field": "<discrepancy_type>", "value": "<item>",
   "reason": "type:<discrepancy_type>"}
  Then: {"action_type": "reconcile", "field": "decision",
         "value": "escalate", "decision": "escalate",
         "reason": "Discrepancies found requiring review"}"""


# ── Logging helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Flatten action to single line, remove newlines
    action_flat = action.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={action_flat} reward={reward:.2f} "
          f"done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rewards_str}", flush=True)


# ── Env helpers ─────────────────────────────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_step(session_id: str, action: dict) -> dict:
    payload = {"session_id": session_id, **action}
    resp = requests.post(
        f"{ENV_URL}/step",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── LLM helper ──────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return ""

def parse_action(text: str) -> dict:
    """Parse LLM text to action dict. Always returns a valid action."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except Exception:
        pass
    return {"action_type": "skip"}

def build_user_message(obs: dict) -> str:
    doc = obs.get("document", {})
    content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
    extracted = obs.get("extracted", {})
    flags     = obs.get("flags", [])
    matches   = obs.get("matches", [])
    aux       = obs.get("aux_documents", {})

    msg  = f"TASK: {obs.get('task_id', '')}\n"
    msg += f"Step: {obs.get('step_count', 0)}/{obs.get('max_steps', 20)}\n\n"
    msg += f"DOCUMENT:\n{content[:2000]}\n"
    if aux:
        msg += f"\nAUX DOCUMENTS:\n{json.dumps(aux, indent=2)[:1000]}\n"
    if extracted:
        msg += f"\nALREADY EXTRACTED: {json.dumps(extracted)}\n"
    if flags:
        msg += f"\nFLAGS RAISED: {json.dumps(flags)}\n"
    if matches:
        msg += f"\nMATCHES: {json.dumps(matches)}\n"
    msg += f"\nVALID ACTIONS: {obs.get('valid_actions', [])}\n"
    msg += "\nRespond with one JSON action object only."
    return msg


# ── Episode runner ───────────────────────────────────────────────────────────────
def run_episode(client: OpenAI, task_id: str) -> dict:
    rewards    = []
    score      = 0.0
    success    = False
    steps_done = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs        = env_reset(task_id)
        session_id = obs["session_id"]
        history    = []

        for step_num in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                score = float(obs.get("score", 0.0))
                break

            # Build prompt
            user_msg = build_user_message(obs)
            history.append({"role": "user", "content": user_msg})

            # Get LLM action
            llm_text = call_llm(client, [{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:])
            action   = parse_action(llm_text)
            history.append({"role": "assistant", "content": llm_text or json.dumps(action)})

            action_str = json.dumps(action, separators=(",", ":"))
            last_error = None

            # Step env
            try:
                obs        = env_step(session_id, action)
                reward     = float(obs.get("reward", 0.0))
                done       = bool(obs.get("done", False))
                if reward < 0:
                    last_error = obs.get("message", None)
            except Exception as e:
                reward     = 0.0
                done       = True
                last_error = str(e)[:100]
                obs        = {"done": True, "score": 0.0}

            rewards.append(reward)
            steps_done = step_num
            log_step(step=step_num, action=action_str, reward=reward,
                     done=done, error=last_error)

            if done:
                score = float(obs.get("score", 0.0))
                break

        if not rewards:
            rewards = [0.0]

        score   = max(0.0, min(1.0, score))
        success = score >= 0.55

    except Exception as e:
        if not rewards:
            rewards = [0.0]
        last_err = str(e)[:100]
        log_step(step=max(steps_done, 1), action='{"action_type":"skip"}',
                 reward=0.0, done=True, error=last_err)
        score   = 0.0
        success = False

    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

    return {"task_id": task_id, "score": score,
            "success": success, "steps": len(rewards)}


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    client  = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    results = []

    for task_id in TASKS:
        try:
            result = run_episode(client, task_id)
        except Exception as e:
            # Never let one task crash the whole script
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
            result = {"task_id": task_id, "score": 0.0, "success": False}
        results.append(result)

    overall = sum(r["score"] for r in results) / len(results)
    print(f"# overall_score={overall:.2f}", flush=True)

if __name__ == "__main__":
    main()
