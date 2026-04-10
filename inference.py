"""
inference.py - FinDocAgent-Env Baseline Inference Script
STDOUT FORMAT:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""
import os, sys, json, requests
from openai import OpenAI
from typing import List, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_URL      = os.getenv("ENV_URL", "https://niyasingh-findocagentenv.hf.space").rstrip("/")
BENCHMARK    = "findocagent"
MAX_STEPS    = 20

TASKS = [
    "task1_invoice_parser",
    "task2_anomaly_detector",
    "task3_three_way_reconciler",
]

# ── Per-task system prompts ─────────────────────────────────────────────────────

PROMPT_TASK1 = """You are a financial document extraction agent.
Extract fields from the invoice one at a time.
Respond ONLY with a single JSON object, no markdown, no explanation.

Format: {"action_type": "extract", "field": "<field_name>", "value": "<extracted_value>"}

Fields to extract (in order):
vendor_name, buyer_name, invoice_number, po_reference, issue_date,
due_date, subtotal, tax_amount, total_amount, currency

When all fields are extracted, respond with: {"action_type": "skip"}

Rules:
- Extract exactly ONE field per response
- Value must be a string exactly as it appears in the document
- Do NOT re-extract already extracted fields
- Do NOT add any text outside the JSON"""

PROMPT_TASK2 = """You are a financial compliance agent detecting invoice anomalies.
Respond ONLY with a single JSON object, no markdown, no explanation.

Policy rules to check:
1. No invoice may exceed Rs 5,00,000 without a PO reference
2. Duplicate invoice numbers = fraud
3. All invoices must have a valid PO number
4. Vendors must be on approved vendor list
5. Invoice date must not be in the future
6. Tax must be exactly 18% GST of subtotal

To flag an anomaly:
{"action_type": "flag", "field": "<anomaly_type>", "value": "<description>", "doc_ref": "<invoice_id>", "severity": "high", "reason": "<explanation>"}

anomaly_type must be one of:
duplicate_invoice, amount_exceeds_po_limit, missing_po_reference,
vendor_not_approved, date_in_future, tax_calculation_error

After flagging ALL anomalies, finalize with:
{"action_type": "reconcile", "field": "done", "value": "", "decision": "escalate", "reason": "Anomaly review complete"}

Rules:
- Review EVERY invoice in the batch
- Flag EACH anomaly you find with a separate action
- Only call reconcile ONCE after all flags are raised
- Do NOT skip - always flag or reconcile"""

PROMPT_TASK3 = """You are a financial reconciliation agent.
Respond ONLY with a single JSON object, no markdown, no explanation.

Compare the PO, Invoice, and GRN documents.
Find discrepancies and make a final decision.

To report a discrepancy:
{"action_type": "match", "field": "<discrepancy_type>", "value": "<item_affected>", "doc_ref": "invoice", "reason": "type:<discrepancy_type>"}

discrepancy_type must be one of:
quantity_mismatch, price_deviation, missing_line_item, short_delivery

After reporting ALL discrepancies, make final decision:
{"action_type": "reconcile", "field": "decision", "value": "<decision>", "decision": "<decision>", "reason": "<justification>"}

decision must be one of:
- "approve"  : all documents match, safe to pay
- "reject"   : critical discrepancy, do not pay
- "escalate" : discrepancies found, needs manager review

Rules:
- Report EACH discrepancy separately before reconciling
- Call reconcile ONCE at the end with your final decision
- If discrepancies exist, decision should be escalate or reject
- Do NOT skip"""

TASK_PROMPTS = {
    "task1_invoice_parser":       PROMPT_TASK1,
    "task2_anomaly_detector":     PROMPT_TASK2,
    "task3_three_way_reconciler": PROMPT_TASK3,
}

# ── Logging ─────────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    act = json.dumps(action, separators=(",", ":")) if isinstance(action, dict) else str(action)
    act = act.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)

# ── Env calls ───────────────────────────────────────────────────────────────────
def env_reset(task_id):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(session_id, action):
    r = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, **action}, timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM ─────────────────────────────────────────────────────────────────────────
def call_llm(client, system_prompt, messages):
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}] + messages[-8:],
            temperature=0.1,
            max_tokens=300,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return ""

def parse_action(text):
    text = text.strip()
    # Strip <think>...</think> tags (Qwen thinking mode)
    if "<think>" in text:
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if "```" in text:
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    # Find JSON object
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except Exception:
            pass
    return {"action_type": "skip"}

def build_user_msg(obs):
    task_id   = obs.get("task_id", "")
    doc       = obs.get("document", {})
    content   = doc.get("content", "") if isinstance(doc, dict) else str(doc)
    aux       = obs.get("aux_documents", {})
    extracted = obs.get("extracted", {})
    flags     = obs.get("flags", [])
    matches   = obs.get("matches", [])

    msg  = f"Step {obs.get('step_count',0)}/{obs.get('max_steps',20)}\n\n"
    msg += f"DOCUMENT:\n{content[:3000]}\n"
    if aux:
        msg += f"\nAUX DOCUMENTS (PO / GRN):\n{json.dumps(aux, indent=2)[:2000]}\n"
    if extracted:
        msg += f"\nALREADY EXTRACTED: {json.dumps(extracted)}\n"
    if flags:
        msg += f"\nFLAGS RAISED SO FAR ({len(flags)}): {json.dumps(flags)}\n"
    if matches:
        msg += f"\nDISCREPANCIES FOUND SO FAR ({len(matches)}): {json.dumps(matches)}\n"
    msg += f"\nVALID ACTIONS: {obs.get('valid_actions', [])}\n"
    msg += "\nRespond with ONE JSON action object."
    return msg

# ── Episode ─────────────────────────────────────────────────────────────────────
def run_episode(client, task_id):
    rewards, score, success, steps_done = [], 0.0, False, 0
    system_prompt = TASK_PROMPTS[task_id]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs        = env_reset(task_id)
        session_id = obs["session_id"]
        history    = []

        for step_num in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                score = float(obs.get("score", 0.0))
                break

            user_msg = build_user_msg(obs)
            history.append({"role": "user", "content": user_msg})

            llm_text = call_llm(client, system_prompt, history)
            action   = parse_action(llm_text)
            history.append({"role": "assistant", "content": llm_text or json.dumps(action)})

            last_error = None
            try:
                obs        = env_step(session_id, action)
                reward     = float(obs.get("reward", 0.0))
                done       = bool(obs.get("done", False))
                if reward < 0:
                    last_error = obs.get("message", None)
            except Exception as e:
                reward, done, last_error = 0.0, True, str(e)[:100]
                obs = {"done": True, "score": 0.0}

            rewards.append(reward)
            steps_done = step_num
            log_step(step=step_num, action=action, reward=reward, done=done, error=last_error)

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
        log_step(step=max(steps_done,1), action={"action_type":"skip"},
                 reward=0.0, done=True, error=str(e)[:100])

    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}

# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    client  = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    results = []
    for task_id in TASKS:
        try:
            result = run_episode(client, task_id)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
            result = {"task_id": task_id, "score": 0.0, "success": False}
        results.append(result)
    overall = sum(r["score"] for r in results) / len(results)
    print(f"# overall_score={overall:.2f}", flush=True)

if __name__ == "__main__":
    main()
