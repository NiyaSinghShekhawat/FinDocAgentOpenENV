"""
inference.py - FinDocAgent-Env Baseline Inference Script

Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Required env vars:
  API_BASE_URL  - LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    - Model identifier  (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      - HuggingFace / API key
  ENV_URL       - HF Space base URL (e.g. https://niyasingh-findocagent-env.hf.space)
"""

import os
import json
import re
import requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]           # REQUIRED — no silent fallback
MODEL_NAME   = os.environ["MODEL_NAME"]             # REQUIRED
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
ENV_URL      = os.environ.get("ENV_URL", "https://niyasingh-findocagent-env.hf.space").rstrip("/")
BENCHMARK    = "findocagent"

# Per-task step budgets (must stay under 20-min total runtime)
TASK_MAX_STEPS = {
    "task1_invoice_parser":       20,
    "task2_anomaly_detector":     30,
    "task3_three_way_reconciler": 50,
}
PASS_THRESHOLD = {
    "task1_invoice_parser":       0.70,
    "task2_anomaly_detector":     0.60,
    "task3_three_way_reconciler": 0.55,
}

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = list(TASK_MAX_STEPS.keys())

# ── System prompts (one per task) ──────────────────────────────────────────────
SYSTEM_PROMPTS = {

"task1_invoice_parser": """You are a financial document extraction agent.

You will receive an invoice document. Extract fields ONE AT A TIME using this exact JSON format:
  {"action_type": "extract", "field": "<field_name>", "value": "<extracted_value>"}

Fields to extract (in this order):
  vendor_name, buyer_name, invoice_number, po_reference,
  issue_date, due_date, subtotal, tax_amount, total_amount, currency, n_line_items

Rules:
- Return the value exactly as written in the document (dates, amounts, etc.)
- For n_line_items: count the number of line items in the ITEMS section
- For currency: extract the currency code (e.g. INR, USD)
- For subtotal/tax_amount/total_amount: return the numeric value only (e.g. "36914.67")
- If a field is truly missing: {"action_type": "skip", "field": "<field_name>"}
- Extract ONE field per response. Do NOT batch multiple fields.
- Respond with ONLY a raw JSON object. No markdown, no explanation.""",

"task2_anomaly_detector": """You are a financial compliance agent reviewing invoices for policy violations.

For each anomaly found, respond with:
  {"action_type": "flag", "field": "<anomaly_type>", "value": "<brief_description>",
   "doc_ref": "<invoice_id>", "severity": "low|medium|high|critical", "reason": "<explanation>"}

Anomaly types to check: duplicate_invoice, amount_mismatch, missing_po, unauthorized_vendor,
  tax_error, date_violation, quantity_error, price_discrepancy

When all anomalies are flagged (or none found), finalize with:
  {"action_type": "reconcile", "field": "done", "value": "reviewed",
   "decision": "escalate", "reason": "Anomaly review complete"}

Respond with ONLY a raw JSON object. No markdown, no explanation.""",

"task3_three_way_reconciler": """You are a 3-way PO/Invoice/GRN reconciliation agent.

You will receive a Purchase Order (PO), an Invoice, and a Goods Receipt Note (GRN).
Compare them and identify discrepancies ONE AT A TIME:
  {"action_type": "match", "field": "<discrepancy_type>", "value": "<item_description>",
   "reason": "type:<discrepancy_type>"}

Discrepancy types: quantity_mismatch, price_mismatch, item_not_in_po, item_not_received,
  date_discrepancy, duplicate_charge, tax_discrepancy

After all discrepancies are matched, make a final decision:
  {"action_type": "reconcile", "field": "decision", "value": "<approve|reject|escalate>",
   "decision": "<approve|reject|escalate>", "reason": "<justification>"}

Decision rules:
- approve: minor discrepancies within tolerance (< 2%)
- reject: major discrepancies or fraud indicators
- escalate: uncertain or requires human review

Respond with ONLY a raw JSON object. No markdown, no explanation.""",
}


# ── HTTP helpers ────────────────────────────────────────────────────────────────
def call_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_step(session_id: str, action: dict) -> dict:
    payload = {"session_id": session_id, **action}
    r = requests.post(f"{ENV_URL}/step", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_grader(session_id: str) -> dict:
    try:
        r = requests.post(f"{ENV_URL}/grader", json={"session_id": session_id}, timeout=30)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


# ── Prompt builder ──────────────────────────────────────────────────────────────
def build_user_message(obs: dict, extracted_fields: list[str]) -> str:
    """Build a grounded prompt that includes the full document text."""
    doc = obs.get("document", {})
    content = doc.get("content", "") if isinstance(doc, dict) else str(doc)

    aux       = obs.get("aux_documents") or {}
    extracted = obs.get("extracted") or {}
    flags     = obs.get("flags") or {}
    matches   = obs.get("matches") or {}
    task_id   = obs.get("task_id", "")
    step      = obs.get("step_count", 0)
    max_steps = obs.get("max_steps", 20)

    msg = f"TASK: {task_id} | Step {step}/{max_steps}\n\n"
    msg += f"DOCUMENT:\n{content}\n"

    if aux:
        msg += f"\nAUX DOCUMENTS:\n{json.dumps(aux, indent=2)}\n"

    if extracted:
        msg += f"\nALREADY EXTRACTED: {json.dumps(extracted)}\n"
        msg += f"FIELDS ALREADY DONE: {extracted_fields}\n"

    if flags:
        msg += f"\nFLAGS RAISED SO FAR: {json.dumps(flags)}\n"

    if matches:
        msg += f"\nMATCHES SO FAR: {json.dumps(matches)}\n"

    valid = obs.get("valid_actions", [])
    msg += f"\nVALID ACTIONS: {valid}\n"
    msg += "\nRespond with a single JSON action object only."
    return msg


# ── LLM call + JSON parse ───────────────────────────────────────────────────────
# Ordered field list for task1 — used to pick the next unextracted field
TASK1_FIELDS = [
    "vendor_name", "buyer_name", "invoice_number", "po_reference",
    "issue_date", "due_date", "subtotal", "tax_amount",
    "total_amount", "currency", "n_line_items",
]

TASK2_ANOMALY_TYPES = [
    "duplicate_invoice", "amount_mismatch", "missing_po",
    "unauthorized_vendor", "tax_error", "date_violation",
    "quantity_error", "price_discrepancy",
]

TASK3_DISCREPANCY_TYPES = [
    "quantity_mismatch", "price_mismatch", "item_not_in_po",
    "item_not_received", "date_discrepancy", "duplicate_charge", "tax_discrepancy",
]


def parse_action(text: str, task_id: str, extracted_fields: list[str], step: int) -> dict:
    """Parse LLM text to action dict. Falls back gracefully — never returns bare skip."""
    clean = text.strip()
    # Strip markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE)
    clean = clean.strip()

    # Extract JSON object if embedded in text
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)

    try:
        action = json.loads(clean)
        if "action_type" in action:
            return action
    except Exception:
        pass

    # ── Fallback: pick the next sensible action instead of blind skip ──
    if "task1" in task_id:
        remaining = [f for f in TASK1_FIELDS if f not in extracted_fields]
        if remaining:
            return {"action_type": "skip", "field": remaining[0]}
        return {"action_type": "skip", "field": "vendor_name"}

    elif "task2" in task_id:
        flagged = set(extracted_fields)
        remaining = [a for a in TASK2_ANOMALY_TYPES if a not in flagged]
        if remaining:
            return {"action_type": "skip", "field": remaining[0]}
        return {"action_type": "reconcile", "field": "done", "value": "reviewed",
                "decision": "escalate", "reason": "Fallback: parse error"}

    elif "task3" in task_id:
        matched = set(extracted_fields)
        remaining = [d for d in TASK3_DISCREPANCY_TYPES if d not in matched]
        if remaining:
            return {"action_type": "skip", "field": remaining[0]}
        return {"action_type": "reconcile", "field": "decision", "value": "escalate",
                "decision": "escalate", "reason": "Fallback: parse error"}

    return {"action_type": "skip", "field": "unknown"}


# ── Episode runner ──────────────────────────────────────────────────────────────
def run_episode(task_id: str) -> dict:
    max_steps       = TASK_MAX_STEPS[task_id]
    pass_thresh     = PASS_THRESHOLD[task_id]
    system_prompt   = SYSTEM_PROMPTS[task_id]
    rewards         = []
    extracted_fields: list[str] = []   # track what we've already done
    history         = []               # conversation history for multi-turn context
    score           = 0.0
    success         = False
    session_id      = ""

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs = call_reset(task_id)
        print(json.dumps(obs, indent=2)[:2000], flush=True)
        session_id = obs["session_id"]

        for step_num in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            # Build grounded prompt
            user_msg = build_user_message(obs, extracted_fields)
            history.append({"role": "user", "content": user_msg})

            # Call LLM
            llm_text   = ""
            last_error = None
            try:
                response = client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [{"role": "system", "content": system_prompt}] + history[-6:],
                    temperature = 0.0,
                    max_tokens  = 256,
                )
                llm_text = response.choices[0].message.content or ""
                history.append({"role": "assistant", "content": llm_text})
            except Exception as e:
                last_error = f"LLM error: {e}"
                print(f"[DEBUG] LLM call failed: {e}", flush=True)

            action = parse_action(llm_text, task_id, extracted_fields, step_num)

            # Track which fields have been handled
            field_done = action.get("field") or action.get("field_name", "")
            if field_done and field_done not in extracted_fields:
                extracted_fields.append(field_done)

            action_str = json.dumps(action)

            # Step environment
            try:
                obs        = call_step(session_id, action)
                reward     = float(obs.get("reward", 0.0))
                done       = obs.get("done", False)
                msg        = obs.get("message", "")
                if reward < 0 and msg:
                    last_error = msg
            except Exception as e:
                reward     = 0.0
                done       = True
                last_error = str(e)

            rewards.append(reward)
            error_str = last_error if last_error else "null"

            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True,
            )

            if done:
                score   = float(obs.get("score", 0.0))
                success = score >= pass_thresh
                break

        # Hit max steps without done — call grader
        if not obs.get("done", False) and session_id:
            grader = call_grader(session_id)
            score   = float(grader.get("score", 0.0))
            success = grader.get("passed", False)

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        rewards = rewards or [0.0]
        score   = 0.0
        success = False

    if not rewards:
        rewards = [0.0]

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"task_id": task_id, "score": score, "success": success,
            "steps": len(rewards), "rewards": rewards}


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    print(f"# FinDocAgent Baseline | model={MODEL_NAME} | env={ENV_URL}", flush=True)
    all_results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        all_results.append(result)
        print(f"# Task {task_id}: score={result['score']:.2f}", flush=True)

    overall = sum(r["score"] for r in all_results) / len(all_results)
    print(f"# Overall score: {overall:.2f}", flush=True)


if __name__ == "__main__":
    main()