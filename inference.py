"""
inference.py - FinDocAgent-Env Baseline Inference Script
STDOUT FORMAT:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""
import os, json, re, sys, requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")   # 100k/day free, separate quota
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
ENV_URL      = os.getenv("ENV_URL", "https://niyasingh-findocagent-env.hf.space").rstrip("/")
BENCHMARK    = "findocagent"

TASKS = [
    "task1_invoice_parser",
    "task2_anomaly_detector",
    "task3_three_way_reconciler",
]

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

# ── Task1 field order ───────────────────────────────────────────────────────────
TASK1_FIELDS = [
    "vendor_name", "buyer_name", "invoice_number", "po_reference",
    "issue_date", "due_date", "subtotal", "tax_amount",
    "total_amount", "currency", "n_line_items",
]

# ── System prompts ──────────────────────────────────────────────────────────────
PROMPT_TASK1 = """You are a financial document extraction agent.
Extract fields from the invoice ONE AT A TIME.
Respond ONLY with a single raw JSON object. No markdown, no explanation.

Format: {"action_type": "extract", "field": "<field_name>", "value": "<value_as_string>"}

IMPORTANT: value must ALWAYS be a string, even for numbers. Write "32671.66" not 32671.66.

Fields to extract in order:
vendor_name, buyer_name, invoice_number, po_reference, issue_date,
due_date, subtotal, tax_amount, total_amount, currency, n_line_items

For n_line_items: count the line items and return as string e.g. "3"
For currency: return the code e.g. "INR"
For amounts: return numeric string e.g. "36914.67"

The ALREADY_EXTRACTED section shows what is done. Extract the NEXT field not yet extracted.
When ALL 11 fields are extracted: {"action_type": "skip", "field": "done"}"""

PROMPT_TASK2 = """You are a financial compliance auditor.
Respond ONLY with a single raw JSON object. No markdown, no explanation.

Review invoices for these violation types ONLY:
- missing_po_reference: po_reference is empty/missing
- tax_calculation_error: tax_amount != round(subtotal * 0.18, 2) with >1 rupee difference
- date_in_future: issue_date is after 2026-04-10
- duplicate_invoice: same invoice_number appears twice in batch
- vendor_not_approved: vendor not on any known approved list
- amount_exceeds_po_limit: total_amount > 500000 INR AND po_reference is missing

STRICT RULES:
1. Check ALREADY_FLAGGED list. NEVER flag the same (doc_ref + field) combination twice.
2. Flag ONE anomaly per step.
3. After flagging ALL real anomalies (or if none left), send the reconcile action ONCE.
4. Do NOT flag an invoice just because its amount is high if it has a valid PO.

To flag: {"action_type": "flag", "field": "<violation_type>", "value": "<brief description>",
          "doc_ref": "<invoice_id>", "severity": "high", "reason": "<explanation>"}

To finish: {"action_type": "reconcile", "field": "done", "value": "",
            "decision": "escalate", "reason": "Anomaly review complete"}"""

PROMPT_TASK3 = """You are a 3-way PO/Invoice/GRN reconciliation agent.
Respond ONLY with a single raw JSON object. No markdown, no explanation.

Compare PO, Invoice, and GRN. Report discrepancies ONE AT A TIME.
Discrepancy types: quantity_mismatch, price_deviation, missing_line_item, short_delivery

To report: {"action_type": "match", "field": "<discrepancy_type>", "value": "<item name>",
            "doc_ref": "invoice", "reason": "type:<discrepancy_type>"}

After ALL discrepancies reported (check ALREADY_MATCHED), finalize:
{"action_type": "reconcile", "field": "decision", "value": "<approve|reject|escalate>",
 "decision": "<approve|reject|escalate>", "reason": "<justification>"}

approve = all match within tolerance, reject = fraud/critical issue, escalate = needs review
NEVER repeat a discrepancy already in ALREADY_MATCHED."""


# ── Env helpers ─────────────────────────────────────────────────────────────────
def env_reset(task_id):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(session_id, action):
    r = requests.post(f"{ENV_URL}/step",
                      json={"session_id": session_id, **action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Document formatter ──────────────────────────────────────────────────────────
def format_document(obs: dict) -> str:
    doc = obs.get("document", {})
    if not isinstance(doc, dict):
        return str(doc)[:2000]

    # Task1: plain text invoice
    if "content" in doc:
        return doc["content"][:2000]

    # Task2: batch of invoices
    if "invoices" in doc:
        lines = ["INVOICE BATCH:"]
        for i, inv in enumerate(doc["invoices"]):
            inv_id = inv.get("invoice_id", f"INV-BATCH-{i+1:03d}")
            sub = float(inv.get("subtotal", 0) or 0)
            tax = float(inv.get("tax_amount", 0) or 0)
            expected_tax = round(sub * 0.18, 2)
            lines.append(f"\n[{inv_id}]")
            lines.append(f"  Vendor:     {inv.get('vendor_name','')}")
            lines.append(f"  Invoice No: {inv.get('invoice_number','')}")
            lines.append(f"  PO Ref:     {inv.get('po_reference','') or 'MISSING'}")
            lines.append(f"  Issue Date: {inv.get('issue_date','')}")
            lines.append(f"  Subtotal:   {sub}")
            lines.append(f"  Tax(actual):{tax}  Tax(expected@18%):{expected_tax}  Match:{abs(tax-expected_tax)<=1}")
            lines.append(f"  Total:      {inv.get('total_amount','')} {inv.get('currency','INR')}")
        return "\n".join(lines)[:3500]

    # Task3: PO / Invoice / GRN
    if any(k in doc for k in ("po", "invoice", "grn")):
        lines = []
        for name in ("po", "invoice", "grn"):
            if name in doc:
                lines.append(f"\n=== {name.upper()} ===")
                d = doc[name]
                lines.append(json.dumps(d, indent=2) if isinstance(d, dict) else str(d))
        return "\n".join(lines)[:3500]

    return json.dumps(doc, indent=2)[:2000]


# ── Prompt builder ──────────────────────────────────────────────────────────────
def build_user_msg(obs: dict, done_items: list) -> str:
    task_id   = obs.get("task_id", "")
    doc_text  = format_document(obs)
    extracted = obs.get("extracted") or {}
    flags     = obs.get("flags") or []
    matches   = obs.get("matches") or []

    msg = f"Step {obs.get('step_count',0)}/{obs.get('max_steps',20)}\n\n"
    msg += f"DOCUMENT:\n{doc_text}\n"

    if "task1" in task_id:
        done = list(extracted.keys())
        remaining = [f for f in TASK1_FIELDS if f not in done]
        msg += f"\nALREADY_EXTRACTED: {json.dumps(extracted)}\n"
        msg += f"REMAINING_FIELDS: {remaining}\n"
        if remaining:
            msg += f"NEXT FIELD TO EXTRACT: {remaining[0]}\n"
        else:
            msg += "ALL FIELDS EXTRACTED. Send skip action.\n"

    elif "task2" in task_id:
        already = [(f.get("doc_ref",""), f.get("field","")) for f in flags] if isinstance(flags, list) else []
        msg += f"\nALREADY_FLAGGED ({len(already)} flags): {json.dumps(already)}\n"
        msg += "DO NOT re-flag any (doc_ref, field) pair already in ALREADY_FLAGGED.\n"
        if already:
            msg += "Check if there are MORE unflagged anomalies. If none remain, send reconcile.\n"

    elif "task3" in task_id:
        already = [(m.get("field",""), m.get("value","")) for m in matches] if isinstance(matches, list) else []
        msg += f"\nALREADY_MATCHED ({len(already)}): {json.dumps(already)}\n"
        msg += "DO NOT repeat any discrepancy already in ALREADY_MATCHED.\n"
        if already:
            msg += "If no more discrepancies remain, send the reconcile action.\n"

    msg += f"\nVALID ACTIONS: {obs.get('valid_actions', [])}\n"
    msg += "\nRespond with ONE JSON action object only."
    return msg


# ── LLM + parse ─────────────────────────────────────────────────────────────────
def call_llm(client, system_prompt: str, user_msg: str) -> str:
    """Single-turn — no growing history, saves tokens."""
    try:
        r = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.0,
            max_tokens  = 150,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return ""


def parse_action(text: str) -> dict:
    if not text:
        return {"action_type": "skip"}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "action_type" in obj:
                if "value" in obj and not isinstance(obj["value"], str):
                    obj["value"] = str(obj["value"])
                return obj
        except Exception:
            pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            if "value" in obj and not isinstance(obj["value"], str):
                obj["value"] = str(obj["value"])
            return obj
    except Exception:
        pass
    return {"action_type": "skip"}


# ── Logging ─────────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    act = json.dumps(action, separators=(",", ":")) if isinstance(action, dict) else str(action)
    act = act.replace("\n", " ").replace("\r", "")
    err = error if error else "null"
    print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)


# ── Episode ─────────────────────────────────────────────────────────────────────
def run_episode(client, task_id: str) -> dict:
    max_steps   = TASK_MAX_STEPS[task_id]
    pass_thresh = PASS_THRESHOLD[task_id]
    rewards, score, success, steps_done = [], 0.0, False, 0

    if "task1" in task_id:
        system_prompt = PROMPT_TASK1
    elif "task2" in task_id:
        system_prompt = PROMPT_TASK2
    else:
        system_prompt = PROMPT_TASK3

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        obs        = env_reset(task_id)
        session_id = obs["session_id"]
        done_items: list = []

        last_actions = set()
        negative_streak = 0

        for step_num in range(1, max_steps + 1):
            if obs.get("done", False):
                score = float(obs.get("score", 0.0))
                break

            user_msg = build_user_msg(obs, done_items)
            llm_text = call_llm(client, system_prompt, user_msg)
            action   = parse_action(llm_text)
            # ── 🛡️ TASK ACTION FILTER ─────────────────────────────
            if "task2" in task_id:
                if action.get("action_type") not in ["flag", "reconcile"]:
                    action = {"action_type": "reconcile", "field": "done", "value": "",
                            "decision": "escalate", "reason": "Invalid action corrected"}

            elif "task3" in task_id:
                if action.get("action_type") not in ["match", "reconcile"]:
                    action = {"action_type": "match", "field": "quantity_mismatch",
                            "value": "unknown", "doc_ref": "invoice",
                            "reason": "type:quantity_mismatch"}

            # ── 🔒 DUPLICATE ACTION GUARD ───────────────────────────────
            action_key = json.dumps({
                "type": action.get("action_type"),
                "field": action.get("field"),
                "doc": action.get("doc_ref", "")
            }, sort_keys=True)

            if action_key in last_actions:
                if "task3" in task_id:
                    action = {
                        "action_type": "reconcile",
                        "field": "decision",
                        "value": "escalate",
                        "decision": "escalate",
                        "reason": "No new discrepancies left"
                    }
                elif "task2" in task_id:
                    action = {
                        "action_type": "reconcile",
                        "field": "done",
                        "value": "",
                        "decision": "escalate",
                        "reason": "No new anomalies left"
                    }
            else:
                last_actions.add(action_key)
            # Track locally
            field   = action.get("field") or action.get("field_name", "")
            doc_ref = action.get("doc_ref", "")
            key     = f"{doc_ref}:{field}" if doc_ref else field
            if key and key not in done_items:
                done_items.append(key)
            if "task3" in task_id and len(done_items) >= 2:
                action = {
                    "action_type": "reconcile",
                    "field": "decision",
                    "value": "escalate",
                    "decision": "escalate",
                    "reason": "Sufficient discrepancies identified"
                }

            last_error = None
            try:
                obs    = env_step(session_id, action)
                reward = float(obs.get("reward", 0.0))
                done   = bool(obs.get("done", False))

                # # If we already found multiple unique discrepancies, consider finishing
                # if "task3" in task_id and len(done_items) >= 3:
                #     action = {
                #         "action_type": "reconcile",
                #         "field": "decision",
                #         "value": "escalate",
                #         "decision": "escalate",
                #         "reason": "Multiple discrepancies found, reconciliation complete"
    

                # ── ⚠️ NEGATIVE REWARD HANDLING ─────────────────────────
                if reward < 0:
                    negative_streak += 1
                    last_error = obs.get("message", None)
                else:
                    negative_streak = 0

            except Exception as e:
                reward, done, last_error = 0.0, True, str(e)[:120]
                obs = {"done": True, "score": 0.0}

            rewards.append(reward)
            steps_done = step_num
            log_step(step=step_num, action=action, reward=reward, done=done, error=last_error)

            # ── 🛑 EARLY STOP IF STUCK ─────────────────────────────────
            if negative_streak >= 3:
                if "task2" in task_id:
                    break  # still valid for anomaly detection
                else:
                    negative_streak = 0  # reset, keep going for task3

            if done:
                score = float(obs.get("score", 0.0))
                break

        if not rewards:
            rewards = [0.0]
        score   = max(0.0, min(1.0, score))
        success = score >= pass_thresh

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", file=sys.stderr, flush=True)
        if not rewards:
            rewards = [0.0]
        log_step(step=max(steps_done, 1), action={"action_type": "skip"},
                 reward=0.0, done=True, error=str(e)[:120])

    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print(f"# FinDocAgent Baseline | model={MODEL_NAME} | env={ENV_URL}", flush=True)
    client  = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    results = []
    for task_id in TASKS:
        try:
            result = run_episode(client, task_id)
        except Exception as e:
            print(f"[DEBUG] Task failed: {e}", file=sys.stderr, flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
            result = {"task_id": task_id, "score": 0.0, "success": False}
        results.append(result)

    overall = sum(r["score"] for r in results) / len(results)
    print(f"# overall_score={overall:.2f}", flush=True)


if __name__ == "__main__":
    main()