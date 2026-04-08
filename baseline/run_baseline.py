"""
FinDocAgent-Env — Baseline Inference Script

Runs GPT-4o-mini and GPT-4o against all 3 tasks and reports scores.
Reads API key from environment variable: OPENAI_API_KEY

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline/run_baseline.py

    # Specify model:
    python baseline/run_baseline.py --model gpt-4o-mini
    python baseline/run_baseline.py --model gpt-4o

    # Run N episodes per task:
    python baseline/run_baseline.py --episodes 3
"""

from models import FinDocAction
from server.findoc_environment import FinDocEnvironment
from openai import OpenAI
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

SEEDS = [42, 123, 7]   # reproducible seeds

TASK_SYSTEM_PROMPTS = {

    "task1_invoice_parser": """You are a financial document processing agent.
Your job is to extract structured fields from invoice documents.

You will receive an invoice (as JSON or text) and must extract these fields:
  vendor_name, buyer_name, invoice_number, po_reference, issue_date,
  due_date, subtotal, tax_amount, total_amount, currency, n_line_items

At each step, respond with a JSON object containing exactly:
{
  "action_type": "extract",
  "field": "<field_name>",
  "value": "<extracted_value>"
}

When you have extracted all fields, respond with:
{
  "action_type": "skip",
  "field": "done",
  "value": "all fields extracted"
}

Rules:
- Extract one field per step
- Use exact field names listed above
- For amounts, extract the numeric value only (e.g. "10356.59")
- For n_line_items, extract the count as a number (e.g. "2")
- If a field is not found, use action_type "skip"
- Do not hallucinate values not present in the document
""",

    "task2_anomaly_detector": """You are a financial compliance agent.
Your job is to review a batch of invoices and flag policy violations.

Policy rules:
1. No single invoice may exceed ₹5,00,000 without a PO reference
2. Duplicate invoice numbers indicate potential fraud
3. All invoices must reference a valid PO number
4. Vendors must be from the approved vendor list (Infosys, TCS, Wipro, HCL, Tech Mahindra, Zensar, Mphasis, Hexaware, Persistent, NIIT)
5. Invoice issue date must not be in the future
6. Tax must be exactly 18% GST of subtotal

At each step, respond with a JSON object:
To flag an anomaly:
{
  "action_type": "flag",
  "field": "<anomaly_type>",
  "value": "<description>",
  "doc_ref": "<invoice_id>",
  "severity": "<low|medium|high|critical>",
  "reason": "<detailed explanation>"
}

Anomaly types: duplicate_invoice, amount_exceeds_po_limit, missing_po_reference,
               vendor_not_approved, date_in_future, tax_calculation_error

When done reviewing, finalize with:
{
  "action_type": "reconcile",
  "field": "review_complete",
  "value": "done",
  "decision": "escalate",
  "reason": "<summary of findings>"
}
""",

    "task3_three_way_reconciler": """You are a senior accounts payable analyst.
Your job is to reconcile a Purchase Order (PO), Invoice, and Goods Receipt Note (GRN).

You will receive:
- document: the Invoice
- aux_documents.purchase_order: the PO
- aux_documents.grn: the Goods Receipt Note

Compare all three documents. For each discrepancy found, respond with:
{
  "action_type": "match",
  "field": "<discrepancy_type>",
  "value": "<affected_item_name>",
  "doc_ref": "<po|invoice|grn>",
  "reason": "type:<discrepancy_type>"
}

Discrepancy types: quantity_mismatch, price_deviation, missing_line_item, short_delivery

After identifying all discrepancies, make a final decision:
{
  "action_type": "reconcile",
  "field": "final_decision",
  "value": "<approve|reject|escalate>",
  "decision": "<approve|reject|escalate>",
  "reason": "<detailed justification>"
}

Decision rules:
- approve: all documents match perfectly
- escalate: discrepancies exist but not critical
- reject: critical discrepancy (missing items, major fraud signals)
"""
}


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

class LLMAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_action(self, task_id: str, observation: dict, history: list) -> dict:
        """Call LLM and parse action from response."""
        system_prompt = TASK_SYSTEM_PROMPTS[task_id]

        # Build user message with current observation
        user_msg = f"""Current observation:
Task: {observation.get('task_id')}
Step: {observation.get('step_count')} / {observation.get('max_steps')}
Valid actions: {observation.get('valid_actions')}
Last message: {observation.get('message', '')}

Document:
{json.dumps(observation.get('document', {}), indent=2)}
"""
        if observation.get("aux_documents"):
            user_msg += f"\nAuxiliary Documents:\n{json.dumps(observation['aux_documents'], indent=2)}"

        if observation.get("extracted"):
            user_msg += f"\nAlready extracted: {json.dumps(observation['extracted'], indent=2)}"

        if observation.get("flags"):
            user_msg += f"\nFlags raised: {json.dumps(observation['flags'], indent=2)}"

        if observation.get("matches"):
            user_msg += f"\nMatches recorded: {json.dumps(observation['matches'], indent=2)}"

        user_msg += "\n\nWhat is your next action? Respond with valid JSON only."

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            action = json.loads(content)
            history.append({"role": "assistant", "content": content})
            return action

        except json.JSONDecodeError:
            # Fallback action
            return {"action_type": "skip", "field": "parse_error", "value": ""}
        except Exception as e:
            print(f"    [API Error] {e}")
            return {"action_type": "skip", "field": "api_error", "value": ""}


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episode(agent: LLMAgent, task_id: str, seed: int, verbose: bool = True) -> dict:
    """Run one full episode and return results."""
    env = FinDocEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    history = []

    if verbose:
        print(f"    Seed {seed} | {task_id}")

    step = 0
    while not obs.done:
        obs_dict = {
            "task_id":       obs.task_id,
            "step_count":    obs.step_count,
            "max_steps":     obs.max_steps,
            "valid_actions": obs.valid_actions,
            "message":       obs.message,
            "document":      obs.document,
            "aux_documents": obs.aux_documents,
            "extracted":     obs.extracted,
            "flags":         obs.flags,
            "matches":       obs.matches,
        }

        action_dict = agent.get_action(task_id, obs_dict, history)

        action = FinDocAction(
            action_type=action_dict.get("action_type", "skip"),
            field=action_dict.get("field", ""),
            value=str(action_dict.get("value", "")),
            reason=action_dict.get("reason", ""),
            severity=action_dict.get("severity", ""),
            decision=action_dict.get("decision", ""),
            doc_ref=action_dict.get("doc_ref", ""),
        )

        obs = env.step(action)
        step += 1

        if verbose:
            print(
                f"      Step {step:02d} | {action.action_type:10s} | reward={obs.reward:+.2f} | {obs.message[:60]}")

    return {
        "seed":             seed,
        "task_id":          task_id,
        "score":            obs.score,
        "cumulative_reward": obs.cumulative_reward,
        "steps":            obs.step_count,
        "passed":           obs.score >= {"task1_invoice_parser": 0.70,
                                          "task2_anomaly_detector": 0.60,
                                          "task3_three_way_reconciler": 0.55}[task_id],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinDocAgent-Env Baseline")
    parser.add_argument("--model",    default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o"],
                        help="OpenAI model to use")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task (uses fixed seeds)")
    parser.add_argument("--task",     default="all",
                        choices=["all", "task1_invoice_parser",
                                 "task2_anomaly_detector",
                                 "task3_three_way_reconciler"],
                        help="Which task to run")
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set.")
        print("   export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    TASKS = (
        ["task1_invoice_parser", "task2_anomaly_detector",
            "task3_three_way_reconciler"]
        if args.task == "all"
        else [args.task]
    )
    seeds = SEEDS[:args.episodes]
    agent = LLMAgent(model=args.model)
    all_results = {}

    print(f"\n{'='*60}")
    print(f"  FinDocAgent-Env Baseline — {args.model}")
    print(f"  Tasks: {len(TASKS)} | Episodes/task: {len(seeds)}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_id in TASKS:
        print(f"▶ {task_id}")
        task_results = []

        for seed in seeds:
            result = run_episode(agent, task_id, seed, verbose=args.verbose)
            task_results.append(result)

        avg_score = sum(r["score"] for r in task_results) / len(task_results)
        avg_reward = sum(r["cumulative_reward"]
                         for r in task_results) / len(task_results)
        pass_rate = sum(
            1 for r in task_results if r["passed"]) / len(task_results)

        all_results[task_id] = {
            "avg_score":         round(avg_score, 4),
            "avg_reward":        round(avg_reward, 4),
            "pass_rate":         round(pass_rate, 4),
            "episodes":          task_results,
        }

        print(
            f"  ✅ Avg Score: {avg_score:.3f} | Pass Rate: {pass_rate:.0%} | Avg Reward: {avg_reward:.3f}\n")

    # Summary
    overall = sum(v["avg_score"]
                  for v in all_results.values()) / len(all_results)

    print(f"{'='*60}")
    print(f"  OVERALL SCORE: {overall:.4f}")
    print(f"{'='*60}")
    for task_id, res in all_results.items():
        status = "✅" if res["pass_rate"] >= 0.5 else "❌"
        print(
            f"  {status} {task_id:<35} score={res['avg_score']:.3f}  pass={res['pass_rate']:.0%}")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "model":         args.model,
        "timestamp":     datetime.now().isoformat(),
        "overall_score": round(overall, 4),
        "task_results":  all_results,
    }

    os.makedirs("baseline/results", exist_ok=True)
    fname = f"baseline/results/{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)
    print(f"📄 Results saved to: {fname}")

    return output


if __name__ == "__main__":
    main()
