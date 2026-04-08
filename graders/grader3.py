"""
FinDocAgent-Env — Grader 3: 3-Way Reconciler (Hard)

Scores the agent's reconciliation against ground truth discrepancies + decision.
Score range: 0.0 – 1.0

Scoring breakdown:
    Decision accuracy  (35%) — approve / reject / escalate correct?
    Discrepancy recall (35%) — how many real discrepancies did agent find?
    Discrepancy precision (20%) — of discrepancies flagged, how many were real?
    Justification quality (10%) — did agent provide a meaningful reason?

Partial credit:
    - Correct discrepancy type, wrong item  → 0.4
    - Correct item, wrong discrepancy type  → 0.3
    - Exact match (type + item)             → 1.0
    - Decision: adjacent decision           → 0.4 (escalate vs reject)
    - Decision: opposite                    → 0.0 (approve vs reject)
"""

import re
from typing import Any


# ─────────────────────────────────────────────
# Aliases
# ─────────────────────────────────────────────

DISCREPANCY_ALIASES = {
    "qty_mismatch":          "quantity_mismatch",
    "quantity_error":        "quantity_mismatch",
    "quantity_diff":         "quantity_mismatch",
    "wrong_quantity":        "quantity_mismatch",
    "price_mismatch":        "price_deviation",
    "price_error":           "price_deviation",
    "price_diff":            "price_deviation",
    "wrong_price":           "price_deviation",
    "unit_price_mismatch":   "price_deviation",
    "missing_line":          "missing_line_item",
    "missing_item":          "missing_line_item",
    "item_missing":          "missing_line_item",
    "removed_item":          "missing_line_item",
    "short_recv":            "short_delivery",
    "under_delivery":        "short_delivery",
    "partial_delivery":      "short_delivery",
    "short_received":        "short_delivery",
}

DECISION_ALIASES = {
    "approved":   "approve",
    "accept":     "approve",
    "accepted":   "approve",
    "ok":         "approve",
    "clear":      "approve",
    "rejected":   "reject",
    "decline":    "reject",
    "declined":   "reject",
    "deny":       "reject",
    "denied":     "reject",
    "escalated":  "escalate",
    "review":     "escalate",
    "hold":       "escalate",
    "pending":    "escalate",
    "flag":       "escalate",
}

# Decision adjacency for partial credit
DECISION_ADJACENCY = {
    ("approve",  "escalate"): 0.3,
    ("escalate", "approve"):  0.3,
    ("escalate", "reject"):   0.4,
    ("reject",   "escalate"): 0.4,
    ("approve",  "reject"):   0.0,
    ("reject",   "approve"):  0.0,
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_disc_type(dtype: str) -> str:
    d = str(dtype).strip().lower().replace(" ", "_").replace("-", "_")
    return DISCREPANCY_ALIASES.get(d, d)


def _normalize_decision(dec: str) -> str:
    d = str(dec).strip().lower()
    return DECISION_ALIASES.get(d, d)


def _normalize_item(item: str) -> str:
    return str(item).strip().lower()


def _item_match(agent_item: str, truth_item: str) -> bool:
    """Fuzzy item name match — one contains the other."""
    a = _normalize_item(agent_item)
    t = _normalize_item(truth_item)
    return a == t or a in t or t in a


def _build_disc_index(discrepancies: list) -> list:
    """Normalize ground truth discrepancies for matching."""
    normed = []
    for d in discrepancies:
        normed.append({
            "type":     _normalize_disc_type(d.get("type", "")),
            "item":     _normalize_item(d.get("item", "")),
            "severity": d.get("severity", "medium"),
            "raw":      d,
        })
    return normed


# ─────────────────────────────────────────────
# Per-discrepancy scoring
# ─────────────────────────────────────────────

def _score_single_discrepancy(agent_disc: dict, truth_discs: list, matched: set) -> dict:
    """
    Score one agent-reported discrepancy.

    Returns match result and score.
    matched: set of truth indices already matched (avoid double-counting).
    """
    a_type = _normalize_disc_type(agent_disc.get(
        "type", agent_disc.get("discrepancy_type", "")))
    a_item = _normalize_item(agent_disc.get(
        "item", agent_disc.get("field", "")))

    best_score = 0.0
    best_idx = -1
    best_detail = ""

    for i, td in enumerate(truth_discs):
        if i in matched:
            continue

        type_match = (a_type == td["type"])
        item_match = _item_match(a_item, td["item"]) if a_item else False

        if type_match and item_match:
            score = 1.0
            detail = f"✅ Exact: {a_type} on '{td['item']}'"
            if score > best_score:
                best_score = score
                best_idx = i
                best_detail = detail

        elif type_match and not item_match:
            score = 0.4
            detail = f"⚠️  Right type ({a_type}), wrong item (got '{a_item}', truth '{td['item']}')"
            if score > best_score:
                best_score = score
                best_idx = i
                best_detail = detail

        elif item_match and not type_match:
            score = 0.3
            detail = f"⚠️  Right item ('{td['item']}'), wrong type (got '{a_type}', truth '{td['type']}')"
            if score > best_score:
                best_score = score
                best_idx = i
                best_detail = detail

    if best_score == 0.0:
        return {
            "score":      0.0,
            "match_idx": -1,
            "match_type": "false_positive",
            "detail":     f"❌ False positive: {a_type} on '{a_item}'",
        }

    return {
        "score":      best_score,
        "match_idx":  best_idx,
        "match_type": "exact" if best_score == 1.0 else "partial",
        "detail":     best_detail,
    }


# ─────────────────────────────────────────────
# Decision scoring
# ─────────────────────────────────────────────

def _score_decision(agent_decision: str, truth_decision: str, reason: str) -> dict:
    """Score the reconciliation decision + justification."""
    a_dec = _normalize_decision(agent_decision)
    t_dec = _normalize_decision(truth_decision)

    if a_dec == t_dec:
        dec_score = 1.0
        dec_detail = f"✅ Correct decision: {a_dec}"
    else:
        partial = DECISION_ADJACENCY.get((a_dec, t_dec), 0.0)
        dec_score = partial
        dec_detail = (
            f"{'⚠️' if partial > 0 else '❌'} "
            f"Wrong decision: got '{a_dec}', truth '{t_dec}' "
            f"(partial credit: {partial})"
        )

    # Justification: basic quality check
    reason_clean = str(reason).strip().lower()
    if len(reason_clean) >= 20:
        just_score = 1.0
    elif len(reason_clean) >= 8:
        just_score = 0.5
    else:
        just_score = 0.0

    return {
        "decision_score":       dec_score,
        "justification_score":  just_score,
        "decision_detail":      dec_detail,
        "agent_decision":       a_dec,
        "truth_decision":       t_dec,
    }


# ─────────────────────────────────────────────
# Main Grader
# ─────────────────────────────────────────────

def grade(
    discrepancies_found: list,
    reconcile_decision: str,
    reconcile_reason: str,
    ground_truth: dict,
) -> dict:
    """
    Grade the agent's 3-way reconciliation.

    Args:
        discrepancies_found : list of dicts from agent MATCH actions
                              Each: {type, item, po_value, inv_value, grn_value}
        reconcile_decision  : str — "approve" | "reject" | "escalate"
        reconcile_reason    : str — justification text
        ground_truth        : dict from generator with keys:
                              discrepancies, n_discrepancies, decision, decision_reason

    Returns:
        {
            score                : float (0.0–1.0)
            decision_score       : float
            precision            : float
            recall               : float
            justification_score  : float
            true_positives       : int
            false_positives      : int
            false_negatives      : int
            disc_details         : list
            missed_discrepancies : list
            feedback             : str
            passed               : bool (score >= 0.55)
        }
    """
    truth_discs = ground_truth.get("discrepancies", [])
    truth_dec = ground_truth.get("decision", "approve")
    n_truth = len(truth_discs)
    normed_truth = _build_disc_index(truth_discs)

    matched_indices = set()
    disc_details = []
    tp_exact = 0
    tp_partial = 0.0
    fp = 0

    for agent_disc in discrepancies_found:
        result = _score_single_discrepancy(
            agent_disc, normed_truth, matched_indices)
        disc_details.append({**agent_disc, **result})

        if result["match_idx"] >= 0:
            matched_indices.add(result["match_idx"])
            if result["match_type"] == "exact":
                tp_exact += 1
            else:
                tp_partial += result["score"]
        else:
            fp += 1

    fn = n_truth - len(matched_indices)

    # Effective TP = exact + fractional partials
    effective_tp = tp_exact + tp_partial

    # Precision
    total_agent = len(discrepancies_found)
    precision = effective_tp / total_agent if total_agent > 0 else 0.0
    precision = min(1.0, precision)

    # Recall
    recall = len(matched_indices) / n_truth if n_truth > 0 else 1.0
    recall = min(1.0, recall)

    # Decision + justification
    dec_result = _score_decision(
        reconcile_decision, truth_dec, reconcile_reason)

    # Final weighted score
    score = (
        0.35 * dec_result["decision_score"] +
        0.35 * recall +
        0.20 * precision +
        0.10 * dec_result["justification_score"]
    )
    score = round(min(score, 1.0), 4)

    # Missed discrepancies
    all_indices = set(range(len(normed_truth)))
    missed_indices = all_indices - matched_indices
    missed_discs = [truth_discs[i] for i in sorted(missed_indices)]

    passed = score >= 0.55

    feedback = (
        f"Score: {score:.2f} | "
        f"Decision: {dec_result['decision_score']:.2f} "
        f"({dec_result['agent_decision']} vs {dec_result['truth_decision']}) | "
        f"Recall: {recall:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Justification: {dec_result['justification_score']:.2f} | "
        f"TP: {len(matched_indices)}/{n_truth} | FP: {fp} | FN: {fn} | "
        f"{'✅ PASSED' if passed else '❌ FAILED (need ≥ 0.55)'}"
    )

    return {
        "score":                score,
        "decision_score":       dec_result["decision_score"],
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "justification_score":  dec_result["justification_score"],
        "true_positives":       len(matched_indices),
        "false_positives":      fp,
        "false_negatives":      fn,
        "disc_details":         disc_details,
        "missed_discrepancies": missed_discs,
        "decision_detail":      dec_result["decision_detail"],
        "feedback":             feedback,
        "passed":               passed,
    }


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generator import generate_task3_episode

    ep = generate_task3_episode(seed=42)
    gt = ep["ground_truth"]

    print(f"Ground truth ({gt['n_discrepancies']} discrepancies):")
    for d in gt["discrepancies"]:
        print(f"  [{d['severity'].upper()}] {d['type']} on '{d['item']}'")
    print(f"  Decision: {gt['decision']} — {gt['decision_reason']}")

    # Perfect agent
    perfect_discs = [
        {
            "type": d["type"],
            "item": d["item"],
        }
        for d in gt["discrepancies"]
    ]

    # Partial agent — finds one correctly, one wrong type, misses one
    partial_discs = [
        {"type": gt["discrepancies"][0]["type"],
            "item": gt["discrepancies"][0]["item"]},
        {"type": "duplicate_invoice",            "item": "some random item"},
    ] if gt["n_discrepancies"] > 0 else []

    # Zero agent — finds nothing
    zero_discs = []

    print("\n=== PERFECT AGENT ===")
    r1 = grade(perfect_discs, gt["decision"], gt["decision_reason"], gt)
    print(r1["feedback"])

    print("\n=== PARTIAL AGENT ===")
    r2 = grade(
        partial_discs,
        "escalate",
        "Found some issues need review",
        gt
    )
    print(r2["feedback"])
    print(f"  Decision detail: {r2['decision_detail']}")

    print("\n=== ZERO AGENT ===")
    r3 = grade(zero_discs, "approve", "Looks fine", gt)
    print(r3["feedback"])
    print(f"  Decision detail: {r3['decision_detail']}")
