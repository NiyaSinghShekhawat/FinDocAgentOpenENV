"""
FinDocAgent-Env — Grader 1: Invoice Parser (Easy)

Scores the agent's extracted fields against ground truth.
Score range: 0.0 – 1.0

Scoring breakdown:
    - Each field has a weight
    - Partial credit for near-matches (amounts within 1%, dates same value diff format)
    - Total weighted score normalised to 0.0–1.0
"""

import re
from typing import Any
from datetime import datetime


# ─────────────────────────────────────────────
# Field weights (must sum to 1.0)
# ─────────────────────────────────────────────

FIELD_WEIGHTS = {
    "vendor_name":    0.12,
    "buyer_name":     0.08,
    "invoice_number": 0.12,
    "po_reference":   0.10,
    "issue_date":     0.08,
    "due_date":       0.08,
    "subtotal":       0.12,
    "tax_amount":     0.10,
    "total_amount":   0.14,
    "currency":       0.04,
    "n_line_items":   0.02,
}

# Aliases the agent might use for field names
FIELD_ALIASES = {
    "vendor":           "vendor_name",
    "supplier":         "vendor_name",
    "from":             "vendor_name",
    "buyer":            "buyer_name",
    "client":           "buyer_name",
    "to":               "buyer_name",
    "invoice_no":       "invoice_number",
    "invoice_num":      "invoice_number",
    "inv_number":       "invoice_number",
    "inv_no":           "invoice_number",
    "po_ref":           "po_reference",
    "po_number":        "po_reference",
    "purchase_order":   "po_reference",
    "date":             "issue_date",
    "invoice_date":     "issue_date",
    "payment_due":      "due_date",
    "due":              "due_date",
    "sub_total":        "subtotal",
    "net_amount":       "subtotal",
    "tax":              "tax_amount",
    "gst":              "tax_amount",
    "total":            "total_amount",
    "grand_total":      "total_amount",
    "amount_due":       "total_amount",
    "num_items":        "n_line_items",
    "item_count":       "n_line_items",
}

DATE_FORMATS = [
    "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y",
    "%d/%m/%Y", "%Y-%m-%d", "%d %B %Y",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_field(field: str) -> str:
    """Resolve field name aliases to canonical name."""
    f = field.strip().lower().replace(" ", "_").replace("-", "_")
    return FIELD_ALIASES.get(f, f)


def _parse_amount(value: Any) -> float | None:
    """Parse numeric value from string or number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[₹$,\s]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_date(value: str) -> datetime | None:
    """Try parsing date string in any known format."""
    if not isinstance(value, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _normalize_string(value: str) -> str:
    """Lowercase, strip, collapse spaces."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


# ─────────────────────────────────────────────
# Per-field comparison logic
# ─────────────────────────────────────────────

def _compare_field(field: str, extracted: Any, truth: Any) -> float:
    """
    Returns per-field score 0.0–1.0.
    Uses fuzzy matching appropriate to each field type.
    """
    if extracted is None or extracted == "" or extracted == "SKIP":
        return 0.0

    # Amount fields — allow 1% tolerance
    if field in ("subtotal", "tax_amount", "total_amount"):
        e = _parse_amount(extracted)
        t = _parse_amount(truth)
        if e is None or t is None:
            return 0.0
        if t == 0:
            return 1.0 if e == 0 else 0.0
        diff_pct = abs(e - t) / abs(t)
        if diff_pct <= 0.01:
            return 1.0
        elif diff_pct <= 0.05:
            return 0.5   # partial credit
        return 0.0

    # Date fields — normalize format before comparing
    if field in ("issue_date", "due_date"):
        e_date = _parse_date(str(extracted))
        t_date = _parse_date(str(truth))
        if e_date is None or t_date is None:
            # Fall back to string match
            return 1.0 if _normalize_string(extracted) == _normalize_string(truth) else 0.0
        return 1.0 if e_date.date() == t_date.date() else 0.0

    # Integer fields
    if field == "n_line_items":
        try:
            return 1.0 if int(extracted) == int(truth) else 0.0
        except (ValueError, TypeError):
            return 0.0

    # Currency
    if field == "currency":
        return 1.0 if str(extracted).strip().upper() == str(truth).strip().upper() else 0.0

    # String fields — normalized match + partial credit for substring
    e_norm = _normalize_string(extracted)
    t_norm = _normalize_string(truth)

    if e_norm == t_norm:
        return 1.0
    # Partial credit: one contains the other
    if e_norm in t_norm or t_norm in e_norm:
        return 0.5
    return 0.0


# ─────────────────────────────────────────────
# Main Grader
# ─────────────────────────────────────────────

def grade(extracted: dict, ground_truth: dict) -> dict:
    """
    Grade agent's extracted fields against ground truth.

    Args:
        extracted    : dict of {field: value} from agent's EXTRACT actions
        ground_truth : dict from generator

    Returns:
        {
            score          : float  (0.0–1.0)
            field_scores   : dict   per-field breakdown
            fields_correct : int
            fields_total   : int
            feedback       : str    human-readable summary
            passed         : bool   (score >= 0.7)
        }
    """
    field_scores = {}
    weighted_score = 0.0
    fields_correct = 0
    missed_fields = []
    wrong_fields = []

    # Normalize extracted keys
    normalized_extracted = {}
    for k, v in extracted.items():
        canon = _normalize_field(k)
        normalized_extracted[canon] = v

    for field, weight in FIELD_WEIGHTS.items():
        agent_value = normalized_extracted.get(field)
        truth_value = ground_truth.get(field)

        fs = _compare_field(field, agent_value, truth_value)
        field_scores[field] = {
            "score":     fs,
            "weight":    weight,
            "extracted": agent_value,
            "truth":     truth_value,
        }
        weighted_score += fs * weight

        if fs == 1.0:
            fields_correct += 1
        elif agent_value is None or agent_value == "":
            missed_fields.append(field)
        else:
            wrong_fields.append(field)

    score = round(min(weighted_score, 1.0), 4)
    passed = score >= 0.70

    feedback_parts = [
        f"Score: {score:.2f} | {fields_correct}/{len(FIELD_WEIGHTS)} fields correct"]
    if missed_fields:
        feedback_parts.append(f"Missed: {', '.join(missed_fields)}")
    if wrong_fields:
        feedback_parts.append(f"Wrong: {', '.join(wrong_fields)}")
    if passed:
        feedback_parts.append("✅ PASSED")
    else:
        feedback_parts.append("❌ FAILED (need ≥ 0.70)")

    return {
        "score":          score,
        "field_scores":   field_scores,
        "fields_correct": fields_correct,
        "fields_total":   len(FIELD_WEIGHTS),
        "missed_fields":  missed_fields,
        "wrong_fields":   wrong_fields,
        "feedback":       " | ".join(feedback_parts),
        "passed":         passed,
    }


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generator import generate_task1_episode

    ep = generate_task1_episode(seed=42)
    gt = ep["ground_truth"]

    # Simulate a near-perfect agent
    perfect = {
        "vendor_name":    gt["vendor_name"],
        "buyer_name":     gt["buyer_name"],
        "invoice_number": gt["invoice_number"],
        "po_reference":   gt["po_reference"],
        "issue_date":     gt["issue_date"],
        "due_date":       gt["due_date"],
        "subtotal":       gt["subtotal"],
        "tax_amount":     gt["tax_amount"],
        "total_amount":   gt["total_amount"],
        "currency":       gt["currency"],
        "n_line_items":   gt["n_line_items"],
    }

    # Simulate a partial agent (misses some fields, wrongs one)
    partial = {
        "vendor_name":    gt["vendor_name"],
        "invoice_number": gt["invoice_number"],
        "total_amount":   gt["total_amount"] + 500,  # wrong amount
        "currency":       "INR",
    }

    print("=== PERFECT AGENT ===")
    r1 = grade(perfect, gt)
    print(f"Score: {r1['score']} | Passed: {r1['passed']}")
    print(r1["feedback"])

    print("\n=== PARTIAL AGENT ===")
    r2 = grade(partial, gt)
    print(f"Score: {r2['score']} | Passed: {r2['passed']}")
    print(r2["feedback"])
