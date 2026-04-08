"""
FinDocAgent-Env — Grader 2: Anomaly Detector (Medium)

Scores the agent's raised flags against ground truth anomalies.
Score range: 0.0 – 1.0

Scoring breakdown:
    Precision  (40%) — of flags raised, how many were real anomalies?
    Recall     (40%) — of real anomalies, how many did agent catch?
    Severity   (20%) — did agent correctly identify severity level?

Partial credit:
    - Correct invoice_id + wrong anomaly_type  → 0.3 per flag
    - Correct anomaly_type + wrong invoice_id  → 0.4 per flag
    - Both correct                             → 1.0 per flag
    - Correct severity bonus                   → +0.2 on top
"""

import re
from typing import Any


# ─────────────────────────────────────────────
# Anomaly type aliases
# ─────────────────────────────────────────────

ANOMALY_ALIASES = {
    "duplicate":              "duplicate_invoice",
    "dup_invoice":            "duplicate_invoice",
    "duplicate_inv":          "duplicate_invoice",
    "over_limit":             "amount_exceeds_po_limit",
    "exceeds_limit":          "amount_exceeds_po_limit",
    "amount_over":            "amount_exceeds_po_limit",
    "high_amount":            "amount_exceeds_po_limit",
    "no_po":                  "missing_po_reference",
    "missing_po":             "missing_po_reference",
    "no_po_ref":              "missing_po_reference",
    "po_missing":             "missing_po_reference",
    "unapproved_vendor":      "vendor_not_approved",
    "unknown_vendor":         "vendor_not_approved",
    "invalid_vendor":         "vendor_not_approved",
    "future_date":            "date_in_future",
    "invalid_date":           "date_in_future",
    "date_error":             "date_in_future",
    "wrong_tax":              "tax_calculation_error",
    "tax_error":              "tax_calculation_error",
    "incorrect_tax":          "tax_calculation_error",
    "tax_mismatch":           "tax_calculation_error",
    "qty_mismatch":           "quantity_mismatch",
    "quantity_error":         "quantity_mismatch",
    "price_error":            "price_deviation",
    "price_mismatch":         "price_deviation",
}

SEVERITY_ALIASES = {
    "lo":       "low",
    "l":        "low",
    "med":      "medium",
    "m":        "medium",
    "hi":       "high",
    "h":        "high",
    "crit":     "critical",
    "c":        "critical",
    "blocker":  "critical",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_anomaly_type(atype: str) -> str:
    a = atype.strip().lower().replace(" ", "_").replace("-", "_")
    return ANOMALY_ALIASES.get(a, a)


def _normalize_severity(sev: str) -> str:
    s = sev.strip().lower()
    return SEVERITY_ALIASES.get(s, s)


def _normalize_invoice_id(inv_id: str) -> str:
    return str(inv_id).strip().upper()


def _build_truth_index(anomalies_truth: list) -> dict:
    """
    Build lookup: {invoice_id: {anomaly_type: truth_record}}
    """
    index = {}
    for a in anomalies_truth:
        inv_id = _normalize_invoice_id(a["invoice_id"])
        atype = _normalize_anomaly_type(a["anomaly_type"])
        if inv_id not in index:
            index[inv_id] = {}
        index[inv_id][atype] = a
    return index


# ─────────────────────────────────────────────
# Per-flag scoring
# ─────────────────────────────────────────────

def _score_single_flag(flag: dict, truth_index: dict) -> dict:
    """
    Score one agent flag against truth index.

    Returns:
        {
            match_type : "exact" | "partial_type" | "partial_id" | "false_positive"
            raw_score  : float (before severity bonus)
            sev_bonus  : float
            total      : float
            detail     : str
        }
    """
    inv_id = _normalize_invoice_id(flag.get("invoice_id", ""))
    atype = _normalize_anomaly_type(flag.get("anomaly_type", ""))
    sev = _normalize_severity(flag.get("severity", ""))

    # Check exact match (correct invoice_id AND correct anomaly_type)
    if inv_id in truth_index and atype in truth_index[inv_id]:
        truth_rec = truth_index[inv_id][atype]
        truth_sev = _normalize_severity(truth_rec.get("severity", ""))
        sev_bonus = 0.2 if sev == truth_sev else 0.0
        return {
            "match_type": "exact",
            "raw_score":  1.0,
            "sev_bonus":  sev_bonus,
            "total":      min(1.0, 1.0 + sev_bonus),
            "detail":     f"✅ Exact match: {inv_id} / {atype}"
        }

    # Partial match: correct anomaly_type found in any invoice
    for tid, atypes in truth_index.items():
        if atype in atypes:
            return {
                "match_type": "partial_type",
                "raw_score":  0.4,
                "sev_bonus":  0.0,
                "total":      0.4,
                "detail":     f"⚠️  Right anomaly type ({atype}) but wrong invoice (got {inv_id}, truth {tid})"
            }

    # Partial match: correct invoice_id but wrong anomaly_type
    if inv_id in truth_index:
        return {
            "match_type": "partial_id",
            "raw_score":  0.3,
            "sev_bonus":  0.0,
            "total":      0.3,
            "detail":     f"⚠️  Right invoice ({inv_id}) but wrong anomaly type (got {atype})"
        }

    # False positive
    return {
        "match_type": "false_positive",
        "raw_score":  0.0,
        "sev_bonus":  0.0,
        "total":      0.0,
        "detail":     f"❌ False positive: {inv_id} / {atype}"
    }


# ─────────────────────────────────────────────
# Main Grader
# ─────────────────────────────────────────────

def grade(flags_raised: list, ground_truth: dict) -> dict:
    """
    Grade agent's anomaly flags against ground truth.

    Args:
        flags_raised  : list of dicts from agent FLAG actions
                        Each: {invoice_id, anomaly_type, severity, reason}
        ground_truth  : dict with key "anomalies" (list of truth records)

    Returns:
        {
            score            : float (0.0–1.0)
            precision        : float
            recall           : float
            severity_score   : float
            true_positives   : int
            false_positives  : int
            false_negatives  : int
            flag_details     : list  per-flag breakdown
            missed_anomalies : list  anomalies the agent missed
            feedback         : str
            passed           : bool  (score >= 0.60)
        }
    """
    anomalies_truth = ground_truth.get("anomalies", [])
    n_truth = len(anomalies_truth)

    if n_truth == 0:
        # Edge case: no anomalies in episode
        fp = len(flags_raised)
        score = max(0.0, 1.0 - fp * 0.2)
        return {
            "score":            round(score, 4),
            "precision":        0.0 if fp > 0 else 1.0,
            "recall":           1.0,
            "severity_score":   1.0,
            "true_positives":   0,
            "false_positives":  fp,
            "false_negatives":  0,
            "flag_details":     [],
            "missed_anomalies": [],
            "feedback":         f"No anomalies exist. Agent raised {fp} false flags.",
            "passed":           score >= 0.60,
        }

    truth_index = _build_truth_index(anomalies_truth)
    matched_keys = set()   # track which truth anomalies have been matched
    flag_details = []
    tp = 0
    fp = 0
    sev_scores = []

    for flag in flags_raised:
        result = _score_single_flag(flag, truth_index)
        flag_details.append({**flag, **result})

        if result["match_type"] == "exact":
            key = (
                _normalize_invoice_id(flag.get("invoice_id", "")),
                _normalize_anomaly_type(flag.get("anomaly_type", ""))
            )
            if key not in matched_keys:
                matched_keys.add(key)
                tp += 1
                sev_scores.append(
                    result["sev_bonus"] / 0.2 if result["sev_bonus"] > 0 else 0.0)
            else:
                # Duplicate flag — treat as false positive
                fp += 1
        elif result["match_type"] in ("partial_type", "partial_id"):
            tp += result["total"]   # fractional TP
        else:
            fp += 1

    fn = n_truth - len(matched_keys)   # missed anomalies

    # Precision = TP / (TP + FP)
    total_flags = len(flags_raised)
    precision = tp / total_flags if total_flags > 0 else 0.0
    precision = min(1.0, precision)

    # Recall = matched / total truth
    recall = len(matched_keys) / n_truth if n_truth > 0 else 1.0

    # Severity score = avg correct severity on TPs
    severity_score = (sum(sev_scores) / len(sev_scores)) if sev_scores else 0.0

    # Weighted final score
    score = (
        0.40 * precision +
        0.40 * recall +
        0.20 * severity_score
    )
    score = round(min(score, 1.0), 4)

    # Missed anomalies
    all_keys = {
        (_normalize_invoice_id(a["invoice_id"]),
         _normalize_anomaly_type(a["anomaly_type"]))
        for a in anomalies_truth
    }
    missed_keys = all_keys - matched_keys
    missed_anomalies = [
        a for a in anomalies_truth
        if (
            _normalize_invoice_id(a["invoice_id"]),
            _normalize_anomaly_type(a["anomaly_type"])
        ) in missed_keys
    ]

    passed = score >= 0.60

    feedback = (
        f"Score: {score:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"Severity: {severity_score:.2f} | "
        f"TP: {len(matched_keys)}/{n_truth} | "
        f"FP: {fp} | "
        f"FN: {fn} | "
        f"{'✅ PASSED' if passed else '❌ FAILED (need ≥ 0.60)'}"
    )

    return {
        "score":            score,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "severity_score":   round(severity_score, 4),
        "true_positives":   len(matched_keys),
        "false_positives":  fp,
        "false_negatives":  fn,
        "flag_details":     flag_details,
        "missed_anomalies": missed_anomalies,
        "feedback":         feedback,
        "passed":           passed,
    }


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generator import generate_task2_episode

    ep = generate_task2_episode(seed=42)
    gt = ep["ground_truth"]

    print(f"Ground truth anomalies ({len(gt['anomalies'])}):")
    for a in gt["anomalies"]:
        print(
            f"  [{a['severity'].upper()}] {a['invoice_id']} — {a['anomaly_type']}")

    # Perfect agent — flags exactly the right anomalies with correct severity
    perfect_flags = [
        {
            "invoice_id":   a["invoice_id"],
            "anomaly_type": a["anomaly_type"],
            "severity":     a["severity"],
            "reason":       a["description"],
        }
        for a in gt["anomalies"]
    ]

    # Partial agent — catches one, misses one, adds one false positive
    partial_flags = [
        {
            "invoice_id":   gt["anomalies"][0]["invoice_id"],
            "anomaly_type": gt["anomalies"][0]["anomaly_type"],
            "severity":     "low",      # wrong severity
            "reason":       "Looks suspicious",
        },
        {
            "invoice_id":   "INV-BATCH-001",
            "anomaly_type": "duplicate_invoice",
            "severity":     "high",
            "reason":       "False positive",
        },
    ]

    # Zero agent — flags nothing
    zero_flags = []

    print("\n=== PERFECT AGENT ===")
    r1 = grade(perfect_flags, gt)
    print(r1["feedback"])

    print("\n=== PARTIAL AGENT ===")
    r2 = grade(partial_flags, gt)
    print(r2["feedback"])

    print("\n=== ZERO AGENT ===")
    r3 = grade(zero_flags, gt)
    print(r3["feedback"])
