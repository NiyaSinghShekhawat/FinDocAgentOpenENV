"""
FinDocAgent-Env — Synthetic Financial Document Generator
Generates realistic noisy financial documents for all 3 tasks.
Each document comes in two variants: structured JSON + noisy text.
"""

import random
import uuid
import json
from datetime import datetime, timedelta
from typing import Tuple


# ─────────────────────────────────────────────
# Master Data Pool
# ─────────────────────────────────────────────

VENDORS = [
    "Infosys Ltd", "Tata Consultancy Services", "Wipro Technologies",
    "HCL Systems", "Tech Mahindra", "Zensar Technologies",
    "Mphasis Corp", "Hexaware Ltd", "Persistent Systems", "NIIT Technologies"
]

BUYERS = [
    "Acme Enterprises Pvt Ltd", "GlobalTech Solutions",
    "Horizon Manufacturing Co", "BrightStar Industries",
    "NovaCorp Pvt Ltd", "Pinnacle Retail Ltd"
]

LINE_ITEM_POOL = [
    ("Software License - Annual",    5000,  50000),
    ("Cloud Hosting Services",       2000,  20000),
    ("IT Consulting - 10 hrs",       1500,  15000),
    ("Network Infrastructure Setup", 8000,  80000),
    ("Data Migration Services",      3000,  30000),
    ("Security Audit",               4000,  40000),
    ("Hardware Procurement",         6000,  60000),
    ("Support & Maintenance",        1000,  10000),
    ("Training & Onboarding",        2500,  25000),
    ("Custom Development - Module",  7000,  70000),
]

ANOMALY_TYPES = [
    "duplicate_invoice",
    "amount_exceeds_po_limit",
    "missing_po_reference",
    "vendor_not_approved",
    "date_in_future",
    "tax_calculation_error",
    "quantity_mismatch",
    "price_deviation",
]

NOISE_PATTERNS = [
    lambda s: s.upper(),
    lambda s: s.lower(),
    lambda s: s.replace(" ", "  "),
    lambda s: s + " ",
    lambda s: "  " + s,
    lambda s: s.replace("Ltd", "LIMITED").replace("Pvt", "PRIVATE"),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _rand_date(days_offset_range=(-30, 60)) -> str:
    offset = random.randint(*days_offset_range)
    d = datetime.today() + timedelta(days=offset)
    # Randomly pick one of 3 date formats (adds noise)
    fmt = random.choice(["%d-%m-%Y", "%Y/%m/%d", "%d %b %Y"])
    return d.strftime(fmt)


def _rand_invoice_number() -> str:
    prefix = random.choice(["INV", "INVC", "Invoice", "inv"])
    return f"{prefix}-{random.randint(1000, 9999)}-{random.randint(10,99)}"


def _rand_po_number() -> str:
    return f"PO-{random.randint(10000, 99999)}"


def _rand_grn_number() -> str:
    return f"GRN-{random.randint(10000, 99999)}"


def _apply_noise(value: str, noise_prob: float = 0.3) -> str:
    if random.random() < noise_prob:
        fn = random.choice(NOISE_PATTERNS)
        return fn(value)
    return value


def _compute_totals(line_items: list) -> Tuple[float, float, float]:
    subtotal = sum(item["quantity"] * item["unit_price"]
                   for item in line_items)
    tax = round(subtotal * 0.18, 2)   # 18% GST
    total = round(subtotal + tax, 2)
    return round(subtotal, 2), tax, total


# ─────────────────────────────────────────────
# TASK 1 — Invoice Parser Documents
# ─────────────────────────────────────────────

def generate_invoice(noisy: bool = False, seed: int = None) -> Tuple[dict, dict]:
    """
    Returns (document, ground_truth).
    document: what the agent sees (JSON with optional noise / noisy text string).
    ground_truth: correct field values the grader checks against.
    """
    if seed is not None:
        random.seed(seed)

    vendor = random.choice(VENDORS)
    buyer = random.choice(BUYERS)
    inv_no = _rand_invoice_number()
    po_ref = _rand_po_number()
    issue_date = _rand_date((-10, 0))
    due_date = _rand_date((15, 45))
    n_items = random.randint(2, 5)

    line_items = []
    for _ in range(n_items):
        name, lo, hi = random.choice(LINE_ITEM_POOL)
        qty = random.randint(1, 10)
        unit_price = round(random.uniform(lo / 10, hi / 10), 2)
        line_items.append({
            "description": name,
            "quantity":    qty,
            "unit_price":  unit_price,
            "amount":      round(qty * unit_price, 2)
        })

    subtotal, tax, total = _compute_totals(line_items)

    # Ground truth (always clean)
    ground_truth = {
        "vendor_name":    vendor,
        "buyer_name":     buyer,
        "invoice_number": inv_no,
        "po_reference":   po_ref,
        "issue_date":     issue_date,
        "due_date":       due_date,
        "line_items":     line_items,
        "subtotal":       subtotal,
        "tax_amount":     tax,
        "total_amount":   total,
        "currency":       "INR",
        "n_line_items":   n_items,
    }

    if noisy:
        # Return as noisy semi-structured text
        document = _render_noisy_invoice_text(ground_truth)
    else:
        # Return as structured JSON with minor noise on string fields
        document = {
            "vendor_name":    _apply_noise(vendor) if noisy else vendor,
            "buyer_name":     buyer,
            "invoice_number": inv_no,
            "po_reference":   po_ref,
            "issue_date":     issue_date,
            "due_date":       due_date,
            "line_items":     line_items,
            "subtotal":       subtotal,
            "tax_amount":     tax,
            "total_amount":   total,
            "currency":       "INR",
        }

    return document, ground_truth


def _render_noisy_invoice_text(gt: dict) -> dict:
    """Renders invoice as a messy text blob with OCR-like noise."""
    lines = [
        f"TAX INVOICE",
        f"",
        f"From: {_apply_noise(gt['vendor_name'], 0.5)}",
        f"To:   {gt['buyer_name']}",
        f"",
        f"Invoice No : {_apply_noise(gt['invoice_number'], 0.4)}",
        f"PO Ref     : {gt['po_reference']}",
        f"Date       : {gt['issue_date']}",
        f"Due Date   : {gt['due_date']}",
        f"",
        f"ITEMS:",
    ]
    for item in gt["line_items"]:
        noise_desc = _apply_noise(item["description"], 0.4)
        lines.append(
            f"  {noise_desc} | Qty: {item['quantity']} | "
            f"Rate: {item['unit_price']} | Amt: {item['amount']}"
        )
    lines += [
        f"",
        f"Subtotal  : {gt['subtotal']}",
        f"GST @18%  : {gt['tax_amount']}",
        f"TOTAL     : {gt['total_amount']} {gt['currency']}",
        f"",
        f"(Computer generated invoice)",
    ]
    return {
        "format": "text",
        "content": "\n".join(lines),
        "invoice_number": gt["invoice_number"],   # anchor field always clean
    }


# ─────────────────────────────────────────────
# TASK 2 — Anomaly Detection Batch
# ─────────────────────────────────────────────

def generate_invoice_batch(
    n: int = 8,
    n_anomalies: int = 3,
    seed: int = None
) -> Tuple[list, list]:
    """
    Returns (invoices, anomalies_ground_truth).
    invoices: list of invoice dicts.
    anomalies_ground_truth: list of {invoice_id, anomaly_type, description, severity}.
    """
    if seed is not None:
        random.seed(seed)

    invoices = []
    anomalies_truth = []
    seen_invoice_numbers = []

    for i in range(n):
        _, gt = generate_invoice(noisy=False)
        inv_id = f"INV-BATCH-{i+1:03d}"
        doc = {**gt, "invoice_id": inv_id}
        invoices.append(doc)
        seen_invoice_numbers.append(gt["invoice_number"])

    # Inject anomalies into random invoices
    anomaly_pool = random.sample(
        ANOMALY_TYPES, min(n_anomalies, len(ANOMALY_TYPES)))

    for i, atype in enumerate(anomaly_pool):
        target_idx = random.randint(0, n - 1)
        inv = invoices[target_idx]

        if atype == "duplicate_invoice":
            # Give this invoice the same number as another
            dup_num = seen_invoice_numbers[0]
            inv["invoice_number"] = dup_num
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "duplicate_invoice",
                "description":  f"Invoice number {dup_num} appears more than once",
                "severity":     "high",
            })

        elif atype == "amount_exceeds_po_limit":
            inv["total_amount"] = round(random.uniform(500001, 1000000), 2)
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "amount_exceeds_po_limit",
                "description":  f"Total amount {inv['total_amount']} exceeds ₹5,00,000 single-invoice limit",
                "severity":     "critical",
            })

        elif atype == "missing_po_reference":
            inv["po_reference"] = ""
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "missing_po_reference",
                "description":  "No PO reference number found on invoice",
                "severity":     "medium",
            })

        elif atype == "vendor_not_approved":
            inv["vendor_name"] = "Unknown Vendor Pvt Ltd"
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "vendor_not_approved",
                "description":  "Vendor 'Unknown Vendor Pvt Ltd' not in approved vendor list",
                "severity":     "high",
            })

        elif atype == "date_in_future":
            future_date = (
                datetime.today() + timedelta(days=random.randint(10, 90))).strftime("%d-%m-%Y")
            inv["issue_date"] = future_date
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "date_in_future",
                "description":  f"Invoice issue date {future_date} is in the future",
                "severity":     "medium",
            })

        elif atype == "tax_calculation_error":
            wrong_tax = round(inv["subtotal"] * 0.28, 2)   # wrong rate
            inv["tax_amount"] = wrong_tax
            inv["total_amount"] = round(inv["subtotal"] + wrong_tax, 2)
            anomalies_truth.append({
                "invoice_id":   inv["invoice_id"],
                "anomaly_type": "tax_calculation_error",
                "description":  f"Tax amount {wrong_tax} does not match 18% GST on subtotal {inv['subtotal']}",
                "severity":     "high",
            })

    return invoices, anomalies_truth


# ─────────────────────────────────────────────
# TASK 3 — 3-Way PO / Invoice / GRN Reconciler
# ─────────────────────────────────────────────

def generate_three_way_set(
    introduce_discrepancies: bool = True,
    seed: int = None
) -> Tuple[dict, dict, dict, dict]:
    """
    Returns (purchase_order, invoice, grn, ground_truth).

    ground_truth contains:
        - discrepancies: list of {field, doc_a, doc_b, expected, found, type}
        - decision: "approve" | "reject" | "escalate"
        - decision_reason: explanation
    """
    if seed is not None:
        random.seed(seed)

    vendor = random.choice(VENDORS)
    buyer = random.choice(BUYERS)
    po_num = _rand_po_number()
    inv_no = _rand_invoice_number()
    grn_no = _rand_grn_number()

    n_items = random.randint(2, 4)
    po_items = []
    for _ in range(n_items):
        name, lo, hi = random.choice(LINE_ITEM_POOL)
        qty = random.randint(2, 20)
        unit_price = round(random.uniform(lo / 10, hi / 10), 2)
        po_items.append({
            "description": name,
            "quantity":    qty,
            "unit_price":  unit_price,
            "amount":      round(qty * unit_price, 2),
        })

    po_subtotal, po_tax, po_total = _compute_totals(po_items)

    # Start invoice and GRN as copies of PO
    inv_items = [dict(i) for i in po_items]
    grn_items = [dict(i) for i in po_items]
    discrepancies = []

    if introduce_discrepancies:
        n_disc = random.randint(1, 3)
        disc_types = random.sample(
            ["qty_mismatch", "price_deviation",
                "missing_item", "grn_short_delivery"],
            min(n_disc, 4)
        )

        for dtype in disc_types:
            idx = random.randint(0, len(inv_items) - 1)

            if dtype == "qty_mismatch":
                original_qty = inv_items[idx]["quantity"]
                wrong_qty = original_qty + random.randint(1, 5)
                inv_items[idx]["quantity"] = wrong_qty
                inv_items[idx]["amount"] = round(
                    wrong_qty * inv_items[idx]["unit_price"], 2)
                discrepancies.append({
                    "type":     "quantity_mismatch",
                    "field":    "quantity",
                    "item":     inv_items[idx]["description"],
                    "po_value": original_qty,
                    "inv_value": wrong_qty,
                    "grn_value": original_qty,
                    "severity": "high",
                })

            elif dtype == "price_deviation":
                original_price = inv_items[idx]["unit_price"]
                deviation = round(
                    original_price * random.uniform(0.10, 0.25), 2)
                wrong_price = round(original_price + deviation, 2)
                inv_items[idx]["unit_price"] = wrong_price
                inv_items[idx]["amount"] = round(
                    inv_items[idx]["quantity"] * wrong_price, 2)
                discrepancies.append({
                    "type":      "price_deviation",
                    "field":     "unit_price",
                    "item":      inv_items[idx]["description"],
                    "po_value":  original_price,
                    "inv_value": wrong_price,
                    "grn_value": "N/A",
                    "severity":  "high",
                    "deviation_pct": round((deviation / original_price) * 100, 1),
                })

            elif dtype == "missing_item":
                if len(inv_items) > 1:
                    removed = inv_items.pop(idx)
                    discrepancies.append({
                        "type":        "missing_line_item",
                        "field":       "line_items",
                        "item":        removed["description"],
                        "po_value":    "present",
                        "inv_value":   "missing",
                        "grn_value":   "present",
                        "severity":    "critical",
                    })

            elif dtype == "grn_short_delivery":
                original_qty = grn_items[idx]["quantity"]
                short_qty = max(1, original_qty - random.randint(1, 3))
                grn_items[idx]["quantity"] = short_qty
                grn_items[idx]["amount"] = round(
                    short_qty * grn_items[idx]["unit_price"], 2)
                discrepancies.append({
                    "type":      "short_delivery",
                    "field":     "quantity",
                    "item":      grn_items[idx]["description"],
                    "po_value":  original_qty,
                    "inv_value": original_qty,
                    "grn_value": short_qty,
                    "severity":  "medium",
                })

    # Recompute totals for invoice with discrepancies
    inv_subtotal, inv_tax, inv_total = _compute_totals(inv_items)
    grn_subtotal, _, _ = _compute_totals(grn_items)

    # Determine correct decision
    critical = any(d["severity"] == "critical" for d in discrepancies)
    high = any(d["severity"] == "high" for d in discrepancies)

    if critical:
        decision = "reject"
        decision_reason = "Critical discrepancy found — missing line items or major violations"
    elif high:
        decision = "escalate"
        decision_reason = "High-severity discrepancies require manager approval"
    elif discrepancies:
        decision = "escalate"
        decision_reason = "Minor discrepancies found — escalate for review"
    else:
        decision = "approve"
        decision_reason = "All three documents match — approve for payment"

    # Build documents
    purchase_order = {
        "document_type": "Purchase Order",
        "po_number":     po_num,
        "vendor":        vendor,
        "buyer":         buyer,
        "issue_date":    _rand_date((-20, -5)),
        "line_items":    po_items,
        "subtotal":      po_subtotal,
        "tax":           po_tax,
        "total":         po_total,
        "currency":      "INR",
        "status":        "issued",
    }

    invoice = {
        "document_type":  "Invoice",
        "invoice_number": inv_no,
        "po_reference":   po_num,
        "vendor":         vendor,
        "buyer":          buyer,
        "issue_date":     _rand_date((-5, 0)),
        "due_date":       _rand_date((15, 30)),
        "line_items":     inv_items,
        "subtotal":       inv_subtotal,
        "tax":            inv_tax,
        "total":          inv_total,
        "currency":       "INR",
    }

    grn = {
        "document_type":  "Goods Receipt Note",
        "grn_number":     grn_no,
        "po_reference":   po_num,
        "invoice_ref":    inv_no,
        "vendor":         vendor,
        "received_by":    buyer,
        "receipt_date":   _rand_date((-3, 0)),
        "line_items":     grn_items,
        "subtotal":       grn_subtotal,
        "currency":       "INR",
        "status":         "received",
    }

    ground_truth = {
        "discrepancies":   discrepancies,
        "n_discrepancies": len(discrepancies),
        "decision":        decision,
        "decision_reason": decision_reason,
        "po_number":       po_num,
        "invoice_number":  inv_no,
        "grn_number":      grn_no,
        "vendor":          vendor,
    }

    return purchase_order, invoice, grn, ground_truth


# ─────────────────────────────────────────────
# Convenience: Generate full task episodes
# ─────────────────────────────────────────────

def generate_task1_episode(seed: int = None) -> dict:
    """Generate one Task 1 episode (noisy invoice + ground truth)."""
    noisy = random.random() > 0.5
    doc, gt = generate_invoice(noisy=noisy, seed=seed)
    return {
        "episode_id":    str(uuid.uuid4())[:8],
        "task_id":       "task1_invoice_parser",
        "document":      doc,
        "ground_truth":  gt,
        "max_steps":     20,
    }


def generate_task2_episode(seed: int = None) -> dict:
    """Generate one Task 2 episode (batch of invoices with injected anomalies)."""
    n_anomalies = random.randint(2, 4)
    invoices, anomalies = generate_invoice_batch(
        n=8, n_anomalies=n_anomalies, seed=seed)
    return {
        "episode_id":       str(uuid.uuid4())[:8],
        "task_id":          "task2_anomaly_detector",
        "document":         {"invoices": invoices, "n_invoices": len(invoices)},
        "anomalies_truth":  anomalies,
        "ground_truth":     {"anomalies": anomalies, "n_anomalies": len(anomalies)},
        "policy_rules": [
            "No single invoice may exceed ₹5,00,000 without a PO",
            "Duplicate invoice numbers indicate potential fraud",
            "All invoices must reference a valid PO number",
            "Vendors must be on the approved vendor list",
            "Invoice date must not be in the future",
            "Tax must be calculated at exactly 18% GST on subtotal",
        ],
        "max_steps": 30,
    }


def generate_task3_episode(seed: int = None) -> dict:
    """Generate one Task 3 episode (PO + Invoice + GRN with discrepancies)."""
    po, invoice, grn, gt = generate_three_way_set(
        introduce_discrepancies=True,
        seed=seed
    )
    return {
        "episode_id":   str(uuid.uuid4())[:8],
        "task_id":      "task3_three_way_reconciler",
        "document":     invoice,          # primary doc = invoice
        "aux_documents": {
            "purchase_order": po,
            "grn":            grn,
        },
        "ground_truth": gt,
        "max_steps":    50,
    }


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TASK 1: Invoice Parser ===")
    ep1 = generate_task1_episode(seed=42)
    print(json.dumps(ep1["document"], indent=2))
    print("\nGround Truth:", json.dumps(ep1["ground_truth"], indent=2))

    print("\n=== TASK 2: Anomaly Detector ===")
    ep2 = generate_task2_episode(seed=42)
    print(f"Batch size: {ep2['document']['n_invoices']} invoices")
    print(f"Injected anomalies: {len(ep2['anomalies_truth'])}")
    for a in ep2["anomalies_truth"]:
        print(
            f"  [{a['severity'].upper()}] {a['invoice_id']} — {a['anomaly_type']}")

    print("\n=== TASK 3: 3-Way Reconciler ===")
    ep3 = generate_task3_episode(seed=42)
    print(f"Decision: {ep3['ground_truth']['decision']}")
    print(f"Discrepancies: {ep3['ground_truth']['n_discrepancies']}")
    for d in ep3["ground_truth"]["discrepancies"]:
        print(f"  [{d['severity'].upper()}] {d['type']} on '{d['item']}'")
