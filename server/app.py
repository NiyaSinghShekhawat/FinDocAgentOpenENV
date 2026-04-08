"""
FinDocAgent-Env - FastAPI Server
"""
import os
import sys
import uuid
from typing import Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import FinDocAction, FinDocObservation
from server.findoc_environment import FinDocEnvironment

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinDocAgent-Env",
    description="OpenEnv environment for financial document processing",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_sessions: dict = {}
DEFAULT_TASK = "task1_invoice_parser"

def _get_or_create_session(session_id: Optional[str]):
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())[:8]
        _sessions[session_id] = FinDocEnvironment()
    return session_id, _sessions[session_id]

# ── Schemas ────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id:    Optional[str] = None
    seed:       Optional[int] = None
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action_type: str
    field:       str           = ""
    value:       str           = ""
    reason:      str           = ""
    severity:    str           = ""
    decision:    str           = ""
    doc_ref:     str           = ""
    session_id:  Optional[str] = None

class GraderRequest(BaseModel):
    session_id: str

class ObsResponse(BaseModel):
    session_id:        str
    task_id:           str
    document:          dict
    aux_documents:     dict
    extracted:         dict
    flags:             list
    matches:           list
    step_count:        int
    max_steps:         int
    done:              bool
    reward:            float
    cumulative_reward: float
    score:             float
    message:           str
    valid_actions:     list

def _obs_to_response(session_id: str, obs: FinDocObservation) -> ObsResponse:
    return ObsResponse(
        session_id=session_id, task_id=obs.task_id,
        document=obs.document, aux_documents=obs.aux_documents,
        extracted=obs.extracted, flags=obs.flags, matches=obs.matches,
        step_count=obs.step_count, max_steps=obs.max_steps,
        done=obs.done, reward=obs.reward,
        cumulative_reward=obs.cumulative_reward, score=obs.score,
        message=obs.message, valid_actions=obs.valid_actions,
    )

# ── Core endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "env": "FinDocAgent-Env", "version": "1.0.0"}

@app.get("/")
def root():
    return {
        "name": "FinDocAgent-Env", "version": "1.0.0",
        "description": "OpenEnv environment for financial document processing",
        "docs": "/docs", "health": "/health", "tasks": "/tasks",
        "endpoints": ["GET /health","POST /reset","POST /step",
                      "GET /state","GET /tasks","POST /grader","POST /baseline"],
    }

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> ObsResponse:
    """Accepts empty body {} OR full ResetRequest."""
    task_id    = DEFAULT_TASK
    seed       = None
    session_id = None
    if req is not None:
        task_id    = req.task_id    or DEFAULT_TASK
        seed       = req.seed
        session_id = req.session_id
    session_id, env = _get_or_create_session(session_id)
    obs = env.reset(task_id=task_id, seed=seed)
    return _obs_to_response(session_id, obs)

@app.post("/step")
def step(req: StepRequest) -> ObsResponse:
    session_id, env = _get_or_create_session(req.session_id)
    if env._state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    action = FinDocAction(
        action_type=req.action_type, field=req.field, value=req.value,
        reason=req.reason, severity=req.severity,
        decision=req.decision, doc_ref=req.doc_ref,
    )
    obs = env.step(action)
    return _obs_to_response(session_id, obs)

@app.get("/state")
def state(session_id: Optional[str] = None) -> dict:
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    return _sessions[session_id].state()

# ── Hackathon endpoints ────────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "id": "task1_invoice_parser", "name": "Invoice Parser",
                "difficulty": "easy", "max_steps": 20, "pass_threshold": 0.70,
                "description": "Extract structured fields from a noisy invoice document.",
                "action_schema": {"action_type": "extract | skip",
                                  "field": "Field name", "value": "Extracted value"},
                "fields_to_extract": ["vendor_name","buyer_name","invoice_number",
                    "po_reference","issue_date","due_date","subtotal",
                    "tax_amount","total_amount","currency","n_line_items"],
            },
            {
                "id": "task2_anomaly_detector", "name": "Anomaly Detector",
                "difficulty": "medium", "max_steps": 30, "pass_threshold": 0.60,
                "description": "Review a batch of invoices and flag policy violations.",
                "action_schema": {"action_type": "flag | reconcile | skip",
                                  "field": "Anomaly type", "severity": "low|medium|high|critical",
                                  "doc_ref": "Invoice ID", "decision": "approve|reject|escalate"},
            },
            {
                "id": "task3_three_way_reconciler", "name": "3-Way PO/Invoice/GRN Reconciler",
                "difficulty": "hard", "max_steps": 50, "pass_threshold": 0.55,
                "description": "Reconcile PO, Invoice, and GRN. Identify discrepancies and decide.",
                "action_schema": {"action_type": "match | reconcile | skip",
                                  "field": "Discrepancy type", "decision": "approve|reject|escalate"},
            },
        ]
    }

@app.post("/grader")
def run_grader(req: GraderRequest) -> dict:
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    env = _sessions[req.session_id]
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode.")
    s = env._state
    gt = s.ground_truth
    if s.task_id == "task1_invoice_parser":
        from graders.grader1 import grade as g1
        result = g1(s.extracted, gt)
    elif s.task_id == "task2_anomaly_detector":
        from graders.grader2 import grade as g2
        result = g2(s.flags, gt)
    else:
        from graders.grader3 import grade as g3
        result = g3(s.matches, s.extracted.get("__decision__",""),
                    s.extracted.get("__reason__",""), gt)
    return {
        "session_id": req.session_id, "task_id": s.task_id,
        "score": result["score"], "passed": result.get("passed", False),
        "feedback": result.get("feedback", ""), "details": result,
    }

@app.post("/baseline")
def run_baseline() -> dict:
    results = {}
    thresholds = {"task1_invoice_parser": 0.70,
                  "task2_anomaly_detector": 0.60,
                  "task3_three_way_reconciler": 0.55}
    for task_id in thresholds:
        env = FinDocEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        s   = env._state
        if task_id == "task1_invoice_parser":
            for field in ["vendor_name","buyer_name","invoice_number","po_reference",
                          "issue_date","due_date","subtotal","tax_amount","total_amount","currency"]:
                if field in s.ground_truth:
                    obs = env.step(FinDocAction(action_type="extract",
                                               field=field, value=str(s.ground_truth[field])))
                    if obs.done: break
        elif task_id == "task2_anomaly_detector":
            if s.anomalies_truth:
                a = s.anomalies_truth[0]
                obs = env.step(FinDocAction(action_type="flag", field=a["anomaly_type"],
                    value=a["description"], doc_ref=a["invoice_id"],
                    severity=a["severity"], reason=a["description"]))
            obs = env.step(FinDocAction(action_type="reconcile", field="done",
                           value="", decision="escalate", reason="Review complete"))
        elif task_id == "task3_three_way_reconciler":
            discs = s.ground_truth.get("discrepancies", [])
            if discs:
                d = discs[0]
                obs = env.step(FinDocAction(action_type="match", field=d["type"],
                               value=d["item"], reason=f"type:{d['type']}"))
            dec = s.ground_truth.get("decision", "escalate")
            obs = env.step(FinDocAction(action_type="reconcile", field="decision",
                value=dec, decision=dec,
                reason=s.ground_truth.get("decision_reason", "Based on analysis")))
        fs = env.state()
        results[task_id] = {
            "score": fs["score"], "cumulative_reward": fs["cumulative_reward"],
            "steps_used": fs["step_count"],
            "passed": fs["score"] >= thresholds[task_id],
        }
    overall = sum(r["score"] for r in results.values()) / len(results)
    return {"agent": "rule_based_baseline", "timestamp": datetime.now().isoformat(),
            "overall_score": round(overall, 4), "task_results": results}

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
