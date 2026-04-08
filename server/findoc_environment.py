"""
FinDocAgent-Env — Core Environment
Implements step() / reset() / state() for all 3 tasks.
"""

import uuid
from typing import Optional
from datetime import datetime
import sys, os

# Fix import path for flat structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import FinDocAction, FinDocObservation, FinDocState, FinDocReward
from data.generator import (
    generate_task1_episode,
    generate_task2_episode,
    generate_task3_episode,
)
from graders.grader1 import grade as grade1
from graders.grader2 import grade as grade2
from graders.grader3 import grade as grade3


# ─────────────────────────────────────────────
# Reward constants
# ─────────────────────────────────────────────

R_CORRECT_EXTRACT   =  0.15
R_WRONG_EXTRACT     = -0.10
R_REPEAT_EXTRACT    = -0.05
R_CORRECT_FLAG      =  0.20
R_WRONG_FLAG        = -0.15
R_CORRECT_MATCH     =  0.15
R_WRONG_MATCH       = -0.10
R_CORRECT_DECISION  =  0.30
R_WRONG_DECISION    = -0.20
R_STEP_PENALTY      = -0.02
R_COMPLETION_BONUS  =  0.50


class FinDocEnvironment:
    """Core FinDocAgent environment. Manages episode state for one session."""

    VALID_TASKS = [
        "task1_invoice_parser",
        "task2_anomaly_detector",
        "task3_three_way_reconciler",
    ]

    MAX_STEPS = {
        "task1_invoice_parser":       20,
        "task2_anomaly_detector":     30,
        "task3_three_way_reconciler": 50,
    }

    def __init__(self):
        self._state: Optional[FinDocState] = None

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self, task_id: str = "task1_invoice_parser", seed: int = None) -> FinDocObservation:
        if task_id not in self.VALID_TASKS:
            task_id = "task1_invoice_parser"

        seed = seed or int(datetime.now().timestamp()) % 100000

        if task_id == "task1_invoice_parser":
            ep = generate_task1_episode(seed=seed)
        elif task_id == "task2_anomaly_detector":
            ep = generate_task2_episode(seed=seed)
        else:
            ep = generate_task3_episode(seed=seed)

        self._state = FinDocState(
            task_id           = task_id,
            task_name         = self._task_name(task_id),
            difficulty        = self._task_difficulty(task_id),
            document          = ep["document"],
            aux_documents     = ep.get("aux_documents", {}),
            ground_truth      = ep["ground_truth"],
            anomalies_truth   = ep.get("anomalies_truth", []),
            extracted         = {},
            flags             = [],
            matches           = [],
            step_count        = 0,
            max_steps         = self.MAX_STEPS[task_id],
            done              = False,
            cumulative_reward = 0.0,
            score             = 0.0,
            episode_id        = ep.get("episode_id", str(uuid.uuid4())[:8]),
        )

        return self._build_observation(
            reward  = 0.0,
            message = f"Episode started. Task: {self._task_name(task_id)}",
        )

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action: FinDocAction) -> FinDocObservation:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            return self._build_observation(reward=0.0, message="Episode already done. Call reset().")

        s = self._state
        s.step_count += 1

        reward  = R_STEP_PENALTY
        message = ""
        atype   = str(action.action_type).lower().strip()

        # ── EXTRACT ──────────────────────────
        if atype == "extract":
            field = str(action.field).strip()
            value = str(action.value).strip()
            if field in s.extracted:
                reward  += R_REPEAT_EXTRACT
                message  = f"Field '{field}' already extracted. Penalty applied."
            else:
                s.extracted[field] = value
                interim = self._interim_extract_reward(field, value)
                reward += interim
                message = (
                    f"Extracted '{field}' = '{value}' (+{interim:.2f})"
                    if interim > 0
                    else f"Extracted '{field}' = '{value}' (looks incorrect)"
                )

        # ── FLAG ─────────────────────────────
        elif atype == "flag":
            flag_record = {
                "invoice_id":   str(action.doc_ref or "").strip(),
                "anomaly_type": str(action.field).strip(),
                "severity":     str(action.severity or "medium").strip(),
                "reason":       str(action.reason or "").strip(),
                "value":        str(action.value or "").strip(),
            }
            already = any(
                f["invoice_id"]   == flag_record["invoice_id"] and
                f["anomaly_type"] == flag_record["anomaly_type"]
                for f in s.flags
            )
            if already:
                reward  += R_WRONG_FLAG
                message  = f"Already flagged {flag_record['invoice_id']} / {flag_record['anomaly_type']}. Penalty."
            else:
                s.flags.append(flag_record)
                interim = self._interim_flag_reward(flag_record)
                reward += interim
                message = f"Flag raised: {flag_record['anomaly_type']} on {flag_record['invoice_id']} ({interim:+.2f})"

        # ── MATCH ────────────────────────────
        elif atype == "match":
            match_record = {
                "field":   str(action.field).strip(),
                "value":   str(action.value).strip(),
                "doc_ref": str(action.doc_ref or "").strip(),
                "reason":  str(action.reason or "").strip(),
                "type":    action.reason.split(":")[0].strip() if action.reason and ":" in action.reason else str(action.field).strip(),
                "item":    str(action.value).strip(),
            }
            s.matches.append(match_record)
            interim = self._interim_match_reward(match_record)
            reward += interim
            message = f"Match recorded: {match_record['field']} ({interim:+.2f})"

        # ── RECONCILE ────────────────────────
        elif atype == "reconcile":
            decision = str(action.decision or action.value or "").strip().lower()
            reason   = str(action.reason or "").strip()
            interim  = self._interim_reconcile_reward(decision)
            reward  += interim
            message  = f"Reconcile decision: '{decision}' ({interim:+.2f})"
            s.extracted["__decision__"] = decision
            s.extracted["__reason__"]   = reason

        # ── SKIP ─────────────────────────────
        elif atype == "skip":
            field = str(action.field).strip()
            s.extracted[field] = "SKIP"
            reward += 0.0
            message = f"Skipped field '{field}'"

        else:
            reward  += R_WRONG_EXTRACT
            message  = f"Unknown action type: '{atype}'"

        reward = round(max(-1.0, min(1.0, reward)), 4)
        s.cumulative_reward = round(s.cumulative_reward + reward, 4)

        done, final_score, end_msg = self._check_done()
        if done:
            s.done  = True
            s.score = final_score
            if final_score >= 0.8:
                bonus = R_COMPLETION_BONUS
                s.cumulative_reward = round(s.cumulative_reward + bonus, 4)
                end_msg += f" Completion bonus: +{bonus}"
            message = end_msg

        return self._build_observation(reward=reward, message=message)

    # ─────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────

    def state(self) -> dict:
        if self._state is None:
            return {"error": "No active episode. Call reset() first."}
        s = self._state
        return {
            "episode_id":        s.episode_id,
            "task_id":           s.task_id,
            "task_name":         s.task_name,
            "difficulty":        s.difficulty,
            "step_count":        s.step_count,
            "max_steps":         s.max_steps,
            "done":              s.done,
            "score":             s.score,
            "cumulative_reward": s.cumulative_reward,
            "extracted":         s.extracted,
            "flags":             s.flags,
            "matches":           s.matches,
            "ground_truth":      s.ground_truth,
            "anomalies_truth":   s.anomalies_truth,
            "document":          s.document,
            "aux_documents":     s.aux_documents,
        }

    # ─────────────────────────────────────────
    # Intermediate reward helpers
    # ─────────────────────────────────────────

    def _interim_extract_reward(self, field: str, value: str) -> float:
        s  = self._state
        gt = s.ground_truth
        if s.task_id == "task1_invoice_parser":
            from graders.grader1 import _compare_field, _normalize_field
            canon = _normalize_field(field)
            if canon not in gt:
                return R_WRONG_EXTRACT
            score = _compare_field(canon, value, gt[canon])
            return R_CORRECT_EXTRACT if score >= 0.5 else R_WRONG_EXTRACT
        if s.task_id == "task3_three_way_reconciler" and field == "__decision__":
            return self._interim_reconcile_reward(value)
        return 0.0

    def _interim_flag_reward(self, flag: dict) -> float:
        s = self._state
        if s.task_id != "task2_anomaly_detector":
            return R_WRONG_FLAG
        from graders.grader2 import _normalize_anomaly_type, _normalize_invoice_id
        a_type = _normalize_anomaly_type(flag.get("anomaly_type", ""))
        a_id   = _normalize_invoice_id(flag.get("invoice_id", ""))
        for truth in s.anomalies_truth:
            t_type = _normalize_anomaly_type(truth["anomaly_type"])
            t_id   = _normalize_invoice_id(truth["invoice_id"])
            if a_type == t_type and a_id == t_id:
                return R_CORRECT_FLAG
            if a_type == t_type or a_id == t_id:
                return R_CORRECT_FLAG * 0.5
        return R_WRONG_FLAG

    def _interim_match_reward(self, match: dict) -> float:
        s = self._state
        if s.task_id != "task3_three_way_reconciler":
            return 0.0
        from graders.grader3 import _normalize_disc_type, _normalize_item, _item_match
        a_type = _normalize_disc_type(match.get("type", match.get("field", "")))
        a_item = _normalize_item(match.get("item", match.get("value", "")))
        for d in s.ground_truth.get("discrepancies", []):
            t_type = _normalize_disc_type(d["type"])
            t_item = _normalize_item(d["item"])
            if a_type == t_type and _item_match(a_item, t_item):
                return R_CORRECT_MATCH
            if a_type == t_type or _item_match(a_item, t_item):
                return R_CORRECT_MATCH * 0.5
        return R_WRONG_MATCH

    def _interim_reconcile_reward(self, decision: str) -> float:
        from graders.grader3 import _normalize_decision, DECISION_ADJACENCY
        s     = self._state
        a_dec = _normalize_decision(decision)
        t_dec = _normalize_decision(s.ground_truth.get("decision", "approve"))
        if a_dec == t_dec:
            return R_CORRECT_DECISION
        partial = DECISION_ADJACENCY.get((a_dec, t_dec), 0.0)
        return partial * R_CORRECT_DECISION if partial > 0 else R_WRONG_DECISION

    # ─────────────────────────────────────────
    # Episode termination
    # ─────────────────────────────────────────

    def _check_done(self):
        s = self._state
        if s.step_count >= s.max_steps:
            score, msg = self._compute_final_score()
            return True, score, f"Max steps reached. Final score: {score:.2f}. {msg}"
        if s.task_id == "task1_invoice_parser":
            attempted = len([v for v in s.extracted.values() if v != ""])
            if attempted >= 9:
                score, msg = self._compute_final_score()
                return True, score, f"Extraction complete. Score: {score:.2f}"
        elif s.task_id == "task2_anomaly_detector":
            if "__decision__" in s.extracted:
                score, msg = self._compute_final_score()
                return True, score, f"Review complete. Score: {score:.2f}"
        elif s.task_id == "task3_three_way_reconciler":
            if "__decision__" in s.extracted:
                score, msg = self._compute_final_score()
                return True, score, f"Reconciliation complete. Score: {score:.2f}"
        return False, 0.0, ""

    def _compute_final_score(self):
        s  = self._state
        gt = s.ground_truth
        if s.task_id == "task1_invoice_parser":
            result = grade1(s.extracted, gt)
            return result["score"], result["feedback"]
        elif s.task_id == "task2_anomaly_detector":
            result = grade2(s.flags, gt)
            return result["score"], result["feedback"]
        elif s.task_id == "task3_three_way_reconciler":
            decision = s.extracted.get("__decision__", "")
            reason   = s.extracted.get("__reason__", "")
            result   = grade3(s.matches, decision, reason, gt)
            return result["score"], result["feedback"]
        return 0.0, "Unknown task"

    # ─────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────

    def _build_observation(self, reward: float, message: str) -> FinDocObservation:
        s = self._state
        return FinDocObservation(
            task_id           = s.task_id,
            document          = s.document,
            aux_documents     = s.aux_documents,
            extracted         = dict(s.extracted),
            flags             = list(s.flags),
            matches           = list(s.matches),
            step_count        = s.step_count,
            max_steps         = s.max_steps,
            done              = s.done,
            reward            = reward,
            cumulative_reward = s.cumulative_reward,
            score             = s.score,
            message           = message,
            valid_actions     = self._valid_actions(),
        )

    def _valid_actions(self) -> list:
        s = self._state
        if s.task_id == "task1_invoice_parser":
            return ["extract", "skip"]
        elif s.task_id == "task2_anomaly_detector":
            return ["flag", "reconcile", "skip"]
        else:
            return ["match", "reconcile", "skip"]

    def _task_name(self, task_id: str) -> str:
        return {
            "task1_invoice_parser":       "Invoice Parser",
            "task2_anomaly_detector":     "Anomaly Detector",
            "task3_three_way_reconciler": "3-Way PO/Invoice/GRN Reconciler",
        }.get(task_id, task_id)

    def _task_difficulty(self, task_id: str) -> str:
        return {
            "task1_invoice_parser":       "easy",
            "task2_anomaly_detector":     "medium",
            "task3_three_way_reconciler": "hard",
        }.get(task_id, "unknown")