from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


# ─────────────────────────────────────────────
# Action Types
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    EXTRACT = "extract"     # Extract a field value from document
    FLAG = "flag"        # Flag an anomaly
    MATCH = "match"       # Assert two fields match across documents
    RECONCILE = "reconcile"   # Final approval / rejection / escalation decision
    SKIP = "skip"        # Mark field as not found / not applicable


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReconcileDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


# ─────────────────────────────────────────────
# Action — what the agent sends each step
# ─────────────────────────────────────────────

@dataclass
class FinDocAction:
    """
    The action the agent takes at each step.

    Fields:
        action_type : one of extract | flag | match | reconcile | skip
        field       : the document field being acted on (e.g. "vendor_name", "total_amount")
        value       : the extracted/asserted value (string representation)
        reason      : explanation for flag / reconcile actions
        severity    : used only with action_type="flag"
        decision    : used only with action_type="reconcile" (approve/reject/escalate)
        doc_ref     : optional document reference (e.g. "invoice", "po", "grn")
    """
    action_type: str           # ActionType value
    field: str           # Field being acted on
    value: str           # Extracted or asserted value
    reason: str = ""     # Required for flag / reconcile
    severity: str = ""     # Required for flag (low/medium/high/critical)
    decision: str = ""     # Required for reconcile (approve/reject/escalate)
    doc_ref: str = ""     # Optional: which doc this action targets


# ─────────────────────────────────────────────
# Observation — what the agent receives each step
# ─────────────────────────────────────────────

@dataclass
class FinDocObservation:
    """
    Observation returned to the agent after each step.

    Fields:
        task_id        : which task is active (task1/task2/task3)
        document       : the primary document content (noisy text or structured dict)
        aux_documents  : secondary documents (PO, GRN) for task3
        extracted      : fields extracted so far {field: value}
        flags          : list of anomalies raised so far
        matches        : list of match assertions made
        step_count     : steps used so far
        max_steps      : episode step limit
        done           : whether episode has ended
        reward         : reward for the last action
        cumulative_reward : total reward so far
        score          : grader score 0.0–1.0 (only populated at done=True)
        message        : human-readable feedback on last action
        valid_actions  : list of valid action_types at this state
    """
    task_id: str
    document: dict
    aux_documents: dict = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 30
    done: bool = False
    reward: float = 0.0
    cumulative_reward: float = 0.0
    score: float = 0.0
    message: str = ""
    valid_actions: list = field(default_factory=lambda: [
        "extract", "flag", "match", "reconcile", "skip"
    ])


# ─────────────────────────────────────────────
# Reward — structured reward breakdown
# ─────────────────────────────────────────────

@dataclass
class FinDocReward:
    """
    Structured reward breakdown for interpretability.

    All fields sum to the total step reward.
    """
    total: float = 0.0   # Final reward value used by trainer

    # Positive signals
    correct_extraction: float = 0.0   # +0.15 per correct field extracted
    correct_flag: float = 0.0   # +0.20 per correct anomaly flag
    correct_match: float = 0.0   # +0.15 per correct match assertion
    correct_decision: float = 0.0   # +0.30 for correct reconcile decision
    completion_bonus: float = 0.0   # +0.50 perfect episode bonus

    # Negative signals
    wrong_extraction: float = 0.0   # -0.10 hallucinated / wrong value
    wrong_flag: float = 0.0   # -0.15 false positive flag
    wrong_match: float = 0.0   # -0.10 incorrect match assertion
    wrong_decision: float = 0.0   # -0.20 wrong reconcile decision
    step_penalty: float = 0.0   # -0.02 per step (efficiency signal)
    repeat_penalty: float = 0.0   # -0.05 for re-extracting same field


# ─────────────────────────────────────────────
# State — full internal environment state
# ─────────────────────────────────────────────

@dataclass
class FinDocState:
    """
    Full internal state of the environment (returned by state() endpoint).
    Contains ground truth — not exposed to agent during episode.
    """
    task_id: str
    task_name: str
    difficulty: str
    document: dict
    aux_documents: dict
    ground_truth: dict        # correct answers
    anomalies_truth: list        # ground truth anomalies for task2
    extracted: dict
    flags: list
    matches: list
    step_count: int
    max_steps: int
    done: bool
    cumulative_reward: float
    score: float
    episode_id: str
