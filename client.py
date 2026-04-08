"""
FinDocAgent-Env — Python Client
Typed sync/async client to interact with the environment server.

Usage (sync):
    from client import FinDocClient
    client = FinDocClient(base_url="http://localhost:7860")
    obs = client.reset("task1_invoice_parser")
    obs = client.step(action_type="extract", field="vendor_name", value="Infosys Ltd")

Usage (async):
    from client import AsyncFinDocClient
    async with AsyncFinDocClient(base_url="http://localhost:7860") as client:
        obs = await client.reset("task1_invoice_parser")
"""

import requests
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────

@dataclass
class StepResult:
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

    @classmethod
    def from_dict(cls, d: dict) -> "StepResult":
        return cls(
            session_id=d.get("session_id", ""),
            task_id=d.get("task_id", ""),
            document=d.get("document", {}),
            aux_documents=d.get("aux_documents", {}),
            extracted=d.get("extracted", {}),
            flags=d.get("flags", []),
            matches=d.get("matches", []),
            step_count=d.get("step_count", 0),
            max_steps=d.get("max_steps", 30),
            done=d.get("done", False),
            reward=d.get("reward", 0.0),
            cumulative_reward=d.get("cumulative_reward", 0.0),
            score=d.get("score", 0.0),
            message=d.get("message", ""),
            valid_actions=d.get("valid_actions", []),
        )


# ─────────────────────────────────────────────
# Sync Client
# ─────────────────────────────────────────────

class FinDocClient:
    """Synchronous HTTP client for FinDocAgent-Env."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None

    def _post(self, endpoint: str, payload: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/{endpoint}",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        resp = requests.get(
            f"{self.base_url}/{endpoint}",
            params=params or {},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str = "task1_invoice_parser",
        seed: Optional[int] = None,
    ) -> StepResult:
        """Reset environment and start a new episode."""
        data = self._post("reset", {
            "task_id":    task_id,
            "seed":       seed,
            "session_id": self.session_id,
        })
        self.session_id = data.get("session_id", self.session_id)
        return StepResult.from_dict(data)

    def step(
        self,
        action_type: str,
        field:       str = "",
        value:       str = "",
        reason:      str = "",
        severity:    str = "",
        decision:    str = "",
        doc_ref:     str = "",
    ) -> StepResult:
        """Take one action in the environment."""
        data = self._post("step", {
            "action_type": action_type,
            "field":       field,
            "value":       value,
            "reason":      reason,
            "severity":    severity,
            "decision":    decision,
            "doc_ref":     doc_ref,
            "session_id":  self.session_id,
        })
        self.session_id = data.get("session_id", self.session_id)
        return StepResult.from_dict(data)

    def state(self) -> dict:
        """Get full internal state (includes ground truth)."""
        return self._get("state", {"session_id": self.session_id})

    def tasks(self) -> dict:
        """List all tasks and their action schemas."""
        return self._get("tasks")

    def grader(self) -> dict:
        """Run grader on current episode state."""
        resp = requests.post(
            f"{self.base_url}/grader",
            json={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def baseline(self) -> dict:
        """Trigger baseline inference and return scores."""
        resp = requests.post(f"{self.base_url}/baseline", timeout=120)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        return self._get("health")

    def run_episode(
        self,
        task_id: str,
        agent_fn,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run a full episode with a given agent function.

        agent_fn: callable(observation: StepResult) → dict of action kwargs
                  Must return dict with keys: action_type, field, value, etc.

        Returns: {score, cumulative_reward, steps, done}
        """
        obs = self.reset(task_id=task_id, seed=seed)
        if verbose:
            print(f"[Episode] Task: {task_id} | Max steps: {obs.max_steps}")

        while not obs.done:
            action_kwargs = agent_fn(obs)
            obs = self.step(**action_kwargs)
            if verbose:
                print(
                    f"  Step {obs.step_count:02d} | reward={obs.reward:+.2f} | {obs.message[:80]}")

        if verbose:
            print(
                f"[Done] Score: {obs.score:.2f} | Reward: {obs.cumulative_reward:.2f}")

        return {
            "score":             obs.score,
            "cumulative_reward": obs.cumulative_reward,
            "steps":             obs.step_count,
            "done":              obs.done,
            "message":           obs.message,
        }


# ─────────────────────────────────────────────
# Async Client
# ─────────────────────────────────────────────

try:
    import httpx

    class AsyncFinDocClient:
        """Async HTTP client for FinDocAgent-Env."""

        def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self.session_id: Optional[str] = None
            self._client: Optional[httpx.AsyncClient] = None

        async def __aenter__(self):
            self._client = httpx.AsyncClient(timeout=self.timeout)
            return self

        async def __aexit__(self, *args):
            if self._client:
                await self._client.aclose()

        async def reset(self, task_id: str = "task1_invoice_parser", seed: Optional[int] = None) -> StepResult:
            resp = await self._client.post(f"{self.base_url}/reset", json={
                "task_id": task_id, "seed": seed, "session_id": self.session_id
            })
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data.get("session_id", self.session_id)
            return StepResult.from_dict(data)

        async def step(self, action_type: str, field: str = "", value: str = "",
                       reason: str = "", severity: str = "", decision: str = "",
                       doc_ref: str = "") -> StepResult:
            resp = await self._client.post(f"{self.base_url}/step", json={
                "action_type": action_type, "field": field, "value": value,
                "reason": reason, "severity": severity, "decision": decision,
                "doc_ref": doc_ref, "session_id": self.session_id,
            })
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data.get("session_id", self.session_id)
            return StepResult.from_dict(data)

        async def state(self) -> dict:
            resp = await self._client.get(f"{self.base_url}/state",
                                          params={"session_id": self.session_id})
            resp.raise_for_status()
            return resp.json()

        async def tasks(self) -> dict:
            resp = await self._client.get(f"{self.base_url}/tasks")
            resp.raise_for_status()
            return resp.json()

        async def grader(self) -> dict:
            resp = await self._client.post(f"{self.base_url}/grader",
                                           json={"session_id": self.session_id})
            resp.raise_for_status()
            return resp.json()

except ImportError:
    # httpx not installed — async client unavailable
    class AsyncFinDocClient:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Install httpx for async support: pip install httpx")


# ─────────────────────────────────────────────
# Quick smoke test (direct env, no server needed)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from server.findoc_environment import FinDocEnvironment
    from models import FinDocAction

    print("=== Direct Environment Smoke Test ===\n")

    for task_id in [
        "task1_invoice_parser",
        "task2_anomaly_detector",
        "task3_three_way_reconciler",
    ]:
        env = FinDocEnvironment()
        obs = env.reset(task_id=task_id, seed=42)
        print(f"Task: {task_id}")
        print(
            f"  Step: {obs.step_count} | Done: {obs.done} | Valid: {obs.valid_actions}")
        print(f"  Message: {obs.message}")

        # Take one step
        if task_id == "task1_invoice_parser":
            action = FinDocAction(
                action_type="extract", field="vendor_name", value="Tata Consultancy Services")
        elif task_id == "task2_anomaly_detector":
            action = FinDocAction(action_type="flag", field="duplicate_invoice",
                                  value="Duplicate found", doc_ref="INV-BATCH-001",
                                  severity="high", reason="Same invoice number appears twice")
        else:
            action = FinDocAction(action_type="match", field="quantity_mismatch",
                                  value="Software License - Annual",
                                  reason="type:quantity_mismatch")

        obs = env.step(action)
        print(f"  After step → reward={obs.reward:+.2f} | {obs.message[:70]}")
        print()
