# backend/agents/supervisor.py
import logging
from typing import Any, Dict, Optional

from backend.core.config import get_settings
from backend.core.constants import (
    ADAPTIVE_BUDGET_LOW,
    ADAPTIVE_BUDGET_MID,
    CRITIC_PASS_THRESHOLD,
)

logger = logging.getLogger(__name__)


class Supervisor:
    """Orchestrates the agent pipeline."""

    _NON_SNAPSHOT_KEYS = frozenset({
        "reasoning_log",
        "editor_rolled_back",
        "iteration",
    })

    _EPHEMERAL_KEYS = frozenset({
        "critic_adequacy",
        "critic_fluency",
        "critic_terminology",
        "critic_score",
        "critic_reasoning",
        "critic_suggestions",
        "critic_missing_terms",
        "critic_failed",
        "critic_passed",
        "validator_errors",
        "validator_warnings",
        "validation_passed",
        "supervisor_iterations",
        "supervisor_final",
        "supervisor_stopped_reason",
        "supervisor_budget",
    })

    _EDITOR_DEFAULTS: Dict[str, Any] = {
        "editor_changed": False,
        "editor_failed": False,
        "editor_original": None,
        "editor_corrected": None,
        "editor_iterations": 0,
    }

    def __init__(self, max_iterations: int = 3, engine: Optional[str] = None):
        self.max_iterations = max_iterations
        self.engine = engine or get_settings().AGENT_ENGINE
        self._iteration = 0
        self._validator = None
        self._critic = None
        self._editor = None

    def _get_validator(self):
        if self._validator is None:
            from backend.agents.validator import Validator
            self._validator = Validator(engine=self.engine)
        return self._validator

    def _get_critic(self):
        if self._critic is None:
            from backend.agents.critic import Critic
            self._critic = Critic(engine=self.engine)
        return self._critic

    def _get_editor(self):
        if self._editor is None:
            from backend.agents.editor import Editor
            self._editor = Editor(engine=self.engine)
        return self._editor

    @staticmethod
    def _has_actionable_suggestions(state: Dict[str, Any]) -> bool:
        return bool(state.get("critic_suggestions") or [])

    @classmethod
    def _adaptive_budget(cls, state: Dict[str, Any]) -> int:
        score = state.get("critic_score")

        if score is None or score < ADAPTIVE_BUDGET_LOW:
            return 3
        if score < ADAPTIVE_BUDGET_MID:
            return 2

        if not state.get("validation_passed", True):
            return 1
        if cls._has_actionable_suggestions(state):
            return 1
        return 0

    @classmethod
    def _needs_correction(cls, state: Dict[str, Any]) -> bool:
        if state.get("critic_failed"):
            return False

        validation_passed = state.get("validation_passed", True)
        critic_score = state.get("critic_score")

        if critic_score is None:
            return not validation_passed

        if not validation_passed:
            return True
        if critic_score < CRITIC_PASS_THRESHOLD:
            return True
        return cls._has_actionable_suggestions(state)

    @staticmethod
    def _is_improvement(prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        if curr.get("critic_failed"):
            return False

        prev_score = prev.get("critic_score")
        curr_score = curr.get("critic_score")

        if curr_score is None and prev_score is not None:
            return False
        if curr_score is not None and prev_score is None:
            return True
        if curr_score is not None and prev_score is not None:
            if curr_score > prev_score:
                return True
            if curr_score < prev_score:
                return False

        if curr.get("validation_passed") and not prev.get("validation_passed"):
            return True
        if prev.get("validation_passed") and not curr.get("validation_passed"):
            return False

        return False

    @staticmethod
    def _compute_final(state: Dict[str, Any]) -> bool:
        if state.get("critic_failed"):
            return False
        if not state.get("validation_passed", False):
            return False
        score = state.get("critic_score")
        if score is None:
            return True
        return score >= CRITIC_PASS_THRESHOLD

    @classmethod
    def _snapshot(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for key, value in state.items():
            if key in cls._NON_SNAPSHOT_KEYS:
                continue
            snap[key] = list(value) if isinstance(value, list) else value
        return snap

    @classmethod
    def _restore(cls, state: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        for key in list(state.keys()):
            if key not in snapshot and key not in cls._NON_SNAPSHOT_KEYS:
                del state[key]
        for key, value in snapshot.items():
            state[key] = list(value) if isinstance(value, list) else value

    def _finalize(
        self,
        state: Dict[str, Any],
        reason: str,
        initial_score: Optional[float] = None,
        budget: int = 0,
    ) -> Dict[str, Any]:
        final_passed = self._compute_final(state)
        state["supervisor_iterations"] = self._iteration
        state["supervisor_final"] = final_passed
        state["supervisor_stopped_reason"] = reason
        state["supervisor_budget"] = budget
        state["reasoning_log"].append({
            "agent": "Supervisor",
            "step": "final",
            "iterations": self._iteration,
            "final_passed": final_passed,
            "stopped_reason": reason,
            "editor_rolled_back": state.get("editor_rolled_back", False),
            "budget": budget,
            "initial_score": initial_score,
            "final_score": state.get("critic_score"),
        })
        logger.info(
            "Final: iterations=%s, passed=%s, rolled_back=%s, "
            "budget=%s, reason=%s",
            self._iteration,
            final_passed,
            state.get("editor_rolled_back", False),
            budget,
            reason,
        )
        return state

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._iteration = 0
        state["editor_rolled_back"] = False
        state["reasoning_log"] = []

        for key in self._EPHEMERAL_KEYS:
            state.pop(key, None)

        for key, value in self._EDITOR_DEFAULTS.items():
            state[key] = value

        use_critic = bool(state.get("use_critic", True))
        use_editor = bool(state.get("use_editor", True))

        validator = self._get_validator()
        state = await validator.process(state)

        if not use_critic and not use_editor:
            logger.info("Agents disabled, returning after validator")
            return self._finalize(state, "agents_disabled")

        if use_critic:
            critic = self._get_critic()
            state = await critic.process(state)

            if state.get("critic_failed"):
                logger.info("Stopping early: critic_failed")
                return self._finalize(state, "critic_failed")

        if not use_editor:
            logger.info("Editor disabled, returning after critic")
            return self._finalize(state, "editor_disabled")

        editor = self._get_editor()
        initial_score = state.get("critic_score")
        needs_correction = self._needs_correction(state)
        adaptive_budget = self._adaptive_budget(state) if needs_correction else 0
        effective_max = min(self.max_iterations, adaptive_budget)
        state["supervisor_budget"] = effective_max

        logger.info(
            "initial_score=%s, adaptive_budget=%s, effective_max=%s, "
            "needs_correction=%s",
            initial_score,
            adaptive_budget,
            effective_max,
            needs_correction,
        )

        snapshot = self._snapshot(state)

        if not needs_correction:
            stopped_reason = "converged"
        elif effective_max == 0:
            stopped_reason = "budget_exhausted"
        else:
            stopped_reason = "max_iterations"

        while needs_correction and self._iteration < effective_max:
            self._iteration += 1
            state["iteration"] = self._iteration

            state = await editor.process(state)
            editor_failed = state.get("editor_failed", False)
            editor_changed = state.get("editor_changed", False)

            state = await validator.process(state)
            if use_critic:
                state = await critic.process(state)

            if editor_failed or (use_critic and state.get("critic_failed")):
                logger.info(
                    "Rollback: editor_failed=%s, critic_failed=%s",
                    editor_failed,
                    state.get("critic_failed"),
                )
                self._restore(state, snapshot)
                state["editor_rolled_back"] = True
                stopped_reason = "rollback_error"
                break

            if not editor_changed:
                logger.info("Editor did not change translation, stopping.")
                state["editor_rolled_back"] = False
                stopped_reason = "no_change"
                break

            if not self._is_improvement(snapshot, state):
                logger.info(
                    "Rollback: no improvement (%s -> %s)",
                    snapshot.get("critic_score"),
                    state.get("critic_score"),
                )
                self._restore(state, snapshot)
                state["editor_rolled_back"] = True
                stopped_reason = "rollback_regression"
                break

            snapshot = self._snapshot(state)
            state["editor_rolled_back"] = False
            needs_correction = self._needs_correction(state)

            if not needs_correction:
                stopped_reason = "converged"

        return self._finalize(
            state,
            stopped_reason,
            initial_score=initial_score,
            budget=effective_max,
        )

    def get_status(self) -> Dict[str, Any]:
        critic = self._critic
        validator = self._validator
        return {
            "iterations": self._iteration,
            "validator_passed": (
                validator.get_last_result().get("passed", False)
                if validator and validator.get_last_result()
                else False
            ),
            "critic_score": (
                critic.get_last_result().get("overall_score", 0)
                if critic and critic.get_last_result()
                else 0
            ),
        }
