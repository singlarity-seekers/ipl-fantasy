"""
RL / Bandit layer for lineup selection.

Uses Thompson Sampling or UCB to balance exploitation (highest-reward
lineup) vs exploration (trying unusual lineups), updating beliefs
after each match day based on actual outcomes.

Contest-type aware: H2H vs small-league vs mega-contest strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ContestType(Enum):
    HEAD_TO_HEAD = "h2h"
    SMALL_LEAGUE = "small"
    MEGA_CONTEST = "mega"


@dataclass
class ArmState:
    """State of one bandit arm (a candidate lineup)."""

    lineup_id: int
    reward_score: float  # from the reward model
    # Beta distribution parameters for Thompson Sampling
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0  # failures + prior
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return self.reward_score
        return self.total_reward / self.pulls

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score."""
        if self.pulls == 0:
            return float("inf")
        exploration = np.sqrt(2 * np.log(max(self.pulls + 10, 1)) / self.pulls)
        return self.mean_reward + exploration


class LineupBandit:
    """
    Multi-armed bandit for lineup selection.

    Each arm is a candidate lineup. The bandit learns which lineup
    styles perform best in different contest types.
    """

    def __init__(
        self,
        strategy: str = "thompson",
        contest_type: ContestType = ContestType.MEGA_CONTEST,
    ):
        """
        Args:
            strategy: "thompson" (Thompson Sampling) or "ucb" (UCB1).
            contest_type: Affects exploration/exploitation balance.
        """
        self.strategy = strategy
        self.contest_type = contest_type
        self.arms: list[ArmState] = []
        self.rng = np.random.default_rng(42)
        self._history: list[dict] = []

    def initialize_arms(
        self,
        lineup_scores: list[tuple[int, float]],
    ) -> None:
        """
        Initialize bandit arms from reward model scores.

        Args:
            lineup_scores: List of (lineup_id, reward_score) tuples.
        """
        self.arms = [
            ArmState(lineup_id=lid, reward_score=score)
            for lid, score in lineup_scores
        ]

        # Contest-type-specific priors
        if self.contest_type == ContestType.MEGA_CONTEST:
            # Favor exploration in mega contests (need differentiation)
            for arm in self.arms:
                arm.alpha = 1.0
                arm.beta = 1.0
        elif self.contest_type == ContestType.HEAD_TO_HEAD:
            # Favor exploitation in H2H (consistency wins)
            for arm in self.arms:
                arm.alpha = max(arm.reward_score * 2, 1.0)
                arm.beta = 1.0
        elif self.contest_type == ContestType.SMALL_LEAGUE:
            # Balanced
            for arm in self.arms:
                arm.alpha = max(arm.reward_score, 1.0)
                arm.beta = 1.0

        logger.info(
            "Initialized %d arms for %s contest",
            len(self.arms), self.contest_type.value,
        )

    def select(self) -> int:
        """
        Select a lineup (arm) to play.

        Returns:
            Index into self.arms of the selected lineup.
        """
        if not self.arms:
            raise ValueError("No arms initialized. Call initialize_arms() first.")

        if self.strategy == "thompson":
            return self._thompson_select()
        elif self.strategy == "ucb":
            return self._ucb_select()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _thompson_select(self) -> int:
        """Thompson Sampling: sample from each arm's posterior, pick highest."""
        samples = [
            self.rng.beta(arm.alpha, arm.beta) * arm.reward_score
            for arm in self.arms
        ]
        selected = int(np.argmax(samples))

        logger.debug(
            "Thompson selected arm %d (lineup %d, sampled=%.3f)",
            selected, self.arms[selected].lineup_id, samples[selected],
        )
        return selected

    def _ucb_select(self) -> int:
        """UCB1: pick arm with highest upper confidence bound."""
        ucb_scores = [arm.ucb_score for arm in self.arms]
        selected = int(np.argmax(ucb_scores))

        logger.debug(
            "UCB selected arm %d (lineup %d, ucb=%.3f)",
            selected, self.arms[selected].lineup_id, ucb_scores[selected],
        )
        return selected

    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the selected arm after observing a match outcome.

        Args:
            arm_index: Index of the arm that was played.
            reward: Observed reward (e.g., contest percentile, 0–1 normalized).
        """
        arm = self.arms[arm_index]
        arm.pulls += 1
        arm.total_reward += reward

        # Update Beta distribution parameters
        # Treat reward as success probability
        if reward > 0.5:
            arm.alpha += reward
        else:
            arm.beta += (1.0 - reward)

        self._history.append({
            "arm_index": arm_index,
            "lineup_id": arm.lineup_id,
            "reward": reward,
            "new_alpha": arm.alpha,
            "new_beta": arm.beta,
        })

        logger.info(
            "Updated arm %d: reward=%.3f, α=%.1f, β=%.1f, pulls=%d",
            arm_index, reward, arm.alpha, arm.beta, arm.pulls,
        )

    def get_recommendation(self) -> dict:
        """
        Get the final lineup recommendation with confidence metrics.

        Returns dict with lineup_id, confidence, and context.
        """
        if not self.arms:
            return {"error": "No arms initialized"}

        selected_idx = self.select()
        arm = self.arms[selected_idx]

        # Confidence: based on number of observations and reward stability
        confidence = min(arm.pulls / 10.0, 1.0) if arm.pulls > 0 else 0.5

        # For mega contests, prefer higher-variance arms
        if self.contest_type == ContestType.MEGA_CONTEST:
            variance_bonus = arm.alpha / (arm.alpha + arm.beta)
        else:
            variance_bonus = 0.0

        return {
            "lineup_id": arm.lineup_id,
            "arm_index": selected_idx,
            "confidence": confidence,
            "mean_reward": arm.mean_reward,
            "reward_score": arm.reward_score,
            "pulls": arm.pulls,
            "contest_type": self.contest_type.value,
            "variance_bonus": variance_bonus,
        }

    @property
    def history(self) -> list[dict]:
        return self._history
