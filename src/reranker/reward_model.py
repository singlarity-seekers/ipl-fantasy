"""
Reward model for team reranking.

Learns to predict contest performance (rank/payout) from lineup features,
going beyond simple expected-points maximization to capture what
actually wins in fantasy contests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class LineupFeatures:
    """Feature representation of a lineup for the reward model."""

    expected_total: float
    ceiling_total: float  # sum of 95th percentiles
    floor_total: float  # sum of 5th percentiles
    captain_expected: float
    captain_ceiling: float
    avg_consistency: float
    role_balance: float  # entropy of role distribution
    team_concentration: float  # max players from one team
    total_credits_used: float
    credits_remaining: float
    n_all_rounders: int
    n_differential_picks: int  # players with < 10% ownership
    captain_ownership: float
    upside_ratio: float  # ceiling / expected

    def to_array(self) -> np.ndarray:
        return np.array([
            self.expected_total,
            self.ceiling_total,
            self.floor_total,
            self.captain_expected,
            self.captain_ceiling,
            self.avg_consistency,
            self.role_balance,
            self.team_concentration,
            self.total_credits_used,
            self.credits_remaining,
            self.n_all_rounders,
            self.n_differential_picks,
            self.captain_ownership,
            self.upside_ratio,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "expected_total", "ceiling_total", "floor_total",
            "captain_expected", "captain_ceiling", "avg_consistency",
            "role_balance", "team_concentration", "total_credits_used",
            "credits_remaining", "n_all_rounders", "n_differential_picks",
            "captain_ownership", "upside_ratio",
        ]


class RewardModel:
    """
    Learned reward model that scores lineups based on historical
    contest outcomes.

    Trained on (lineup_features, contest_rank) pairs to learn what
    lineup characteristics correlate with high placements.
    """

    def __init__(self):
        self._model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(
        self,
        features: list[LineupFeatures],
        rewards: list[float],
    ) -> RewardModel:
        """
        Train the reward model.

        Args:
            features: List of LineupFeatures from historical contests.
            rewards: Corresponding reward signal (higher = better).
                     Could be: -rank, percentile, payout, or custom score.
        """
        X = np.array([f.to_array() for f in features])
        y = np.array(rewards)

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._fitted = True

        logger.info("Reward model fitted on %d samples", len(features))
        return self

    def score(self, features: LineupFeatures) -> float:
        """Score a single lineup. Higher = better predicted contest performance."""
        if not self._fitted:
            return self._heuristic_score(features)

        X = features.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict(X_scaled)[0])

    def rank_lineups(
        self,
        lineup_features: list[LineupFeatures],
    ) -> list[tuple[int, float]]:
        """
        Rank multiple lineups by predicted reward.

        Returns list of (original_index, score) sorted by score descending.
        """
        scores = [(i, self.score(f)) for i, f in enumerate(lineup_features)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _heuristic_score(self, features: LineupFeatures) -> float:
        """Fallback heuristic scoring when model is not fitted."""
        return (
            0.4 * features.expected_total
            + 0.3 * features.ceiling_total
            + 0.15 * features.captain_ceiling
            + 0.1 * features.avg_consistency * 100
            + 0.05 * features.n_all_rounders * 10
        )

    @property
    def feature_importance(self) -> dict[str, float]:
        """Return feature importances from the trained model."""
        if not self._fitted:
            return {}
        names = LineupFeatures.feature_names()
        importances = self._model.feature_importances_
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


def extract_lineup_features(
    lineup,  # OptimizedLineup
    sim_summary: pd.DataFrame | None = None,
    ownership_pcts: dict[str, float] | None = None,
) -> LineupFeatures:
    """
    Extract features from an OptimizedLineup for the reward model.

    Args:
        lineup: OptimizedLineup instance.
        sim_summary: Simulation summary DataFrame (optional, enriches features).
        ownership_pcts: {player: ownership%} (optional).
    """
    from src.config import CONSTRAINTS

    players = lineup.players
    ownership_pcts = ownership_pcts or {}

    expected_total = sum(p.expected_points for p in players)
    ceiling_total = sum(p.ceiling_95 for p in players)
    floor_total = sum(max(p.expected_points - p.ceiling_95 * 0.5, 0) for p in players)

    # Captain metrics
    cap_name = lineup.captain
    cap = next((p for p in players if p.name == cap_name), players[0])
    captain_expected = cap.expected_points
    captain_ceiling = cap.ceiling_95

    # Consistency from simulation
    avg_consistency = 0.5
    if sim_summary is not None and "consistency" in sim_summary.columns:
        player_names = [p.name for p in players]
        relevant = sim_summary[sim_summary["player"].isin(player_names)]
        if not relevant.empty:
            avg_consistency = float(relevant["consistency"].mean())

    # Role balance (entropy)
    role_counts = lineup.role_counts
    total = sum(role_counts.values())
    probs = [c / total for c in role_counts.values() if c > 0]
    role_balance = float(-sum(p * np.log2(p) for p in probs if p > 0))

    # Team concentration
    team_counts = lineup.team_counts
    team_concentration = max(team_counts.values()) if team_counts else 0

    # Credits
    total_credits = sum(p.credit_cost for p in players)

    # Differential picks
    n_differential = sum(
        1 for p in players if ownership_pcts.get(p.name, 10.0) < 10.0
    )

    # Captain ownership
    captain_ownership = ownership_pcts.get(cap_name, 10.0)

    return LineupFeatures(
        expected_total=expected_total,
        ceiling_total=ceiling_total,
        floor_total=floor_total,
        captain_expected=captain_expected,
        captain_ceiling=captain_ceiling,
        avg_consistency=avg_consistency,
        role_balance=role_balance,
        team_concentration=team_concentration,
        total_credits_used=total_credits,
        credits_remaining=CONSTRAINTS.total_credits - total_credits,
        n_all_rounders=role_counts.get("AR", 0),
        n_differential_picks=n_differential,
        captain_ownership=captain_ownership,
        upside_ratio=ceiling_total / max(expected_total, 1.0),
    )
