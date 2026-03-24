"""
Simulation-aware Captain and Vice-Captain selector.

Picks captain (2x points) and vice-captain (1.5x) based on
simulation distributions rather than raw expected value alone.

Strategies:
  - Safe: maximize expected value with multipliers
  - Differential: pick underowned high-upside captain
  - Contrarian: deliberately avoid popular picks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.simulation.monte_carlo import SimulationResult

logger = logging.getLogger(__name__)


class CaptainStrategy(Enum):
    SAFE = "safe"
    DIFFERENTIAL = "differential"
    CONTRARIAN = "contrarian"


@dataclass
class CaptainPick:
    """Captain/VC recommendation with supporting metrics."""

    captain: str
    vice_captain: str
    captain_score: float
    vc_score: float
    strategy: CaptainStrategy
    captain_metrics: dict
    vc_metrics: dict


class CaptainSelector:
    """
    Selects captain and vice-captain for a IPL Fantasy lineup
    using Monte Carlo simulation data.
    """

    def __init__(
        self,
        alpha: float = 0.5,   # weight on E[points]
        beta: float = 0.3,    # weight on ceiling (95th %ile)
        gamma: float = 0.2,   # weight on consistency
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def select(
        self,
        lineup_players: list[str],
        sim_result: SimulationResult,
        strategy: CaptainStrategy = CaptainStrategy.SAFE,
        ownership_pcts: dict[str, float] | None = None,
    ) -> CaptainPick:
        """
        Select optimal captain and vice-captain.

        Args:
            lineup_players: Names of players in the lineup (11 players).
            sim_result: Monte Carlo simulation results.
            strategy: Selection strategy.
            ownership_pcts: Optional {player: ownership%} for differential picks.

        Returns:
            CaptainPick with captain, VC, and metrics.
        """
        # Get simulation summary for lineup players only
        summary = sim_result.summary[
            sim_result.summary["player"].isin(lineup_players)
        ].copy()

        if summary.empty:
            raise ValueError("No simulation data for lineup players")

        # Compute captain scores based on strategy
        if strategy == CaptainStrategy.SAFE:
            scores = self._safe_scores(summary, sim_result)
        elif strategy == CaptainStrategy.DIFFERENTIAL:
            scores = self._differential_scores(summary, sim_result, ownership_pcts)
        elif strategy == CaptainStrategy.CONTRARIAN:
            scores = self._contrarian_scores(summary, sim_result, ownership_pcts)
        else:
            scores = self._safe_scores(summary, sim_result)

        # Sort by composite score
        scores = scores.sort_values("captain_score", ascending=False)

        captain = scores.iloc[0]["player"]

        # VC: best remaining player (exclude captain)
        vc_candidates = scores[scores["player"] != captain]
        vice_captain = vc_candidates.iloc[0]["player"]

        captain_row = scores[scores["player"] == captain].iloc[0]
        vc_row = scores[scores["player"] == vice_captain].iloc[0]

        return CaptainPick(
            captain=captain,
            vice_captain=vice_captain,
            captain_score=float(captain_row["captain_score"]),
            vc_score=float(vc_row["captain_score"]),
            strategy=strategy,
            captain_metrics=captain_row.to_dict(),
            vc_metrics=vc_row.to_dict(),
        )

    def _safe_scores(
        self,
        summary: pd.DataFrame,
        sim_result: SimulationResult,
    ) -> pd.DataFrame:
        """
        Safe strategy: maximize expected multiplied points.

        Score = α · E[2x · points] + β · ceiling_95 + γ · consistency
        """
        df = summary.copy()

        # E[2x · points] = 2 * E[points]
        df["e_2x"] = 2.0 * df["mean_fp"]

        # Normalize components
        max_e2x = df["e_2x"].max() or 1.0
        max_ceiling = df["ceiling_95"].max() or 1.0
        max_consistency = df["consistency"].max() or 1.0

        df["captain_score"] = (
            self.alpha * (df["e_2x"] / max_e2x)
            + self.beta * (df["ceiling_95"] / max_ceiling)
            + self.gamma * (df["consistency"] / max_consistency)
        )

        return df

    def _differential_scores(
        self,
        summary: pd.DataFrame,
        sim_result: SimulationResult,
        ownership_pcts: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Differential strategy: pick high-upside captains that are
        under-owned. Ideal for mega-contests.
        """
        df = summary.copy()

        # Upside metric: how often does this player finish as top scorer?
        df["top_scorer_pct"] = 0.0
        for _, row in df.iterrows():
            player = row["player"]
            if player in sim_result.player_names:
                player_sims = sim_result.get_player_distribution(player)
                # Compare each player's sims to max across all lineup players
                lineup_indices = [
                    sim_result.player_names.index(p)
                    for p in df["player"].values
                    if p in sim_result.player_names
                ]
                all_sims = sim_result.points_matrix[lineup_indices]
                is_top = player_sims >= all_sims.max(axis=0)
                df.loc[df["player"] == player, "top_scorer_pct"] = float(is_top.mean())

        # Ownership penalty (lower ownership = higher differential value)
        if ownership_pcts:
            df["ownership"] = df["player"].map(ownership_pcts).fillna(10.0)
        else:
            df["ownership"] = 10.0  # assume 10% if unknown

        df["differential_bonus"] = 100.0 / (df["ownership"] + 1.0)

        # Composite score
        df["captain_score"] = (
            0.3 * df["mean_fp"] / (df["mean_fp"].max() or 1.0)
            + 0.3 * df["ceiling_95"] / (df["ceiling_95"].max() or 1.0)
            + 0.2 * df["top_scorer_pct"]
            + 0.2 * df["differential_bonus"] / (df["differential_bonus"].max() or 1.0)
        )

        return df

    def _contrarian_scores(
        self,
        summary: pd.DataFrame,
        sim_result: SimulationResult,
        ownership_pcts: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Contrarian strategy: deliberately avoid the most popular picks.
        Only viable for large GPP contests.
        """
        df = self._differential_scores(summary, sim_result, ownership_pcts)

        # Heavy penalty for high-ownership players
        if ownership_pcts:
            df["ownership_penalty"] = df["player"].map(ownership_pcts).fillna(10.0)
        else:
            df["ownership_penalty"] = 10.0

        df["captain_score"] = df["captain_score"] - 0.3 * (
            df["ownership_penalty"] / (df["ownership_penalty"].max() or 1.0)
        )

        return df

    def evaluate_captain_impact(
        self,
        sim_result: SimulationResult,
        lineup_players: list[str],
    ) -> pd.DataFrame:
        """
        For each candidate captain in the lineup, compute:
          - E[total lineup score with this captain (2x)]
          - P(top score) with this captain
          - Risk (downside variance)

        Returns DataFrame ranked by expected total.
        """
        lineup_indices = [
            sim_result.player_names.index(p)
            for p in lineup_players
            if p in sim_result.player_names
        ]

        records = []
        for player in lineup_players:
            if player not in sim_result.player_names:
                continue

            cap_idx = sim_result.player_names.index(player)

            # Compute lineup score with this player as captain
            from src.simulation.monte_carlo import MonteCarloSimulator
            lineup_scores = MonteCarloSimulator().compute_lineup_scores(
                sim_result,
                lineup_indices,
                captain_idx=cap_idx,
            )

            records.append({
                "player": player,
                "e_total_as_captain": float(np.mean(lineup_scores)),
                "std_total_as_captain": float(np.std(lineup_scores)),
                "p90_total": float(np.percentile(lineup_scores, 90)),
                "p10_total": float(np.percentile(lineup_scores, 10)),
            })

        return pd.DataFrame(records).sort_values("e_total_as_captain", ascending=False)
