"""
Monte Carlo match simulation engine.

PURPOSE:
--------
Transform probabilistic forecasts into actionable distributions.
Instead of just saying "Kohli scores 45 points on average",
we generate 10,000 possible outcomes showing the full range:
- Floor (5th percentile): 20 points (worst case)
- Ceiling (95th percentile): 75 points (best case)
- Probability of 50+ points: 35%

HOW IT WORKS:
-------------
1. For each of N=10,000 simulations:
   - Sample fantasy points for each player from their forecast distribution
   - This uses the std (uncertainty) calculated in PlayerForecaster
   - High std = wider spread of samples = more volatile outcomes

2. Aggregate across all simulations:
   - Build (players × simulations) matrix
   - Calculate percentiles, probabilities, consistency metrics

WHY THIS MATTERS:
----------------
Fantasy scoring is inherently uncertain. A player with mean=40, std=25
is very different from mean=40, std=8, even though both "expect" 40 pts.
- High std = boom-or-bust (good for differential captains)
- Low std = consistent floor (good for safe lineups)

The optimizer uses these distributions to pick lineups that maximize
expected points while managing risk (floor) and upside (ceiling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import SIM_CONFIG
from src.forecast.models import PlayerForecast

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation for a match."""

    # (n_players, n_simulations) matrix of fantasy points
    points_matrix: np.ndarray
    player_names: list[str]
    player_roles: list[str]
    n_simulations: int

    # Per-player summary statistics
    summary: pd.DataFrame = field(default=None)

    def __post_init__(self):
        if self.summary is None:
            self.summary = self._build_summary()

    def _build_summary(self) -> pd.DataFrame:
        """Compute summary statistics from the simulation matrix."""
        records = []
        for i, (name, role) in enumerate(zip(self.player_names, self.player_roles)):
            pts = self.points_matrix[i]
            records.append(
                {
                    "player": name,
                    "role": role,
                    "mean_fp": float(np.mean(pts)),
                    "median_fp": float(np.median(pts)),
                    "std_fp": float(np.std(pts)),
                    "floor_5": float(np.percentile(pts, 5)),
                    "q10": float(np.percentile(pts, 10)),
                    "q25": float(np.percentile(pts, 25)),
                    "q75": float(np.percentile(pts, 75)),
                    "q90": float(np.percentile(pts, 90)),
                    "ceiling_95": float(np.percentile(pts, 95)),
                    "max_fp": float(np.max(pts)),
                    "prob_30plus": float(np.mean(pts >= 30)),
                    "prob_50plus": float(np.mean(pts >= 50)),
                    "prob_75plus": float(np.mean(pts >= 75)),
                    "consistency": float(1.0 / (1.0 + np.std(pts))),  # higher = more consistent
                    "upside_ratio": float(np.percentile(pts, 95) / max(np.mean(pts), 1)),
                }
            )
        return pd.DataFrame(records)

    def get_player_distribution(self, player: str) -> np.ndarray:
        """Get the full simulation distribution for a player."""
        idx = self.player_names.index(player)
        return self.points_matrix[idx]

    def rank_by(self, metric: str = "mean_fp", ascending: bool = False) -> pd.DataFrame:
        """Rank players by a summary metric."""
        return self.summary.sort_values(metric, ascending=ascending).reset_index(drop=True)


class MonteCarloSimulator:
    """
    Monte Carlo match simulation engine.

    For each simulation:
      1. Sample fantasy points for each player from their forecast distribution
      2. Record the sampled points
      3. Aggregate across N simulations → point distributions
    """

    def __init__(
        self,
        n_simulations: int | None = None,
        random_seed: int | None = None,
    ):
        self.n_simulations = n_simulations or SIM_CONFIG.n_simulations
        self.rng = np.random.default_rng(
            random_seed if random_seed is not None else SIM_CONFIG.random_seed
        )

    def simulate_match(
        self,
        forecasts: list[PlayerForecast],
        venue_adjustment: dict[str, float] | None = None,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulations for a match.

        Args:
            forecasts: List of PlayerForecast objects for all players.
            venue_adjustment: Optional {player: multiplier} for venue conditioning.

        Returns:
            SimulationResult with the full (n_players × n_simulations) matrix.
        """
        n_players = len(forecasts)
        n_sims = self.n_simulations

        logger.info(
            "Running %d simulations for %d players",
            n_sims,
            n_players,
        )

        # Pre-allocate points matrix
        points_matrix = np.zeros((n_players, n_sims))

        for i, forecast in enumerate(forecasts):
            # Sample from player's distribution
            samples = forecast.sample(n=n_sims, rng=self.rng)

            # Apply venue adjustment if provided
            if venue_adjustment and forecast.player in venue_adjustment:
                adjustment = venue_adjustment[forecast.player]
                samples = samples * adjustment

            points_matrix[i] = samples

        result = SimulationResult(
            points_matrix=points_matrix,
            player_names=[f.player for f in forecasts],
            player_roles=[f.role for f in forecasts],
            n_simulations=n_sims,
        )

        logger.info(
            "Simulation complete. Top player: %s (μ=%.1f, σ=%.1f)",
            result.summary.iloc[result.summary["mean_fp"].argmax()]["player"],
            result.summary["mean_fp"].max(),
            result.summary.loc[result.summary["mean_fp"].argmax(), "std_fp"],
        )

        return result

    def simulate_with_correlations(
        self,
        forecasts: list[PlayerForecast],
        correlation_matrix: np.ndarray | None = None,
    ) -> SimulationResult:
        """
        Simulate with player correlations (e.g., opening pair batsmen
        are positively correlated; bowler and opposing batsman are
        negatively correlated).

        Uses Cholesky decomposition on correlated normal samples,
        then maps to each player's marginal distribution.
        """
        n_players = len(forecasts)
        n_sims = self.n_simulations

        if correlation_matrix is None:
            # Default: independent simulations
            return self.simulate_match(forecasts)

        # Generate correlated standard normal samples
        L = np.linalg.cholesky(correlation_matrix)
        z = self.rng.standard_normal((n_players, n_sims))
        correlated_z = L @ z

        # Map to each player's distribution using inverse CDF
        from scipy.stats import norm

        points_matrix = np.zeros((n_players, n_sims))
        for i, forecast in enumerate(forecasts):
            # Convert correlated z-scores to uniform [0,1]
            u = norm.cdf(correlated_z[i])

            # Map through empirical distribution or parametric
            if forecast.distribution_params.get("type") == "empirical":
                values = np.sort(forecast.distribution_params["values"])
                # Quantile mapping
                indices = (u * (len(values) - 1)).astype(int)
                indices = np.clip(indices, 0, len(values) - 1)
                points_matrix[i] = values[indices]
            else:
                # Parametric: truncated normal
                mu = forecast.expected_points
                sigma = max(forecast.std_points, 1.0)
                points_matrix[i] = np.clip(norm.ppf(u, mu, sigma), 0, None)

        return SimulationResult(
            points_matrix=points_matrix,
            player_names=[f.player for f in forecasts],
            player_roles=[f.role for f in forecasts],
            n_simulations=n_sims,
        )

    def compute_lineup_scores(
        self,
        result: SimulationResult,
        lineup_indices: list[int],
        captain_idx: int | None = None,
        vc_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute total lineup scores across all simulations.

        Args:
            result: SimulationResult from a simulation.
            lineup_indices: Indices into result.player_names for the 11 players.
            captain_idx: Index of captain (2x multiplier).
            vc_idx: Index of vice-captain (1.5x multiplier).

        Returns:
            Array of shape (n_simulations,) with total lineup scores.
        """
        lineup_points = result.points_matrix[lineup_indices].copy()

        if captain_idx is not None:
            cap_lineup_pos = lineup_indices.index(captain_idx)
            lineup_points[cap_lineup_pos] *= 2.0

        if vc_idx is not None:
            vc_lineup_pos = lineup_indices.index(vc_idx)
            lineup_points[vc_lineup_pos] *= 1.5

        return lineup_points.sum(axis=0)
