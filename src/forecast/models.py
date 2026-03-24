"""
Probabilistic player performance forecasting for IPL Fantasy.

HOW IT WORKS:
-------------
1. Fit on historical scorecards (player performance data from past IPL seasons)
2. For each player, calculate:
   - Expected points: blend of career average + recent form (last 5 matches)
   - Uncertainty (std): standard deviation of historical fantasy scores
   - Percentiles: 5th, 10th, 50th, 90th, 95th from empirical distribution

3. Output: PlayerForecast object containing the full distribution

UNCERTAINTY QUANTIFICATION:
--------------------------
- For known players (3+ matches): std comes from actual historical variance
  Example: If Kohli scored [45, 32, 68, 12, 55] in last 5 matches,
  std = np.std([45, 32, 68, 12, 55]) = ±21 points

- For cold-start players (no history): use role-based priors
  BOWL: 35±25, BAT: 32±24, AR: 28±20, WK: 18±14

MONTE CARLO SAMPLING:
--------------------
The sample() method draws from either:
  - Empirical: bootstrap from historical values (rng.choice(history))
  - Normal: truncated normal with (mean, std) clipped at 0 (can't score negative)

This uncertainty feeds into the Monte Carlo simulation to generate
point distributions, not just single expected values.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class PlayerForecast:
    """Probabilistic forecast for a single player in a match."""

    player: str
    role: str
    expected_points: float
    std_points: float
    quantile_10: float
    quantile_50: float
    quantile_90: float
    ceiling_95: float  # 95th percentile — upside metric
    floor_5: float  # 5th percentile — downside risk

    # Optional: full distribution (for Monte Carlo sampling)
    distribution_params: dict = field(default_factory=dict)

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Draw n samples from this player's forecast distribution.

        Uses a truncated normal by default (fantasy points ≥ 0).
        """
        rng = rng or np.random.default_rng()

        if self.distribution_params.get("type") == "empirical":
            # Resample from historical observations
            samples = rng.choice(
                self.distribution_params["values"],
                size=n,
                replace=True,
            )
        else:
            # Truncated normal (clipped at 0)
            samples = rng.normal(self.expected_points, max(self.std_points, 1.0), size=n)
            samples = np.clip(samples, 0, None)

        return samples


class PlayerForecaster:
    """
    Forecasts IPL Fantasy fantasy points for players in upcoming matches.

    Uses historical scorecards to build per-player distributions.
    Can be upgraded to use XGBoost with contextual features.
    """

    # Default path to player alias mapping (ESPNcricinfo → Cricsheet names)
    _ALIAS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "player_aliases.json"

    def __init__(self, alias_path: Path | str | None = None):
        self._player_history: dict[str, np.ndarray] = {}
        self._player_roles: dict[str, str] = {}
        self._global_mean: float = 30.0
        self._global_std: float = 25.0
        self._aliases: dict[str, str] = {}  # ESPN name → Cricsheet name
        self._load_aliases(alias_path)

    def fit(self, scorecards: pd.DataFrame) -> PlayerForecaster:
        """
        Fit the forecaster on historical player scorecards.

        Args:
            scorecards: DataFrame with columns [player, role, fantasy_points, ...]
        """
        for player, group in scorecards.groupby("player"):
            points = group["fantasy_points"].values
            self._player_history[player] = points

            # Most common role
            if "role" in group.columns:
                self._player_roles[player] = group["role"].mode().iloc[0]

        # Global baseline
        all_points = scorecards["fantasy_points"].dropna()
        self._global_mean = float(all_points.mean())
        self._global_std = float(all_points.std())

        logger.info(
            "Fitted forecaster on %d players (global μ=%.1f, σ=%.1f), %d aliases loaded",
            len(self._player_history),
            self._global_mean,
            self._global_std,
            len(self._aliases),
        )
        return self

    def forecast(
        self,
        player: str,
        role: str | None = None,
        venue: str | None = None,
        opposition: str | None = None,
        recent_form_weight: float = 0.6,
    ) -> PlayerForecast:
        """
        Generate a probabilistic forecast for a player.

        Args:
            player: Player name.
            role: Override role (WK/BAT/AR/BOWL). Uses fitted role if None.
            venue: Venue name for conditioning (future enhancement).
            opposition: Opposition team for conditioning (future enhancement).
            recent_form_weight: Weight for recent matches vs career avg.
        """
        # Resolve ESPNcricinfo name → Cricsheet historical name
        lookup_name = self._resolve_alias(player)
        role = role or self._player_roles.get(lookup_name, "BAT")
        history = self._player_history.get(lookup_name)

        if history is not None and len(history) >= 3:
            # Blend career avg with recent form
            career_mean = float(np.mean(history))
            career_std = float(np.std(history))

            recent = history[-5:] if len(history) >= 5 else history
            recent_mean = float(np.mean(recent))

            expected = recent_form_weight * recent_mean + (1 - recent_form_weight) * career_mean
            std = max(career_std, 5.0)  # floor on uncertainty

            quantiles = np.percentile(history, [5, 10, 50, 90, 95])

            return PlayerForecast(
                player=player,
                role=role,
                expected_points=expected,
                std_points=std,
                floor_5=float(quantiles[0]),
                quantile_10=float(quantiles[1]),
                quantile_50=float(quantiles[2]),
                quantile_90=float(quantiles[3]),
                ceiling_95=float(quantiles[4]),
                distribution_params={"type": "empirical", "values": history},
            )
        else:
            # Cold start: use global priors
            return self._cold_start_forecast(player, role)

    def forecast_match(
        self,
        players: list[str],
        roles: dict[str, str] | None = None,
        venue: str | None = None,
    ) -> list[PlayerForecast]:
        """
        Forecast all players for a specific match.

        Args:
            players: List of player names in the match.
            roles: Optional {player: role} mapping.
            venue: Venue for conditioning.

        Returns:
            List of PlayerForecast objects.
        """
        roles = roles or {}
        forecasts = []
        for p in players:
            role = roles.get(p)
            forecasts.append(self.forecast(p, role=role, venue=venue))
        return forecasts

    def _cold_start_forecast(self, player: str, role: str) -> PlayerForecast:
        """Generate a prior-based forecast for an unknown/new player."""
        # Role-based priors (typical uncapped player ranges)
        role_priors = {
            "BAT": (20.0, 15.0),
            "BOWL": (22.0, 18.0),
            "AR": (28.0, 20.0),
            "WK": (18.0, 14.0),
        }
        mean, std = role_priors.get(role, (20.0, 15.0))

        return PlayerForecast(
            player=player,
            role=role,
            expected_points=mean,
            std_points=std,
            floor_5=max(mean - 2 * std, 0),
            quantile_10=max(mean - 1.3 * std, 0),
            quantile_50=mean,
            quantile_90=mean + 1.3 * std,
            ceiling_95=mean + 2 * std,
            distribution_params={"type": "normal", "mean": mean, "std": std},
        )

    def _load_aliases(self, alias_path: Path | str | None = None) -> None:
        """Load player name aliases from JSON."""
        path = Path(alias_path) if alias_path else self._ALIAS_PATH
        if path.exists():
            try:
                with open(path) as f:
                    raw = json.load(f)
                # Filter out the _comment key
                self._aliases = {k: v for k, v in raw.items() if not k.startswith("_")}
                logger.debug("Loaded %d player aliases from %s", len(self._aliases), path)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load player aliases: %s", e)

    def _resolve_alias(self, espn_name: str) -> str:
        """Resolve an ESPNcricinfo name to the Cricsheet historical name."""
        return self._aliases.get(espn_name, espn_name)


class XGBoostForecaster(PlayerForecaster):
    """
    XGBoost-based forecaster that uses contextual features
    (venue, opposition, recent form, pitch conditions) to predict
    fantasy points more accurately.

    Upgrade path from the base statistical forecaster.
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._feature_columns: list[str] = []

    def fit(self, scorecards: pd.DataFrame) -> XGBoostForecaster:
        """
        Fit XGBoost on engineered features.

        Expected feature columns in scorecards:
            rolling_5_avg_fp, rolling_10_avg_fp, ewm_avg_fp,
            venue_avg_fp, career_avg_fp, matches_played, ...
        """
        super().fit(scorecards)

        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed; falling back to statistical forecaster")
            return self

        # Define feature columns (must be present in scorecards)
        candidate_features = [
            "rolling_5_avg_fp",
            "rolling_10_avg_fp",
            "ewm_avg_fp",
            "avg_runs",
            "avg_wickets",
            "matches_played",
        ]
        self._feature_columns = [c for c in candidate_features if c in scorecards.columns]

        if not self._feature_columns:
            logger.warning("No feature columns found; XGBoost not fitted")
            return self

        train = scorecards.dropna(subset=self._feature_columns + ["fantasy_points"])
        if len(train) < 50:
            logger.warning("Too few training samples (%d); XGBoost not fitted", len(train))
            return self

        X = train[self._feature_columns].values
        y = train["fantasy_points"].values

        self._model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model.fit(X, y)

        logger.info(
            "XGBoost fitted on %d samples with %d features",
            len(train),
            len(self._feature_columns),
        )
        return self
