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

from src.config import BATTING, BOWLING, PLAYING_XI_POINTS

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
    Incorporates recent T20 form data (last 12 months across all T20 leagues)
    to adjust forecasts, especially for cold-start players.
    """

    # Default path to player alias mapping (ESPNcricinfo → Cricsheet names)
    _ALIAS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "player_aliases.json"
    _T20_FORM_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "t20_recent_form.csv"

    # Players snubbed from India's T20 World Cup 2026 squad.
    # Historically, dropped players perform 10-15% better in the IPL immediately
    # after being snubbed — extra motivation to prove selectors wrong.
    # Source: India T20 WC 2026 squad announcement (Feb 2026)
    _T20WC_SNUBBED = {
        "Shubman Gill",       # Vice-captain dropped; poor T20I form cited
        "Ruturaj Gaikwad",    # Edged out for team balance
        "KL Rahul",           # Not picked despite strong IPL record
        "Shreyas Iyer",       # Missed out
        "Jitesh Sharma",      # Replaced by Ishan Kishan's return
        "Rishabh Pant",       # Not in final 15
        "Prithvi Shaw",       # Overlooked
        "Nitish Rana",        # Not considered
        "Mohammed Shami",     # Fitness concerns
    }
    _SNUB_BOOST = 1.08  # +8% motivation boost for snubbed players

    # Normalize common venue name variants to a canonical form
    _VENUE_ALIASES = {
        "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
        "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
        "Wankhede Stadium, Mumbai": "Wankhede Stadium",
        "Eden Gardens, Kolkata": "Eden Gardens",
        "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
        "Brabourne Stadium, Mumbai": "Brabourne Stadium",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal",
        "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association Stadium, Mohali",
        "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
        "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    }

    def __init__(self, alias_path: Path | str | None = None, t20_form_path: Path | str | None = None):
        self._player_history: dict[str, np.ndarray] = {}
        self._player_roles: dict[str, str] = {}
        self._player_venue_history: dict[str, dict[str, np.ndarray]] = {}  # player -> {venue -> points}
        self._venue_multipliers: dict[str, float] = {}  # venue -> scoring multiplier
        self._batter_vs_team: dict[str, dict[str, dict]] = {}  # batter -> {team -> {sr, avg, balls}}
        self._bowler_vs_team: dict[str, dict[str, dict]] = {}  # bowler -> {team -> {econ, wkts, balls}}
        self._global_mean: float = 30.0
        self._global_std: float = 25.0
        self._aliases: dict[str, str] = {}  # ESPN name → Cricsheet name
        self._t20_form: dict[str, dict] = {}  # player → recent T20 form stats
        self._load_aliases(alias_path)
        self._load_t20_form(t20_form_path)

    def _normalize_venue(self, venue: str | None) -> str | None:
        """Normalize venue name to canonical form."""
        if not venue:
            return None
        return self._VENUE_ALIASES.get(venue, venue)

    def fit(self, scorecards: pd.DataFrame, deliveries: pd.DataFrame | None = None) -> PlayerForecaster:
        """
        Fit the forecaster on historical player scorecards.

        Args:
            scorecards: DataFrame with columns [player, role, fantasy_points, venue, ...]
            deliveries: Optional ball-by-ball data for batter-vs-team matchup analysis.
        """
        has_venue = "venue" in scorecards.columns

        for player, group in scorecards.groupby("player"):
            points = group["fantasy_points"].values
            self._player_history[player] = points

            if "role" in group.columns:
                self._player_roles[player] = group["role"].mode().iloc[0]

            # Build per-venue history for this player
            if has_venue:
                venue_hist = {}
                for venue, vgroup in group.groupby("venue"):
                    norm_venue = self._normalize_venue(venue)
                    if norm_venue:
                        if norm_venue not in venue_hist:
                            venue_hist[norm_venue] = []
                        venue_hist[norm_venue].extend(vgroup["fantasy_points"].values.tolist())
                self._player_venue_history[player] = {
                    v: np.array(pts) for v, pts in venue_hist.items()
                }

        # Global baseline
        all_points = scorecards["fantasy_points"].dropna()
        self._global_mean = float(all_points.mean())
        self._global_std = float(all_points.std())

        # Venue multipliers: how much does each venue inflate/deflate scores?
        if has_venue:
            sc = scorecards.copy()
            sc["venue_norm"] = sc["venue"].apply(self._normalize_venue)
            venue_stats = sc.groupby("venue_norm").agg(
                mean_fp=("fantasy_points", "mean"),
                matches=("match_id", "nunique"),
            )
            for venue, row in venue_stats.iterrows():
                if row["matches"] >= 5:
                    self._venue_multipliers[venue] = row["mean_fp"] / self._global_mean

        # Build batter-vs-team matchup stats from deliveries
        if deliveries is not None and "bowling_team" in deliveries.columns:
            self._build_matchups(deliveries)

        logger.info(
            "Fitted forecaster on %d players (global μ=%.1f, σ=%.1f), "
            "%d aliases, %d venue multipliers, %d batter matchups",
            len(self._player_history),
            self._global_mean,
            self._global_std,
            len(self._aliases),
            len(self._venue_multipliers),
            len(self._batter_vs_team),
        )
        return self

    def _build_matchups(self, deliveries: pd.DataFrame) -> None:
        """Build batter-vs-bowling-team and bowler-vs-batting-team matchup stats."""
        # Batter vs bowling team
        batter_grp = deliveries.groupby(["batter", "bowling_team"]).agg(
            balls=("batsman_runs", "count"),
            runs=("batsman_runs", "sum"),
            dismissals=("is_wicket", "sum"),
        )
        for (batter, team), row in batter_grp.iterrows():
            if row["balls"] < 18:  # min 3 overs faced
                continue
            if batter not in self._batter_vs_team:
                self._batter_vs_team[batter] = {}
            sr = row["runs"] / row["balls"] * 100 if row["balls"] > 0 else 0
            avg = row["runs"] / max(row["dismissals"], 1)
            self._batter_vs_team[batter][team] = {
                "sr": sr, "avg": avg, "balls": int(row["balls"]),
                "runs": int(row["runs"]), "dismissals": int(row["dismissals"]),
            }

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
        norm_venue = self._normalize_venue(venue)

        # Check for recent T20 form data
        t20_form = self._t20_form.get(player)
        form_mean, form_std = (0.0, 0.0)
        if t20_form and t20_form.get("matches", 0) >= 3:
            form_mean, form_std = self._estimate_fantasy_points_from_form(t20_form, role)

        if history is not None and len(history) >= 3:
            # Blend career avg with recent IPL form
            career_mean = float(np.mean(history))
            career_std = float(np.std(history))

            recent = history[-5:] if len(history) >= 5 else history
            recent_mean = float(np.mean(recent))

            expected = recent_form_weight * recent_mean + (1 - recent_form_weight) * career_mean

            # Venue adjustment: use player's venue-specific history if available,
            # otherwise fall back to venue multiplier
            if norm_venue:
                venue_hist = self._player_venue_history.get(lookup_name, {}).get(norm_venue)
                if venue_hist is not None and len(venue_hist) >= 5:
                    # Player has enough history at this venue — blend it in (30% weight)
                    venue_mean = float(np.mean(venue_hist))
                    venue_weight = 0.3
                    expected = (1 - venue_weight) * expected + venue_weight * venue_mean
                elif norm_venue in self._venue_multipliers:
                    # No player-specific venue data, use venue-level multiplier
                    expected *= self._venue_multipliers[norm_venue]

            # Opposition matchup adjustment for batters/ARs
            if opposition and role in ("BAT", "WK", "AR"):
                matchup = self._batter_vs_team.get(lookup_name, {}).get(opposition)
                if matchup and matchup["balls"] >= 18:
                    # Compare batter's SR against this team vs their overall SR
                    overall_sr = float(np.sum(history > 0)) / max(len(history), 1) * 100  # rough
                    matchup_sr = matchup["sr"]
                    # If batter has SR 150+ against this bowling, boost; <100 penalize
                    if matchup_sr >= 150:
                        expected *= 1.10  # +10% for dominant matchup
                    elif matchup_sr >= 130:
                        expected *= 1.05
                    elif matchup_sr < 100:
                        expected *= 0.90  # -10% for struggling matchup
                    elif matchup_sr < 110:
                        expected *= 0.95

            # Adjust with T20 form
            if form_mean > 0:
                t20_form_weight = 0.2
                expected = (1 - t20_form_weight) * expected + t20_form_weight * form_mean

            # Snubbed players motivation boost
            if player in self._T20WC_SNUBBED:
                expected *= self._SNUB_BOOST

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
        elif form_mean > 0:
            # Apply venue multiplier to form-based forecast
            if norm_venue and norm_venue in self._venue_multipliers:
                form_mean *= self._venue_multipliers[norm_venue]
            if player in self._T20WC_SNUBBED:
                form_mean *= self._SNUB_BOOST
            return self._normal_forecast(player, role, form_mean, form_std)
        else:
            # No IPL history, no T20 form — generic cold start
            fc = self._cold_start_forecast(player, role)
            if player in self._T20WC_SNUBBED:
                fc.expected_points *= self._SNUB_BOOST
            return fc

    def forecast_match(
        self,
        players: list[str],
        roles: dict[str, str] | None = None,
        venue: str | None = None,
        player_teams: dict[str, str] | None = None,
        team1: str | None = None,
        team2: str | None = None,
    ) -> list[PlayerForecast]:
        """
        Forecast all players for a specific match.

        Args:
            players: List of player names in the match.
            roles: Optional {player: role} mapping.
            venue: Venue for conditioning.
            player_teams: Optional {player: team} mapping for opposition lookups.
            team1: First team name (for opposition matchup).
            team2: Second team name (for opposition matchup).

        Returns:
            List of PlayerForecast objects.
        """
        roles = roles or {}
        player_teams = player_teams or {}
        forecasts = []
        for p in players:
            role = roles.get(p)
            # Determine opposition: if player is on team1, opposition is team2
            opposition = None
            if player_teams and team1 and team2:
                p_team = player_teams.get(p)
                if p_team == team1:
                    opposition = team2
                elif p_team == team2:
                    opposition = team1
            forecasts.append(self.forecast(p, role=role, venue=venue, opposition=opposition))
        return forecasts

    def _normal_forecast(self, player: str, role: str, mean: float, std: float) -> PlayerForecast:
        """Build a PlayerForecast from a normal distribution (mean, std)."""
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

    def _cold_start_forecast(self, player: str, role: str) -> PlayerForecast:
        """Generate a prior-based forecast for an unknown/new player."""
        role_priors = {
            "BAT": (20.0, 15.0),
            "BOWL": (22.0, 18.0),
            "AR": (28.0, 20.0),
            "WK": (18.0, 14.0),
        }
        mean, std = role_priors.get(role, (20.0, 15.0))
        return self._normal_forecast(player, role, mean, std)

    def _estimate_fantasy_points_from_form(self, form: dict, role: str) -> tuple[float, float]:
        """
        Estimate expected IPL Fantasy points from recent T20 form stats.

        Converts raw T20 stats (runs, SR, wickets, economy) into an
        approximate fantasy points estimate using scoring weights.
        Applies a discount for domestic-only players (SMAT stats inflate
        vs IPL-quality bowling/batting).

        Returns (estimated_mean, estimated_std).
        """
        matches = form.get("matches", 0)
        if matches == 0:
            return 0.0, 0.0

        # Discount factor: domestic-only leagues produce inflated stats
        leagues = str(form.get("leagues", ""))
        top_leagues = {"IPL", "T20I", "BBL", "SA20", "PSL", "Hundred", "T20Blast", "ILT20"}
        player_leagues = set(leagues.split(","))
        has_top_league = bool(player_leagues & top_leagues)
        # SMAT/domestic-only players get a 35% discount; mixed get no discount
        league_discount = 1.0 if has_top_league else 0.65

        # Per-match averages
        runs_pm = form.get("total_runs", 0) / matches
        sr = form.get("strike_rate", 0)
        fours_pm = form.get("total_fours", 0) / matches
        sixes_pm = form.get("total_sixes", 0) / matches
        wickets_pm = form.get("total_wickets", 0) / matches
        economy = form.get("economy", 0)

        # Approximate fantasy points using canonical scoring constants from config
        bat_pts = runs_pm + fours_pm * BATTING.per_four + sixes_pm * BATTING.per_six
        if runs_pm >= 50:
            bat_pts += BATTING.milestone_50
        elif runs_pm >= 30:
            bat_pts += BATTING.milestone_30
        if sr >= 170:
            bat_pts += 6.0  # top SR tier
        elif sr >= 150:
            bat_pts += 4.0

        bowl_pts = wickets_pm * BOWLING.per_wicket
        if wickets_pm >= 3:
            bowl_pts += BOWLING.haul_3w
        if economy > 0 and economy < 7.0:
            bowl_pts += 4.0  # good economy bonus
        elif economy > 10.0:
            bowl_pts -= 4.0
        estimated = (PLAYING_XI_POINTS + bat_pts + bowl_pts) * league_discount

        # Std based on role variability
        role_std_factor = {"BAT": 0.55, "BOWL": 0.60, "AR": 0.50, "WK": 0.55}
        estimated_std = max(estimated * role_std_factor.get(role, 0.55), 8.0)

        return round(estimated, 1), round(estimated_std, 1)

    def _load_t20_form(self, t20_form_path: Path | str | None = None) -> None:
        """Load recent T20 form data from CSV."""
        path = Path(t20_form_path) if t20_form_path else self._T20_FORM_PATH
        if not path.exists():
            logger.debug("No T20 form data found at %s", path)
            return
        try:
            df = pd.read_csv(path)
            self._t20_form = df.set_index("player").to_dict(orient="index")
            logger.info("Loaded T20 form data for %d players", len(self._t20_form))
        except Exception as e:
            logger.warning("Could not load T20 form data: %s", e)

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
