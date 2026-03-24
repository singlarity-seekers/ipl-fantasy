"""
Feature engineering pipeline for the IPL Fantasy system.

Transforms raw ball-by-ball and match data into player-level features
suitable for forecasting models. Produces:
  - Per-match player scorecards (batting + bowling + fielding stats)
  - Historical aggregates (career, recent form, venue splits, vs-opposition)
  - Rolling exponentially-weighted features
  - Venue characteristics
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.scoring.fantasy import PlayerMatchStats, compute_fantasy_points

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-match player scorecards
# ═══════════════════════════════════════════════════════════════════════════════

def build_batting_scorecards(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data into per-match batting scorecards.

    Returns DataFrame with columns:
        match_id, batter, runs_scored, balls_faced, fours, sixes, is_out
    """
    # Filter out extras that don't count as balls faced for the batter
    bat = deliveries.copy()

    batting = bat.groupby(["match_id", "batter"]).agg(
        runs_scored=("batsman_runs", "sum"),
        balls_faced=("batsman_runs", "count"),  # each row = 1 ball faced
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
    ).reset_index()

    # Determine dismissals
    dismissed = deliveries[deliveries["is_wicket"] == 1][
        ["match_id", "player_dismissed"]
    ].drop_duplicates()
    dismissed = dismissed.rename(columns={"player_dismissed": "batter"})
    dismissed["is_out"] = True

    batting = batting.merge(dismissed, on=["match_id", "batter"], how="left")
    batting["is_out"] = batting["is_out"].fillna(False)

    return batting


def build_bowling_scorecards(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data into per-match bowling scorecards.

    Returns DataFrame with columns:
        match_id, bowler, wickets, lbw_bowled_wickets, overs_bowled,
        runs_conceded, maiden_overs
    """
    bowl = deliveries.copy()

    # Runs conceded = total_runs - byes - legbyes (bowler not charged)
    # Simplified: use total_runs minus extras that aren't the bowler's fault
    if "extras_type" in bowl.columns:
        bowler_not_charged = bowl["extras_type"].isin(["byes", "legbyes"])
        bowl["bowler_runs"] = np.where(
            bowler_not_charged,
            bowl["total_runs"] - bowl["extra_runs"],
            bowl["total_runs"],
        )
    else:
        bowl["bowler_runs"] = bowl["total_runs"]

    # Wickets (exclude run-outs — bowler doesn't get credit)
    bowl["is_bowler_wicket"] = (
        (bowl["is_wicket"] == 1)
        & (~bowl["dismissal_kind"].isin(["run out", "retired hurt", "retired out", "obstructing the field"]).fillna(False))
    )

    # LBW / Bowled wickets
    bowl["is_lbw_bowled"] = (
        (bowl["is_wicket"] == 1)
        & (bowl["dismissal_kind"].isin(["lbw", "bowled"]).fillna(False))
    )

    # Per-over maiden detection
    bowl["over_key"] = bowl["match_id"].astype(str) + "_" + bowl["inning"].astype(str) + "_" + bowl["bowler"] + "_" + bowl["over"].astype(str)
    over_runs = bowl.groupby("over_key")["bowler_runs"].sum().reset_index()
    over_runs["is_maiden"] = (over_runs["bowler_runs"] == 0).astype(int)
    over_balls = bowl.groupby("over_key").size().reset_index(name="balls_in_over")
    over_runs = over_runs.merge(over_balls, on="over_key")
    # Only count as maiden if it's a complete over (6 balls)
    over_runs["is_maiden"] = over_runs["is_maiden"] & (over_runs["balls_in_over"] >= 6)

    # Map maiden back to bowler
    bowl = bowl.merge(
        over_runs[["over_key", "is_maiden"]],
        on="over_key",
        how="left",
    )

    bowling = bowl.groupby(["match_id", "bowler"]).agg(
        wickets=("is_bowler_wicket", "sum"),
        lbw_bowled_wickets=("is_lbw_bowled", "sum"),
        balls_bowled=("bowler_runs", "count"),
        runs_conceded=("bowler_runs", "sum"),
        maiden_overs=("is_maiden", lambda x: x.drop_duplicates().sum()),
    ).reset_index()

    # Convert balls to overs notation (e.g. 22 balls → 3.4 overs)
    bowling["overs_bowled"] = (
        (bowling["balls_bowled"] // 6) + (bowling["balls_bowled"] % 6) / 10
    )

    bowling = bowling.drop(columns=["balls_bowled"])
    return bowling


def build_fielding_scorecards(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data into per-match fielding scorecards.

    Returns DataFrame with columns:
        match_id, fielder, catches, stumpings, run_out_direct, run_out_indirect
    """
    wickets = deliveries[deliveries["is_wicket"] == 1].copy()

    if wickets.empty:
        return pd.DataFrame(columns=[
            "match_id", "fielder", "catches", "stumpings",
            "run_out_direct", "run_out_indirect",
        ])

    # Catches
    catches = wickets[wickets["dismissal_kind"] == "caught"]
    catch_counts = catches.groupby(["match_id", "fielder"]).size().reset_index(name="catches")

    # Stumpings
    stumpings = wickets[wickets["dismissal_kind"] == "stumped"]
    stump_counts = stumpings.groupby(["match_id", "fielder"]).size().reset_index(name="stumpings")

    # Run-outs (simplified: any fielder involved gets indirect credit;
    # direct detection would need more granular data)
    runouts = wickets[wickets["dismissal_kind"] == "run out"]
    ro_counts = runouts.groupby(["match_id", "fielder"]).size().reset_index(name="run_out_indirect")

    # Merge all fielding
    all_fielders = set()
    for df in [catch_counts, stump_counts, ro_counts]:
        if not df.empty:
            all_fielders.update(
                df[["match_id", "fielder"]].apply(tuple, axis=1).tolist()
            )

    if not all_fielders:
        return pd.DataFrame(columns=[
            "match_id", "fielder", "catches", "stumpings",
            "run_out_direct", "run_out_indirect",
        ])

    base = pd.DataFrame(list(all_fielders), columns=["match_id", "fielder"])

    for df, col in [(catch_counts, "catches"), (stump_counts, "stumpings"), (ro_counts, "run_out_indirect")]:
        if not df.empty:
            base = base.merge(df, on=["match_id", "fielder"], how="left")
        else:
            base[col] = 0

    base = base.fillna(0)
    for col in ["catches", "stumpings", "run_out_indirect"]:
        base[col] = base[col].astype(int)

    base["run_out_direct"] = 0  # Need more data to distinguish direct vs indirect

    return base


def build_player_scorecards(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build complete per-match player scorecards by combining batting,
    bowling, and fielding stats, and computing IPL Fantasy fantasy points.

    Returns DataFrame indexed by (match_id, player) with all stats and
    computed fantasy points.
    """
    batting = build_batting_scorecards(deliveries)
    bowling = build_bowling_scorecards(deliveries)
    fielding = build_fielding_scorecards(deliveries)

    # Get all unique (match_id, player) combinations
    batters = batting[["match_id", "batter"]].rename(columns={"batter": "player"})
    bowlers = bowling[["match_id", "bowler"]].rename(columns={"bowler": "player"})
    fielders = fielding[["match_id", "fielder"]].rename(columns={"fielder": "player"})

    players = pd.concat([batters, bowlers, fielders]).drop_duplicates()

    # Merge batting
    players = players.merge(
        batting.rename(columns={"batter": "player"}),
        on=["match_id", "player"],
        how="left",
    )

    # Merge bowling
    players = players.merge(
        bowling.rename(columns={"bowler": "player"}),
        on=["match_id", "player"],
        how="left",
    )

    # Merge fielding
    players = players.merge(
        fielding.rename(columns={"fielder": "player"}),
        on=["match_id", "player"],
        how="left",
    )

    # Fill NaN with defaults
    fill_defaults = {
        "runs_scored": 0, "balls_faced": 0, "fours": 0, "sixes": 0,
        "is_out": False, "wickets": 0, "lbw_bowled_wickets": 0,
        "overs_bowled": 0.0, "runs_conceded": 0, "maiden_overs": 0,
        "catches": 0, "stumpings": 0, "run_out_direct": 0, "run_out_indirect": 0,
    }
    players = players.fillna(fill_defaults)

    # Infer role (simplified heuristic — can be overridden by explicit mapping)
    players["role"] = _infer_role(players)

    # Add match context if available
    if matches is not None:
        match_cols = ["id", "season", "venue", "date", "team1", "team2"]
        available = [c for c in match_cols if c in matches.columns]
        players = players.merge(
            matches[available],
            left_on="match_id",
            right_on="id",
            how="left",
        )
        if "id" in players.columns:
            players = players.drop(columns=["id"])

    # Compute IPL Fantasy fantasy points
    players["fantasy_points"] = players.apply(_row_to_fantasy_points, axis=1)

    logger.info("Built scorecards for %d player-match records", len(players))
    return players


def _infer_role(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic role classification based on stats.

    - If player bowled ≥ 2 overs AND scored ≥ 1 run → AR
    - If player bowled ≥ 2 overs → BOWL
    - Else → BAT

    Note: WK detection requires external roster data (not in deliveries).
    Default to BAT for now; can be overridden.
    """
    bowled = df["overs_bowled"] >= 2.0
    batted = df["balls_faced"] >= 1

    role = pd.Series("BAT", index=df.index)
    role[bowled & ~batted] = "BOWL"
    role[bowled & batted] = "AR"

    return role


def _row_to_fantasy_points(row: pd.Series) -> float:
    """Convert a scorecard row to IPL Fantasy fantasy points."""
    stats = PlayerMatchStats(
        player_name=row["player"],
        role=row["role"],
        runs_scored=int(row["runs_scored"]),
        balls_faced=int(row["balls_faced"]),
        fours=int(row["fours"]),
        sixes=int(row["sixes"]),
        is_out=bool(row["is_out"]),
        wickets=int(row["wickets"]),
        lbw_bowled_wickets=int(row["lbw_bowled_wickets"]),
        overs_bowled=float(row["overs_bowled"]),
        runs_conceded=int(row["runs_conceded"]),
        maiden_overs=int(row["maiden_overs"]),
        catches=int(row["catches"]),
        stumpings=int(row["stumpings"]),
        run_out_direct=int(row["run_out_direct"]),
        run_out_indirect=int(row["run_out_indirect"]),
        in_playing_xi=True,
    )
    return compute_fantasy_points(stats)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Historical aggregates & rolling features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_player_career_stats(scorecards: pd.DataFrame) -> pd.DataFrame:
    """
    Compute career-level aggregated stats per player.

    Returns DataFrame with columns:
        player, matches_played, total_runs, avg_runs, total_wickets,
        avg_wickets, total_catches, avg_fantasy_points, std_fantasy_points,
        median_fantasy_points
    """
    career = scorecards.groupby("player").agg(
        matches_played=("match_id", "nunique"),
        total_runs=("runs_scored", "sum"),
        avg_runs=("runs_scored", "mean"),
        total_wickets=("wickets", "sum"),
        avg_wickets=("wickets", "mean"),
        total_catches=("catches", "sum"),
        avg_fantasy_points=("fantasy_points", "mean"),
        std_fantasy_points=("fantasy_points", "std"),
        median_fantasy_points=("fantasy_points", "median"),
    ).reset_index()

    career["std_fantasy_points"] = career["std_fantasy_points"].fillna(0.0)
    return career


def compute_venue_stats(
    scorecards: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute player performance splits by venue.

    Returns DataFrame with: player, venue, venue_matches, venue_avg_fp, venue_std_fp
    """
    if "venue" not in scorecards.columns:
        logger.warning("No venue column in scorecards; skipping venue stats")
        return pd.DataFrame()

    venue = scorecards.groupby(["player", "venue"]).agg(
        venue_matches=("match_id", "nunique"),
        venue_avg_fp=("fantasy_points", "mean"),
        venue_std_fp=("fantasy_points", "std"),
    ).reset_index()

    venue["venue_std_fp"] = venue["venue_std_fp"].fillna(0.0)
    return venue


def compute_opposition_stats(
    scorecards: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute player performance splits vs each opponent.

    Requires 'team1' and 'team2' columns to determine opponent.
    Returns: player, opponent, opp_matches, opp_avg_fp
    """
    if "team1" not in scorecards.columns or "team2" not in scorecards.columns:
        logger.warning("No team columns in scorecards; skipping opposition stats")
        return pd.DataFrame()

    # TODO: Need to determine which team the player belongs to.
    # For now, return empty — requires roster/team mapping.
    logger.info("Opposition stats require team roster mapping (future enhancement)")
    return pd.DataFrame()


def compute_rolling_features(
    scorecards: pd.DataFrame,
    windows: list[int] | None = None,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """
    Compute rolling and exponentially-weighted features per player.

    Args:
        scorecards: Must contain 'date' column for time ordering.
        windows: Rolling window sizes (default [5, 10]).
        alpha: EWM alpha parameter.

    Returns: DataFrame with rolling columns appended.
    """
    if "date" not in scorecards.columns:
        logger.warning("No date column; cannot compute rolling features")
        return scorecards

    windows = windows or [5, 10]
    df = scorecards.sort_values(["player", "date"]).copy()

    for w in windows:
        df[f"rolling_{w}_avg_fp"] = (
            df.groupby("player")["fantasy_points"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_{w}_avg_runs"] = (
            df.groupby("player")["runs_scored"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

    # Exponentially weighted
    df["ewm_avg_fp"] = (
        df.groupby("player")["fantasy_points"]
        .transform(lambda x: x.ewm(alpha=alpha, min_periods=1).mean())
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Venue characteristics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_venue_characteristics(
    scorecards: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute venue-level characteristics (avg scores, pace/spin friendliness).

    Returns: venue, avg_first_innings_score, avg_total_wickets, matches_at_venue
    """
    if "venue" not in scorecards.columns:
        return pd.DataFrame()

    venue_stats = scorecards.groupby("venue").agg(
        matches_at_venue=("match_id", "nunique"),
        avg_runs_per_player=("runs_scored", "mean"),
        avg_wickets_per_player=("wickets", "mean"),
        avg_fp_at_venue=("fantasy_points", "mean"),
    ).reset_index()

    return venue_stats
