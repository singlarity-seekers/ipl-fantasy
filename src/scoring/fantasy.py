"""
IPL Fantasy Cricket scoring engine.

Pure-function scorer: takes a player's raw match stats → fantasy points.
Covers batting, bowling, fielding, economy rate, strike rate, and participation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    BATTING,
    BOWLING,
    DUCK_ELIGIBLE_ROLES,
    ECONOMY_RATE_TIERS,
    FIELDING,
    PLAYING_XI_POINTS,
    STRIKE_RATE_EXEMPT_ROLES,
    STRIKE_RATE_TIERS,
)


@dataclass
class PlayerMatchStats:
    """Raw stats for a single player in a single match."""

    player_name: str
    role: str  # WK, BAT, AR, BOWL

    # Batting
    runs_scored: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False  # True if dismissed (for duck check)

    # Bowling
    wickets: int = 0
    lbw_bowled_wickets: int = 0  # subset of wickets via LBW or bowled
    overs_bowled: float = 0.0  # e.g. 3.4 means 3 overs + 4 balls
    runs_conceded: int = 0
    maiden_overs: int = 0

    # Fielding
    catches: int = 0
    stumpings: int = 0
    run_out_direct: int = 0
    run_out_indirect: int = 0  # throw/assist leading to run-out

    # Meta
    in_playing_xi: bool = True


def _overs_to_balls(overs: float) -> int:
    """Convert cricket overs notation (e.g. 3.4) to total balls."""
    full_overs = int(overs)
    partial_balls = round((overs - full_overs) * 10)
    return full_overs * 6 + partial_balls


def _tier_points(value: float, tiers: list[tuple[float, float, float]]) -> float:
    """Look up bonus/penalty points from a tier table."""
    for lower, upper, pts in tiers:
        if lower <= value < upper:
            return pts
    return 0.0


def compute_batting_points(stats: PlayerMatchStats) -> float:
    """Calculate batting fantasy points."""
    pts = 0.0

    # Per-run + boundary bonuses
    pts += stats.runs_scored * BATTING.per_run
    pts += stats.fours * BATTING.per_four
    pts += stats.sixes * BATTING.per_six

    # Milestones
    if stats.runs_scored >= 100:
        pts += BATTING.milestone_100
    elif stats.runs_scored >= 50:
        pts += BATTING.milestone_50
    elif stats.runs_scored >= 30:
        pts += BATTING.milestone_30

    # Duck penalty (only for BAT, WK, AR)
    if (
        stats.is_out
        and stats.runs_scored == 0
        and stats.role in DUCK_ELIGIBLE_ROLES
    ):
        pts += BATTING.duck_penalty

    return pts


def compute_bowling_points(stats: PlayerMatchStats) -> float:
    """Calculate bowling fantasy points."""
    pts = 0.0

    pts += stats.wickets * BOWLING.per_wicket
    pts += stats.lbw_bowled_wickets * BOWLING.lbw_bowled_bonus

    # Wicket hauls (bonuses stack with per-wicket)
    if stats.wickets >= 5:
        pts += BOWLING.haul_5w
    elif stats.wickets >= 4:
        pts += BOWLING.haul_4w
    elif stats.wickets >= 3:
        pts += BOWLING.haul_3w

    pts += stats.maiden_overs * BOWLING.per_maiden

    return pts


def compute_fielding_points(stats: PlayerMatchStats) -> float:
    """Calculate fielding fantasy points."""
    pts = 0.0

    pts += stats.catches * FIELDING.per_catch
    if stats.catches >= 3:
        pts += FIELDING.catch_bonus_3

    pts += stats.stumpings * FIELDING.per_stumping
    pts += stats.run_out_direct * FIELDING.run_out_direct
    pts += stats.run_out_indirect * FIELDING.run_out_indirect

    return pts


def compute_economy_points(stats: PlayerMatchStats) -> float:
    """Calculate economy rate bonus/penalty (min 2 overs bowled)."""
    total_balls = _overs_to_balls(stats.overs_bowled)
    if total_balls < 12:  # less than 2 overs
        return 0.0

    economy = (stats.runs_conceded / total_balls) * 6.0
    return _tier_points(economy, ECONOMY_RATE_TIERS)


def compute_strike_rate_points(stats: PlayerMatchStats) -> float:
    """Calculate strike rate bonus/penalty (min 10 balls, excludes BOWL)."""
    if stats.role in STRIKE_RATE_EXEMPT_ROLES:
        return 0.0
    if stats.balls_faced < 10:
        return 0.0

    strike_rate = (stats.runs_scored / stats.balls_faced) * 100.0
    return _tier_points(strike_rate, STRIKE_RATE_TIERS)


def compute_fantasy_points(stats: PlayerMatchStats) -> float:
    """
    Compute total IPL Fantasy points for a player in a match.

    Returns the total points (sum of batting, bowling, fielding,
    economy, strike rate, and participation bonuses).
    """
    pts = 0.0

    # Participation
    if stats.in_playing_xi:
        pts += PLAYING_XI_POINTS

    pts += compute_batting_points(stats)
    pts += compute_bowling_points(stats)
    pts += compute_fielding_points(stats)
    pts += compute_economy_points(stats)
    pts += compute_strike_rate_points(stats)

    return pts


def compute_fantasy_points_breakdown(
    stats: PlayerMatchStats,
) -> dict[str, float]:
    """
    Return a detailed breakdown of fantasy points by category.

    Useful for debugging and dashboard display.
    """
    batting = compute_batting_points(stats)
    bowling = compute_bowling_points(stats)
    fielding = compute_fielding_points(stats)
    economy = compute_economy_points(stats)
    strike_rate = compute_strike_rate_points(stats)
    participation = PLAYING_XI_POINTS if stats.in_playing_xi else 0.0

    return {
        "participation": participation,
        "batting": batting,
        "bowling": bowling,
        "fielding": fielding,
        "economy": economy,
        "strike_rate": strike_rate,
        "total": participation + batting + bowling + fielding + economy + strike_rate,
    }
