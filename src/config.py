"""
Central configuration for the IPL Fantasy Agentic System.

Contains IPL Fantasy scoring tables, team constraints, simulation defaults,
and all tunable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── IPL Fantasy Scoring ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BattingPoints:
    """IPL Fantasy batting point values."""

    per_run: float = 1.0
    per_four: float = 1.0  # bonus on top of run point
    per_six: float = 2.0  # bonus on top of run point
    milestone_30: float = 4.0
    milestone_50: float = 8.0
    milestone_100: float = 16.0
    duck_penalty: float = -2.0  # applies to BAT, WK, AR only


@dataclass(frozen=True)
class BowlingPoints:
    """IPL Fantasy bowling point values."""

    per_wicket: float = 25.0  # excludes run-outs
    lbw_bowled_bonus: float = 8.0
    haul_3w: float = 4.0
    haul_4w: float = 8.0
    haul_5w: float = 16.0
    per_maiden: float = 12.0


@dataclass(frozen=True)
class FieldingPoints:
    """IPL Fantasy fielding point values."""

    per_catch: float = 8.0
    catch_bonus_3: float = 4.0  # 3+ catches in a match
    per_stumping: float = 12.0
    run_out_direct: float = 12.0
    run_out_indirect: float = 6.0  # throw or catch assist


# Economy rate bonus/penalty (min 2 overs bowled)
ECONOMY_RATE_TIERS: list[tuple[float, float, float]] = [
    # (lower_bound, upper_bound, points)
    (0.0, 5.0, 6.0),
    (5.0, 6.0, 4.0),
    (6.0, 7.0, 2.0),
    (7.0, 10.0, 0.0),
    (10.0, 11.0, -2.0),
    (11.0, 12.0, -4.0),
    (12.0, float("inf"), -6.0),
]

# Strike rate bonus/penalty (min 10 balls faced, excludes BOWL role)
STRIKE_RATE_TIERS: list[tuple[float, float, float]] = [
    # (lower_bound, upper_bound, points)
    (170.0, float("inf"), 6.0),
    (150.0, 170.0, 4.0),
    (130.0, 150.0, 2.0),
    (70.0, 130.0, 0.0),
    (60.0, 70.0, -2.0),
    (50.0, 60.0, -4.0),
    (0.0, 50.0, -6.0),
]

# Playing XI participation
PLAYING_XI_POINTS: float = 4.0


# ── IPL Fantasy Team Constraints ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TeamConstraints:
    """IPL Fantasy League 2026 team formation rules (Season Long format)."""

    squad_size: int = 11
    total_credits: float = 100.0
    max_per_team: int = 7  # Franchisee Spread rule
    max_overseas: int = 4  # Overseas Limit rule
    # (role, min, max) — Season Long Fantasy constraints
    role_limits: tuple[tuple[str, int, int], ...] = (
        ("WK", 1, 4),
        ("BAT", 3, 6),
        ("AR", 1, 4),
        ("BOWL", 3, 6),
    )
    captain_multiplier: float = 2.0
    vc_multiplier: float = 1.5
    triple_captain_multiplier: float = 3.0  # Triple Captain booster


# ── Boosters ─────────────────────────────────────────────────────────────────

BOOSTERS = {
    "triple_captain": {
        "description": "Captain earns 3x points instead of 2x",
        "max_uses": 2,
        "effect": "captain_3x",
    },
    "free_hit": {
        "description": "No budget restriction; team reverts after match",
        "max_uses": 1,
        "effect": "no_budget",
    },
    "foreign_stars": {
        "description": "All overseas players earn 2x points",
        "max_uses": 2,
        "effect": "overseas_2x",
    },
    "indian_warrior": {
        "description": "All Indian players earn 2x points",
        "max_uses": 2,
        "effect": "indian_2x",
    },
    "double_points": {
        "description": "Entire team earns 2x points",
        "max_uses": 2,
        "effect": "all_2x",
    },
    "wildcard": {
        "description": "Unlimited transfers within 100-credit budget",
        "max_uses": 1,
        "effect": "transfers_only",
    },
}


# ── Player Roles ──────────────────────────────────────────────────────────────

ROLES = ("WK", "BAT", "AR", "BOWL")

# Roles that receive duck penalty
DUCK_ELIGIBLE_ROLES = {"WK", "BAT", "AR"}

# Roles excluded from strike-rate bonus/penalty
STRIKE_RATE_EXEMPT_ROLES = {"BOWL"}


# ── Simulation Defaults ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo simulation parameters."""

    n_simulations: int = 10_000
    random_seed: int | None = 42
    min_overs_for_economy: int = 2
    min_balls_for_strike_rate: int = 10


# ── Transfer Optimizer Config ───────────────────────────────────────────────


@dataclass(frozen=True)
class TransferConfig:
    """
    Season-long transfer optimization parameters.

    - league_stage_budget: Max transfers during league stage (matches 1-70)
    - playoff_budget: Max transfers during playoffs (post-match 70)
    - default_transfer_penalty: Penalty per transfer in ILP objective (~5 pts)
    - look_ahead_alpha: Weight for future fixture value (~0.3)
    - look_ahead_decay: Decay factor for future matches (~0.9 per match)
    """

    league_stage_budget: int = 160
    playoff_budget: int = 10
    league_stage_end: int = 70
    default_transfer_penalty: float = 5.0
    look_ahead_decay: float = 0.9
    look_ahead_alpha: float = 0.3


# ── Singleton Instances ───────────────────────────────────────────────────────

BATTING = BattingPoints()
BOWLING = BowlingPoints()
FIELDING = FieldingPoints()
CONSTRAINTS = TeamConstraints()
SIM_CONFIG = SimulationConfig()
TRANSFER = TransferConfig()
