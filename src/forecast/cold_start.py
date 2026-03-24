"""
Cold-start handling for uncapped / new / under-sampled players.

Detects players with insufficient historical data and generates
prior-based forecasts using role-based baselines and price-to-value
ratio analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncappedPlayerProfile:
    """Profile for a player with limited or no IPL history."""

    player: str
    role: str  # WK, BAT, AR, BOWL
    credit_cost: float
    matches_played: int = 0
    is_uncapped: bool = True
    domestic_stats: dict | None = None  # Hook for domestic data
    estimated_value: float = 0.0  # Expected points / credit cost
    priority_score: float = 0.0  # Overall recommendation score


# Role-based priors for uncapped players (mean, std of fantasy points)
UNCAPPED_PRIORS = {
    "BAT": {"mean": 18.0, "std": 14.0, "ceiling": 55.0},
    "BOWL": {"mean": 20.0, "std": 16.0, "ceiling": 60.0},
    "AR": {"mean": 25.0, "std": 18.0, "ceiling": 70.0},
    "WK": {"mean": 16.0, "std": 12.0, "ceiling": 50.0},
}

# Minimum matches for a player to be considered "known"
MIN_MATCHES_THRESHOLD = 5


def detect_uncapped_players(
    available_players: list[dict],
    player_history: dict[str, int],  # player → matches played
) -> list[UncappedPlayerProfile]:
    """
    Identify uncapped or under-sampled players from the available pool.

    Args:
        available_players: List of dicts with keys: name, role, credit_cost.
        player_history: {player_name: matches_played} from historical data.

    Returns:
        List of UncappedPlayerProfile for players below threshold.
    """
    uncapped = []
    for p in available_players:
        name = p["name"]
        role = p.get("role", "BAT")
        cost = p.get("credit_cost", 6.0)
        matches = player_history.get(name, 0)

        if matches < MIN_MATCHES_THRESHOLD:
            prior = UNCAPPED_PRIORS.get(role, UNCAPPED_PRIORS["BAT"])
            estimated_value = prior["mean"] / max(cost, 1.0)

            profile = UncappedPlayerProfile(
                player=name,
                role=role,
                credit_cost=cost,
                matches_played=matches,
                is_uncapped=matches == 0,
                estimated_value=estimated_value,
                priority_score=_compute_priority(
                    estimated_value, prior["ceiling"], cost, matches
                ),
            )
            uncapped.append(profile)

    # Sort by priority (best value opportunities first)
    uncapped.sort(key=lambda x: x.priority_score, reverse=True)
    logger.info("Detected %d uncapped/under-sampled players", len(uncapped))
    return uncapped


def _compute_priority(
    estimated_value: float,
    ceiling: float,
    cost: float,
    matches: int,
) -> float:
    """
    Compute a priority score for an uncapped player.

    Higher score = better opportunity. Factors:
      - Value efficiency (expected points per credit)
      - Ceiling potential
      - Low cost (budget-friendly picks)
      - Novelty bonus (fewer matches → more upside uncertainty)
    """
    value_score = estimated_value * 2.0
    ceiling_score = (ceiling / max(cost, 1.0)) * 0.5
    cost_score = max(0, (8.0 - cost) / 8.0) * 1.0  # cheaper = higher score
    novelty_bonus = max(0, (MIN_MATCHES_THRESHOLD - matches)) * 0.3

    return value_score + ceiling_score + cost_score + novelty_bonus


def generate_uncapped_samples(
    profile: UncappedPlayerProfile,
    n: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate fantasy point samples for an uncapped player
    using role-based priors with a right-skewed distribution
    (log-normal mixture to capture potential breakout games).
    """
    rng = rng or np.random.default_rng(42)
    prior = UNCAPPED_PRIORS.get(profile.role, UNCAPPED_PRIORS["BAT"])

    # 80% from truncated normal (typical performances)
    normal_samples = rng.normal(prior["mean"], prior["std"], size=int(n * 0.8))
    normal_samples = np.clip(normal_samples, 0, None)

    # 20% from higher distribution (breakout potential)
    breakout_mean = prior["mean"] * 1.8
    breakout_std = prior["std"] * 1.5
    breakout_samples = rng.normal(breakout_mean, breakout_std, size=int(n * 0.2))
    breakout_samples = np.clip(breakout_samples, 0, None)

    samples = np.concatenate([normal_samples, breakout_samples])
    rng.shuffle(samples)

    return samples[:n]
