"""
IPLFantasy constrained lineup optimization using Integer Linear Programming.

Formulation:
  - Decision variables: x_i ∈ {0, 1} for each player i
  - Objective: maximize Σ x_i · E[points_i]
  - Constraints: squad size, budget, role limits, team caps
  - Top-K: iterative solve with diversity constraints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from src.config import BOOSTERS, CONSTRAINTS

logger = logging.getLogger(__name__)


@dataclass
class PlayerSlot:
    """A player available for selection with their attributes."""

    name: str
    role: str  # WK, BAT, AR, BOWL
    team: str  # IPL team name
    credit_cost: float
    expected_points: float
    # Optional: simulation-derived metrics
    ceiling_95: float = 0.0
    consistency: float = 0.0
    nationality: str = "Indian"  # "Indian" or "Overseas"


@dataclass
class OptimizedLineup:
    """A generated lineup with metadata."""

    players: list[PlayerSlot]
    total_expected_points: float
    total_credits: float
    captain: str | None = None
    vice_captain: str | None = None
    booster: str | None = None

    def __repr__(self) -> str:
        lines = [
            f"Lineup (E[pts]={self.total_expected_points:.1f}, "
            f"credits={self.total_credits:.1f})",
        ]
        if self.captain:
            lines.append(f"  C:  {self.captain}")
        if self.vice_captain:
            lines.append(f"  VC: {self.vice_captain}")

        for role in ["WK", "BAT", "AR", "BOWL"]:
            role_players = [p for p in self.players if p.role == role]
            if role_players:
                names = ", ".join(
                    f"{p.name} ({p.credit_cost}cr, E={p.expected_points:.1f})"
                    for p in role_players
                )
                lines.append(f"  {role}: {names}")

        return "\n".join(lines)

    @property
    def role_counts(self) -> dict[str, int]:
        counts = {}
        for p in self.players:
            counts[p.role] = counts.get(p.role, 0) + 1
        return counts

    @property
    def team_counts(self) -> dict[str, int]:
        counts = {}
        for p in self.players:
            counts[p.team] = counts.get(p.team, 0) + 1
        return counts

    def validate(self) -> list[str]:
        """Check all IPLFantasy constraints; return list of violations."""
        violations = []

        if len(self.players) != CONSTRAINTS.squad_size:
            violations.append(
                f"Squad size: {len(self.players)} (need {CONSTRAINTS.squad_size})"
            )

        if self.booster != "free_hit" and self.total_credits > CONSTRAINTS.total_credits:
            violations.append(
                f"Budget: {self.total_credits:.1f} > {CONSTRAINTS.total_credits}"
            )

        for role, min_count, max_count in CONSTRAINTS.role_limits:
            count = self.role_counts.get(role, 0)
            if count < min_count or count > max_count:
                violations.append(
                    f"Role {role}: {count} (need {min_count}–{max_count})"
                )

        for team, count in self.team_counts.items():
            if count > CONSTRAINTS.max_per_team:
                violations.append(
                    f"Team {team}: {count} > {CONSTRAINTS.max_per_team}"
                )

        overseas_count = sum(
            1 for p in self.players
            if getattr(p, "nationality", "Indian") == "Overseas"
        )
        if overseas_count > CONSTRAINTS.max_overseas:
            violations.append(
                f"Overseas: {overseas_count} > {CONSTRAINTS.max_overseas}"
            )

        return violations


class IPLFantasyOptimizer:
    """
    Integer Linear Programming optimizer for IPLFantasy team selection.

    Uses PuLP to solve the constrained optimization problem.
    """

    def __init__(self, objective: str = "expected", booster: str | None = None):
        """
        Args:
            objective: Optimization objective.
                - "expected": maximize expected fantasy points
                - "ceiling": maximize 95th percentile (boom-or-bust)
                - "floor": maximize worst-case (5th percentile)
                - "balanced": blend of expected + ceiling
            booster: Active booster for this match (None, "triple_captain",
                "free_hit", "foreign_stars", "indian_warrior", "double_points",
                "wildcard").
        """
        self.objective = objective
        self.booster = booster
        if booster and booster not in BOOSTERS:
            raise ValueError(
                f"Unknown booster '{booster}'. Valid: {list(BOOSTERS.keys())}"
            )

    def optimize(
        self,
        players: list[PlayerSlot],
        excluded: set[str] | None = None,
        must_include: set[str] | None = None,
    ) -> OptimizedLineup | None:
        """
        Find the optimal IPLFantasy lineup.

        Args:
            players: Available player pool.
            excluded: Player names to exclude.
            must_include: Player names that must be in the lineup.

        Returns:
            OptimizedLineup or None if infeasible.
        """
        excluded = excluded or set()
        must_include = must_include or set()

        # Filter out excluded players
        pool = [p for p in players if p.name not in excluded]

        if len(pool) < CONSTRAINTS.squad_size:
            logger.error("Not enough players in pool (%d)", len(pool))
            return None

        # Create ILP problem
        prob = pulp.LpProblem("IPLFantasy_Lineup", pulp.LpMaximize)

        # Decision variables
        x = {
            p.name: pulp.LpVariable(f"x_{p.name}", cat="Binary")
            for p in pool
        }

        # ── Booster-adjusted points ──────────────────────────────────────
        def _effective_points(p: PlayerSlot) -> float:
            pts = p.expected_points
            if self.booster == "foreign_stars" and p.nationality == "Overseas":
                pts *= 2.0
            elif self.booster == "indian_warrior" and p.nationality == "Indian":
                pts *= 2.0
            elif self.booster == "double_points":
                pts *= 2.0
            return pts

        def _effective_ceiling(p: PlayerSlot) -> float:
            c = p.ceiling_95
            if self.booster == "foreign_stars" and p.nationality == "Overseas":
                c *= 2.0
            elif self.booster == "indian_warrior" and p.nationality == "Indian":
                c *= 2.0
            elif self.booster == "double_points":
                c *= 2.0
            return c

        # Objective function
        if self.objective == "expected":
            prob += pulp.lpSum(x[p.name] * _effective_points(p) for p in pool)
        elif self.objective == "ceiling":
            prob += pulp.lpSum(x[p.name] * _effective_ceiling(p) for p in pool)
        elif self.objective == "floor":
            prob += pulp.lpSum(
                x[p.name] * max(_effective_points(p) - 1.5 * (_effective_ceiling(p) - _effective_points(p)) / 1.645, 0)
                for p in pool
            )
        elif self.objective == "balanced":
            prob += pulp.lpSum(
                x[p.name] * (0.6 * _effective_points(p) + 0.4 * _effective_ceiling(p))
                for p in pool
            )

        # ── Constraints ───────────────────────────────────────────────────

        # Exactly 11 players
        prob += pulp.lpSum(x[p.name] for p in pool) == CONSTRAINTS.squad_size

        # Budget constraint (Free Hit removes this)
        if self.booster != "free_hit":
            prob += (
                pulp.lpSum(x[p.name] * p.credit_cost for p in pool)
                <= CONSTRAINTS.total_credits
            )

        # Role constraints
        for role, min_count, max_count in CONSTRAINTS.role_limits:
            role_players = [p for p in pool if p.role == role]
            if role_players:
                prob += pulp.lpSum(x[p.name] for p in role_players) >= min_count
                prob += pulp.lpSum(x[p.name] for p in role_players) <= max_count

        # Max players per team
        teams = set(p.team for p in pool)
        for team in teams:
            team_players = [p for p in pool if p.team == team]
            prob += (
                pulp.lpSum(x[p.name] for p in team_players)
                <= CONSTRAINTS.max_per_team
            )

        # Max overseas players
        overseas = [p for p in pool if getattr(p, "nationality", "Indian") == "Overseas"]
        if overseas:
            prob += (
                pulp.lpSum(x[p.name] for p in overseas)
                <= CONSTRAINTS.max_overseas
            )

        # Must-include constraints
        for name in must_include:
            if name in x:
                prob += x[name] == 1

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.constants.LpStatusOptimal:
            logger.warning("Optimization infeasible (status=%d)", prob.status)
            return None

        # Extract solution
        selected = [
            p for p in pool if pulp.value(x[p.name]) > 0.5
        ]

        lineup = OptimizedLineup(
            players=selected,
            total_expected_points=sum(_effective_points(p) for p in selected),
            total_credits=sum(p.credit_cost for p in selected),
            booster=self.booster,
        )

        violations = lineup.validate()
        if violations:
            logger.warning("Lineup has violations: %s", violations)

        return lineup

    def generate_top_k(
        self,
        players: list[PlayerSlot],
        k: int = 5,
        min_diff: int = 2,
    ) -> list[OptimizedLineup]:
        """
        Generate K diverse optimal lineups.

        Uses iterative solving with "differ by ≥ min_diff players"
        constraints between each new lineup and all previous ones.

        Args:
            players: Available player pool.
            k: Number of lineups to generate.
            min_diff: Minimum number of different players between any two lineups.

        Returns:
            List of OptimizedLineup objects, best first.
        """
        lineups: list[OptimizedLineup] = []
        previous_selections: list[set[str]] = []

        for i in range(k):
            pool = [p for p in players]

            # Create problem
            prob = pulp.LpProblem(f"IPLFantasy_Lineup_{i}", pulp.LpMaximize)

            x = {
                p.name: pulp.LpVariable(f"x_{i}_{p.name}", cat="Binary")
                for p in pool
            }

            # Booster-adjusted points (same logic as optimize())
            def _eff_pts(p: PlayerSlot) -> float:
                pts = p.expected_points
                if self.booster == "foreign_stars" and p.nationality == "Overseas":
                    pts *= 2.0
                elif self.booster == "indian_warrior" and p.nationality == "Indian":
                    pts *= 2.0
                elif self.booster == "double_points":
                    pts *= 2.0
                return pts

            # Objective
            prob += pulp.lpSum(x[p.name] * _eff_pts(p) for p in pool)

            # Base constraints
            prob += pulp.lpSum(x[p.name] for p in pool) == CONSTRAINTS.squad_size
            if self.booster != "free_hit":
                prob += (
                    pulp.lpSum(x[p.name] * p.credit_cost for p in pool)
                    <= CONSTRAINTS.total_credits
                )

            for role, min_count, max_count in CONSTRAINTS.role_limits:
                role_players = [p for p in pool if p.role == role]
                if role_players:
                    prob += pulp.lpSum(x[p.name] for p in role_players) >= min_count
                    prob += pulp.lpSum(x[p.name] for p in role_players) <= max_count

            teams = set(p.team for p in pool)
            for team in teams:
                team_players = [p for p in pool if p.team == team]
                prob += (
                    pulp.lpSum(x[p.name] for p in team_players)
                    <= CONSTRAINTS.max_per_team
                )

            # Max overseas players
            overseas = [p for p in pool if getattr(p, "nationality", "Indian") == "Overseas"]
            if overseas:
                prob += (
                    pulp.lpSum(x[p.name] for p in overseas)
                    <= CONSTRAINTS.max_overseas
                )

            # Diversity constraints vs previous lineups
            for prev_names in previous_selections:
                # At least min_diff players must be different
                # i.e., overlap ≤ squad_size - min_diff
                overlap = [
                    p for p in pool if p.name in prev_names
                ]
                prob += (
                    pulp.lpSum(x[p.name] for p in overlap)
                    <= CONSTRAINTS.squad_size - min_diff
                )

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status != pulp.constants.LpStatusOptimal:
                logger.info("Could not generate lineup %d (infeasible)", i + 1)
                break

            selected = [p for p in pool if pulp.value(x[p.name]) > 0.5]
            lineup = OptimizedLineup(
                players=selected,
                total_expected_points=sum(_eff_pts(p) for p in selected),
                total_credits=sum(p.credit_cost for p in selected),
                booster=self.booster,
            )
            lineups.append(lineup)
            previous_selections.append({p.name for p in selected})

        logger.info("Generated %d diverse lineups", len(lineups))
        return lineups
