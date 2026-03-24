"""
Transfer-aware optimizer with ILP formulation.

Extends the base optimizer to penalize transfers and consider future fixture value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    lpSum,
    value,
)

from src.config import CONSTRAINTS, TRANSFER
from src.optimizer.fantasy_ilp import PlayerSlot

if TYPE_CHECKING:
    from src.season.schedule import Schedule


@dataclass
class TransferPlan:
    """Result of transfer optimization."""

    new_squad: list[PlayerSlot]
    transfers_in: list[str]
    transfers_out: list[str]
    num_transfers: int
    expected_points: float
    kept_players: list[str]

    def __repr__(self) -> str:
        return (
            f"TransferPlan("
            f"transfers={self.num_transfers}, "
            f"points={self.expected_points:.1f}, "
            f"kept={len(self.kept_players)}/11)"
        )


class TransferOptimizer:
    """
    Transfer-aware ILP optimizer.

    Formulation:
        maximize Σ x_i * (E[pts_i] + α * future_value_i) - λ * Σ t_i

    Where:
        x_i ∈ {0,1} = player i in new squad
        t_i ∈ {0,1} = player i is a NEW transfer in
        s_i (const) = 1 if player i already in current squad

    Constraints:
        t_i >= x_i - s_i           (detect transfers)
        Σ t_i <= max_transfers     (respect transfer budget)
        + all standard constraints (11 players, 100cr, roles, overseas, franchise)
    """

    def __init__(
        self,
        transfer_penalty: float = TRANSFER.default_transfer_penalty,
        look_ahead_alpha: float = TRANSFER.look_ahead_alpha,
        schedule: Schedule | None = None,
    ):
        self.transfer_penalty = transfer_penalty
        self.look_ahead_alpha = look_ahead_alpha
        self.schedule = schedule

    def _calculate_future_value(
        self,
        player: PlayerSlot,
        current_match: int,
        look_ahead: int = 5,
    ) -> float:
        """
        Calculate future fixture value for look-ahead optimization.

        Returns fixture_density * player_avg_points * decay_factor
        """
        if self.schedule is None:
            return 0.0

        # Count how many matches this player's team plays in next N matches
        fixture_count = self.schedule.player_match_count(player.team, current_match, look_ahead)

        if fixture_count == 0:
            return 0.0

        # Average expected points per match (conservative estimate)
        avg_points = max(player.expected_points, 20.0)

        # Decay factor: future matches are worth less
        decay = TRANSFER.look_ahead_decay ** ((fixture_count - 1) / 2)

        return fixture_count * avg_points * decay

    def optimize(
        self,
        current_squad: list[str],
        available_players: list[PlayerSlot],
        max_transfers: int,
        current_match: int = 1,
        look_ahead: int = 0,
    ) -> TransferPlan:
        """
        Optimize squad with transfer constraints.

        Args:
            current_squad: List of 11 player names currently in squad.
                          Can be empty for initial squad creation.
            available_players: All players to consider (245 from squads)
            max_transfers: Maximum transfers allowed for this window
            current_match: Current match number (for fixture density)
            look_ahead: Number of future matches to consider (0 = disabled)

        Returns:
            TransferPlan with optimized squad
        """
        # Handle initial squad creation (empty current squad)
        is_initial_squad = len(current_squad) == 0

        if not is_initial_squad and len(current_squad) != CONSTRAINTS.squad_size:
            raise ValueError(
                f"Current squad must have {CONSTRAINTS.squad_size} players "
                f"(or be empty for initial squad creation)"
            )

        current_squad_set = set(current_squad)
        n = len(available_players)

        # Create player index mapping
        player_idx = {p.name: i for i, p in enumerate(available_players)}

        # Check all current squad players are in available pool
        missing = current_squad_set - set(player_idx.keys())
        if missing:
            raise ValueError(f"Current squad players not in pool: {missing}")

        # Build objective coefficients
        # E[pts] + α * future_value - λ * transfer_penalty
        objective = []
        for i, player in enumerate(available_players):
            base_points = player.expected_points

            # Future value (if look_ahead enabled)
            future_value = 0.0
            if look_ahead > 0 and self.schedule is not None:
                future_value = self._calculate_future_value(player, current_match, look_ahead)

            # Transfer penalty (applied if player not in current squad)
            is_current = player.name in current_squad_set
            transfer_cost = 0.0 if is_current else self.transfer_penalty

            total_value = base_points + self.look_ahead_alpha * future_value - transfer_cost
            objective.append(total_value)

        # ILP Problem
        prob = LpProblem("TransferOptimizer", LpMaximize)

        # Decision variables
        x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]  # Player in squad
        t = [LpVariable(f"t_{i}", cat=LpBinary) for i in range(n)]  # Is new transfer

        # Objective: maximize total value
        prob += lpSum([objective[i] * x[i] for i in range(n)])

        # Transfer detection: t_i >= x_i - s_i
        # s_i = 1 if player already in squad
        # For initial squad creation, treat all selected as transfers in
        for i, player in enumerate(available_players):
            s_i = 1 if player.name in current_squad_set else 0
            if is_initial_squad:
                # For initial squad: t_i = x_i (everyone is a "transfer in")
                prob += t[i] == x[i], f"transfer_detect_{i}"
            else:
                # Normal case: t_i >= x_i - s_i (detect new transfers)
                prob += t[i] >= x[i] - s_i, f"transfer_detect_{i}"

        # Transfer budget constraint
        prob += lpSum(t) <= max_transfers, "transfer_budget"

        # Squad size constraint
        prob += lpSum(x) == CONSTRAINTS.squad_size, "squad_size"

        # Credit budget constraint
        prob += (
            (
                lpSum([available_players[i].credit_cost * x[i] for i in range(n)])
                <= CONSTRAINTS.total_credits
            ),
            "credit_budget",
        )

        # Role constraints (min, max for each role)
        for role, min_count, max_count in CONSTRAINTS.role_limits:
            role_indices = [i for i, p in enumerate(available_players) if p.role == role]
            if role_indices:
                prob += (lpSum([x[i] for i in role_indices]) >= min_count), f"{role}_min"
                prob += (lpSum([x[i] for i in role_indices]) <= max_count), f"{role}_max"

        # Overseas constraint
        overseas_indices = [
            i for i, p in enumerate(available_players) if p.nationality == "Overseas"
        ]
        if overseas_indices:
            prob += (
                (lpSum([x[i] for i in overseas_indices]) <= CONSTRAINTS.max_overseas),
                "overseas_limit",
            )

        # Franchise spread constraint (max per team)
        teams = set(p.team for p in available_players)
        for team in teams:
            team_indices = [i for i, p in enumerate(available_players) if p.team == team]
            prob += (
                (lpSum([x[i] for i in team_indices]) <= CONSTRAINTS.max_per_team),
                f"team_{team}_limit",
            )

        # Solve
        prob.solve()

        if prob.status != LpStatusOptimal:
            raise RuntimeError(f"Optimizer could not find optimal solution: {prob.status}")

        # Extract solution
        selected_indices = [i for i in range(n) if value(x[i]) > 0.5]
        new_squad = [available_players[i] for i in selected_indices]

        # Identify transfers
        transfers_in = []
        transfers_out = []
        kept_players = []

        for player in new_squad:
            if player.name not in current_squad_set:
                transfers_in.append(player.name)
            else:
                kept_players.append(player.name)

        for player_name in current_squad:
            if player_name not in [p.name for p in new_squad]:
                transfers_out.append(player_name)

        expected_points = sum(p.expected_points for p in new_squad)

        return TransferPlan(
            new_squad=new_squad,
            transfers_in=transfers_in,
            transfers_out=transfers_out,
            num_transfers=len(transfers_in),
            expected_points=expected_points,
            kept_players=kept_players,
        )
