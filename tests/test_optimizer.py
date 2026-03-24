"""Tests for the IPL Fantasy ILP optimizer."""

import pytest

from src.config import CONSTRAINTS
from src.optimizer.fantasy_ilp import IPLFantasyOptimizer, OptimizedLineup, PlayerSlot


@pytest.fixture
def player_pool():
    """Create a realistic player pool for two IPL teams."""
    players = [
        # Team A — WK, BAT, AR, BOWL
        PlayerSlot("WK_A1", "WK", "Team_A", 9.0, 35, 55),
        PlayerSlot("WK_A2", "WK", "Team_A", 7.0, 25, 40),
        PlayerSlot("BAT_A1", "BAT", "Team_A", 10.0, 45, 70),
        PlayerSlot("BAT_A2", "BAT", "Team_A", 9.0, 38, 60),
        PlayerSlot("BAT_A3", "BAT", "Team_A", 8.0, 32, 50),
        PlayerSlot("BAT_A4", "BAT", "Team_A", 7.5, 28, 45),
        PlayerSlot("AR_A1", "AR", "Team_A", 9.5, 42, 68),
        PlayerSlot("AR_A2", "AR", "Team_A", 8.5, 35, 55),
        PlayerSlot("BOWL_A1", "BOWL", "Team_A", 9.0, 34, 58),
        PlayerSlot("BOWL_A2", "BOWL", "Team_A", 8.0, 30, 50),
        PlayerSlot("BOWL_A3", "BOWL", "Team_A", 7.0, 26, 42),
        # Team B — WK, BAT, AR, BOWL
        PlayerSlot("WK_B1", "WK", "Team_B", 8.5, 30, 48),
        PlayerSlot("BAT_B1", "BAT", "Team_B", 10.5, 48, 75),
        PlayerSlot("BAT_B2", "BAT", "Team_B", 9.0, 36, 58),
        PlayerSlot("BAT_B3", "BAT", "Team_B", 7.5, 27, 44),
        PlayerSlot("BAT_B4", "BAT", "Team_B", 6.5, 22, 38),
        PlayerSlot("AR_B1", "AR", "Team_B", 9.0, 38, 62),
        PlayerSlot("AR_B2", "AR", "Team_B", 7.5, 30, 48),
        PlayerSlot("BOWL_B1", "BOWL", "Team_B", 9.5, 36, 60),
        PlayerSlot("BOWL_B2", "BOWL", "Team_B", 8.0, 28, 46),
        PlayerSlot("BOWL_B3", "BOWL", "Team_B", 6.5, 20, 35),
    ]
    return players


class TestIPLFantasyOptimizer:
    def test_generates_valid_lineup(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool)

        assert lineup is not None
        assert len(lineup.players) == CONSTRAINTS.squad_size
        violations = lineup.validate()
        assert violations == [], f"Violations: {violations}"

    def test_budget_constraint(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool)

        assert lineup is not None
        assert lineup.total_credits <= CONSTRAINTS.total_credits

    def test_role_limits(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool)

        assert lineup is not None
        counts = lineup.role_counts
        for role, min_c, max_c in CONSTRAINTS.role_limits:
            assert min_c <= counts.get(role, 0) <= max_c, (
                f"Role {role}: {counts.get(role, 0)} not in [{min_c}, {max_c}]"
            )

    def test_team_cap(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool)

        assert lineup is not None
        for team, count in lineup.team_counts.items():
            assert count <= CONSTRAINTS.max_per_team, (
                f"Team {team}: {count} > {CONSTRAINTS.max_per_team}"
            )

    def test_must_include(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool, must_include={"BAT_B1"})

        assert lineup is not None
        names = {p.name for p in lineup.players}
        assert "BAT_B1" in names

    def test_excluded_players(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool, excluded={"BAT_A1", "BAT_B1"})

        assert lineup is not None
        names = {p.name for p in lineup.players}
        assert "BAT_A1" not in names
        assert "BAT_B1" not in names

    def test_objective_ceiling(self, player_pool):
        opt_expected = IPLFantasyOptimizer(objective="expected")
        opt_ceiling = IPLFantasyOptimizer(objective="ceiling")

        lineup_exp = opt_expected.optimize(player_pool)
        lineup_ceil = opt_ceiling.optimize(player_pool)

        assert lineup_exp is not None
        assert lineup_ceil is not None

        # Ceiling optimizer should have higher total ceiling
        ceil_sum_exp = sum(p.ceiling_95 for p in lineup_exp.players)
        ceil_sum_ceil = sum(p.ceiling_95 for p in lineup_ceil.players)
        assert ceil_sum_ceil >= ceil_sum_exp * 0.95  # ceiling should be at least close

    def test_top_k_diversity(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineups = optimizer.generate_top_k(player_pool, k=3, min_diff=2)

        assert len(lineups) >= 2  # should get at least 2

        # Check diversity: any two lineups differ by at least 2 players
        for i in range(len(lineups)):
            for j in range(i + 1, len(lineups)):
                names_i = {p.name for p in lineups[i].players}
                names_j = {p.name for p in lineups[j].players}
                overlap = len(names_i & names_j)
                assert overlap <= CONSTRAINTS.squad_size - 2, (
                    f"Lineups {i} and {j} overlap by {overlap}"
                )

    def test_all_generated_lineups_valid(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineups = optimizer.generate_top_k(player_pool, k=5)

        for i, lineup in enumerate(lineups):
            violations = lineup.validate()
            assert violations == [], f"Lineup {i} violations: {violations}"


class TestOptimizedLineup:
    def test_repr(self, player_pool):
        optimizer = IPLFantasyOptimizer()
        lineup = optimizer.optimize(player_pool)
        assert lineup is not None
        r = repr(lineup)
        assert "Lineup" in r
        assert "E[pts]" in r
