"""Tests for the Captain/VC selector."""

import numpy as np
import pytest

from src.captain.selector import CaptainSelector, CaptainStrategy
from src.forecast.models import PlayerForecast
from src.simulation.monte_carlo import MonteCarloSimulator


@pytest.fixture
def sim_result():
    """Create a simulation result for testing."""
    forecasts = [
        PlayerForecast("Star_Bat", "BAT", 50, 15, 30, 48, 70, 75, 25),
        PlayerForecast("Good_Bowl", "BOWL", 35, 18, 10, 32, 58, 62, 5),
        PlayerForecast("Great_AR", "AR", 45, 20, 15, 42, 68, 72, 8),
        PlayerForecast("Safe_WK", "WK", 28, 10, 15, 27, 40, 42, 12),
        PlayerForecast("Avg_Bat", "BAT", 30, 12, 12, 28, 45, 50, 8),
    ]
    sim = MonteCarloSimulator(n_simulations=5000, random_seed=42)
    return sim.simulate_match(forecasts)


class TestCaptainSelector:
    def test_safe_picks_high_expected(self, sim_result):
        selector = CaptainSelector()
        lineup = ["Star_Bat", "Good_Bowl", "Great_AR", "Safe_WK", "Avg_Bat"]

        pick = selector.select(lineup, sim_result, CaptainStrategy.SAFE)

        assert pick.captain in lineup
        assert pick.vice_captain in lineup
        assert pick.captain != pick.vice_captain
        # Star batter or AR should be captain (highest expected)
        assert pick.captain in ["Star_Bat", "Great_AR"]

    def test_captain_always_in_lineup(self, sim_result):
        selector = CaptainSelector()
        lineup = ["Star_Bat", "Good_Bowl", "Great_AR"]

        pick = selector.select(lineup, sim_result, CaptainStrategy.SAFE)

        assert pick.captain in lineup
        assert pick.vice_captain in lineup

    def test_differential_strategy(self, sim_result):
        selector = CaptainSelector()
        lineup = ["Star_Bat", "Good_Bowl", "Great_AR", "Safe_WK", "Avg_Bat"]

        pick = selector.select(
            lineup, sim_result, CaptainStrategy.DIFFERENTIAL,
            ownership_pcts={"Star_Bat": 60, "Great_AR": 30, "Good_Bowl": 5},
        )

        assert pick.captain in lineup
        assert pick.strategy == CaptainStrategy.DIFFERENTIAL

    def test_captain_impact_evaluation(self, sim_result):
        selector = CaptainSelector()
        lineup = ["Star_Bat", "Good_Bowl", "Great_AR"]

        impact = selector.evaluate_captain_impact(sim_result, lineup)

        assert len(impact) == 3
        assert "e_total_as_captain" in impact.columns
        # Higher expected player as captain should give higher total
        assert impact.iloc[0]["e_total_as_captain"] >= impact.iloc[-1]["e_total_as_captain"]
