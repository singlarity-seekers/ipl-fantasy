"""Tests for the Monte Carlo simulation engine."""

import numpy as np
import pytest

from src.forecast.models import PlayerForecast
from src.simulation.monte_carlo import MonteCarloSimulator, SimulationResult


@pytest.fixture
def sample_forecasts():
    """Create sample forecasts for testing."""
    return [
        PlayerForecast(
            player="Batter_A", role="BAT",
            expected_points=40, std_points=15,
            quantile_10=20, quantile_50=38, quantile_90=60,
            ceiling_95=65, floor_5=12,
        ),
        PlayerForecast(
            player="Bowler_A", role="BOWL",
            expected_points=30, std_points=18,
            quantile_10=10, quantile_50=28, quantile_90=55,
            ceiling_95=60, floor_5=5,
        ),
        PlayerForecast(
            player="AR_A", role="AR",
            expected_points=45, std_points=20,
            quantile_10=15, quantile_50=42, quantile_90=70,
            ceiling_95=75, floor_5=8,
        ),
    ]


class TestMonteCarloSimulator:
    def test_simulation_shape(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        assert result.points_matrix.shape == (3, 1000)
        assert len(result.player_names) == 3
        assert result.n_simulations == 1000

    def test_summary_statistics(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=5000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        assert len(result.summary) == 3
        assert "mean_fp" in result.summary.columns
        assert "ceiling_95" in result.summary.columns
        assert "prob_30plus" in result.summary.columns

    def test_mean_convergence(self, sample_forecasts):
        """Mean of simulation should converge to expected points."""
        sim = MonteCarloSimulator(n_simulations=50_000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        for fc in sample_forecasts:
            sim_mean = result.summary[
                result.summary["player"] == fc.player
            ]["mean_fp"].iloc[0]
            # Should be within ~3 of expected
            assert abs(sim_mean - fc.expected_points) < 5.0, (
                f"{fc.player}: sim_mean={sim_mean:.1f}, expected={fc.expected_points}"
            )

    def test_non_negative_points(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)
        assert np.all(result.points_matrix >= 0)

    def test_player_distribution(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        dist = result.get_player_distribution("Batter_A")
        assert len(dist) == 1000

    def test_rank_by(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        ranked = result.rank_by("mean_fp")
        assert len(ranked) == 3
        # AR should generally rank highest (highest expected points)
        assert ranked.iloc[0]["mean_fp"] >= ranked.iloc[-1]["mean_fp"]

    def test_venue_adjustment(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=5000, random_seed=42)

        # Without adjustment
        result_base = sim.simulate_match(sample_forecasts)
        base_mean = result_base.summary[
            result_base.summary["player"] == "Batter_A"
        ]["mean_fp"].iloc[0]

        # With 1.5x adjustment
        sim2 = MonteCarloSimulator(n_simulations=5000, random_seed=42)
        result_adj = sim2.simulate_match(
            sample_forecasts,
            venue_adjustment={"Batter_A": 1.5},
        )
        adj_mean = result_adj.summary[
            result_adj.summary["player"] == "Batter_A"
        ]["mean_fp"].iloc[0]

        assert adj_mean > base_mean

    def test_lineup_scores(self, sample_forecasts):
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        result = sim.simulate_match(sample_forecasts)

        lineup_scores = sim.compute_lineup_scores(
            result,
            lineup_indices=[0, 1, 2],
            captain_idx=2,  # AR as captain
        )
        assert len(lineup_scores) == 1000
        assert np.all(lineup_scores > 0)
