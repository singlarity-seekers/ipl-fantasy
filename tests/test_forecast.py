"""Tests for the forecasting module."""

import numpy as np
import pandas as pd
import pytest

from src.forecast.models import PlayerForecaster, PlayerForecast
from src.forecast.cold_start import (
    detect_uncapped_players,
    generate_uncapped_samples,
    UncappedPlayerProfile,
)


@pytest.fixture
def sample_scorecards():
    """Create sample scorecard data for testing."""
    np.random.seed(42)
    records = []
    players = ["Player_A", "Player_B", "Player_C"]
    roles = ["BAT", "BOWL", "AR"]

    for match_id in range(1, 21):
        for player, role in zip(players, roles):
            records.append({
                "match_id": match_id,
                "player": player,
                "role": role,
                "fantasy_points": np.random.normal(
                    {"BAT": 35, "BOWL": 30, "AR": 40}[role],
                    10,
                ),
            })

    return pd.DataFrame(records)


class TestPlayerForecaster:
    def test_fit(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        assert len(forecaster._player_history) == 3

    def test_forecast_known_player(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        fc = forecaster.forecast("Player_A")

        assert isinstance(fc, PlayerForecast)
        assert fc.player == "Player_A"
        assert fc.expected_points > 0
        assert fc.std_points > 0
        assert fc.quantile_10 < fc.quantile_50 < fc.quantile_90

    def test_forecast_unknown_player_cold_start(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        fc = forecaster.forecast("Unknown_Player", role="BAT")

        assert fc.player == "Unknown_Player"
        assert fc.role == "BAT"
        assert fc.expected_points > 0  # has a prior

    def test_forecast_match(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        forecasts = forecaster.forecast_match(
            players=["Player_A", "Player_B"],
            roles={"Player_A": "BAT", "Player_B": "BOWL"},
        )
        assert len(forecasts) == 2
        assert all(isinstance(f, PlayerForecast) for f in forecasts)


class TestPlayerForecastSampling:
    def test_sample_shape(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        fc = forecaster.forecast("Player_A")

        samples = fc.sample(n=1000)
        assert samples.shape == (1000,)
        assert np.all(samples >= 0)  # non-negative

    def test_sample_mean_close_to_expected(self, sample_scorecards):
        forecaster = PlayerForecaster()
        forecaster.fit(sample_scorecards)
        fc = forecaster.forecast("Player_A")

        samples = fc.sample(n=10_000)
        # Mean of samples should be within ~5 of expected
        assert abs(np.mean(samples) - fc.expected_points) < 10.0


class TestColdStart:
    def test_detect_uncapped(self):
        available = [
            {"name": "New_Player", "role": "BAT", "credit_cost": 6.0},
            {"name": "Known_Player", "role": "BOWL", "credit_cost": 9.0},
        ]
        history = {"Known_Player": 20}

        uncapped = detect_uncapped_players(available, history)
        assert len(uncapped) == 1
        assert uncapped[0].player == "New_Player"

    def test_uncapped_priority_ordering(self):
        available = [
            {"name": "Cheap_AR", "role": "AR", "credit_cost": 5.5},
            {"name": "Expensive_BAT", "role": "BAT", "credit_cost": 9.5},
        ]
        history = {}

        uncapped = detect_uncapped_players(available, history)
        assert len(uncapped) == 2
        # AR with lower cost should rank higher
        assert uncapped[0].player == "Cheap_AR"

    def test_generate_samples(self):
        profile = UncappedPlayerProfile(
            player="Test", role="BAT", credit_cost=6.0, is_uncapped=True,
        )
        samples = generate_uncapped_samples(profile, n=1000)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.mean(samples) > 0
