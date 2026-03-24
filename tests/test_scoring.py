"""Tests for the IPL Fantasy scoring engine."""

import pytest

from src.scoring.fantasy import (
    PlayerMatchStats,
    compute_batting_points,
    compute_bowling_points,
    compute_economy_points,
    compute_fantasy_points,
    compute_fantasy_points_breakdown,
    compute_fielding_points,
    compute_strike_rate_points,
)


class TestBattingPoints:
    def test_basic_runs(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=30, balls_faced=20, fours=3, sixes=1)
        pts = compute_batting_points(stats)
        # 30 runs + 3 fours + 2 sixes (1*2) + 4 (30 milestone) = 30 + 3 + 2 + 4 = 39
        assert pts == 39.0

    def test_half_century(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=55, balls_faced=35, fours=5, sixes=2)
        pts = compute_batting_points(stats)
        # 55 + 5 + 4 + 8 (50 milestone) = 72
        assert pts == 72.0

    def test_century(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=102, balls_faced=60, fours=10, sixes=4)
        pts = compute_batting_points(stats)
        # 102 + 10 + 8 + 16 (100 milestone) = 136
        assert pts == 136.0

    def test_duck_penalty_batsman(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=0, balls_faced=3, is_out=True)
        pts = compute_batting_points(stats)
        assert pts == -2.0

    def test_duck_penalty_wicketkeeper(self):
        stats = PlayerMatchStats(player_name="Test", role="WK", runs_scored=0, balls_faced=2, is_out=True)
        pts = compute_batting_points(stats)
        assert pts == -2.0

    def test_duck_penalty_allrounder(self):
        stats = PlayerMatchStats(player_name="Test", role="AR", runs_scored=0, balls_faced=1, is_out=True)
        pts = compute_batting_points(stats)
        assert pts == -2.0

    def test_no_duck_penalty_bowler(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", runs_scored=0, balls_faced=2, is_out=True)
        pts = compute_batting_points(stats)
        assert pts == 0.0

    def test_no_duck_not_out(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=0, balls_faced=5, is_out=False)
        pts = compute_batting_points(stats)
        assert pts == 0.0

    def test_zero_runs_no_balls(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=0, balls_faced=0)
        pts = compute_batting_points(stats)
        assert pts == 0.0


class TestBowlingPoints:
    def test_single_wicket(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", wickets=1, overs_bowled=4.0)
        pts = compute_bowling_points(stats)
        assert pts == 25.0

    def test_three_wicket_haul(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", wickets=3, overs_bowled=4.0)
        pts = compute_bowling_points(stats)
        # 3*25 + 4 (3w bonus) = 79
        assert pts == 79.0

    def test_five_wicket_haul(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", wickets=5, overs_bowled=4.0)
        pts = compute_bowling_points(stats)
        # 5*25 + 16 (5w bonus) = 141
        assert pts == 141.0

    def test_lbw_bowled_bonus(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", wickets=2, lbw_bowled_wickets=1, overs_bowled=4.0)
        pts = compute_bowling_points(stats)
        # 2*25 + 1*8 = 58
        assert pts == 58.0

    def test_maiden_over(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", maiden_overs=2, overs_bowled=4.0)
        pts = compute_bowling_points(stats)
        assert pts == 24.0


class TestFieldingPoints:
    def test_catches(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", catches=2)
        pts = compute_fielding_points(stats)
        assert pts == 16.0  # 2 * 8

    def test_three_catches_bonus(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", catches=3)
        pts = compute_fielding_points(stats)
        # 3*8 + 4 = 28
        assert pts == 28.0

    def test_stumping(self):
        stats = PlayerMatchStats(player_name="Test", role="WK", stumpings=1)
        pts = compute_fielding_points(stats)
        assert pts == 12.0

    def test_run_out_direct(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", run_out_direct=1)
        pts = compute_fielding_points(stats)
        assert pts == 12.0

    def test_run_out_indirect(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", run_out_indirect=1)
        pts = compute_fielding_points(stats)
        assert pts == 6.0


class TestEconomyRate:
    def test_excellent_economy(self):
        # 12 balls (2 overs), 8 runs → economy = 4.0
        stats = PlayerMatchStats(player_name="Test", role="BOWL", overs_bowled=2.0, runs_conceded=8)
        pts = compute_economy_points(stats)
        assert pts == 6.0

    def test_poor_economy(self):
        # 12 balls, 26 runs → economy = 13.0
        stats = PlayerMatchStats(player_name="Test", role="BOWL", overs_bowled=2.0, runs_conceded=26)
        pts = compute_economy_points(stats)
        assert pts == -6.0

    def test_less_than_two_overs_no_bonus(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", overs_bowled=1.5, runs_conceded=5)
        pts = compute_economy_points(stats)
        assert pts == 0.0


class TestStrikeRate:
    def test_high_strike_rate(self):
        # 20 runs off 10 balls → SR = 200
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=20, balls_faced=10)
        pts = compute_strike_rate_points(stats)
        assert pts == 6.0

    def test_low_strike_rate(self):
        # 4 runs off 10 balls → SR = 40
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=4, balls_faced=10)
        pts = compute_strike_rate_points(stats)
        assert pts == -6.0

    def test_bowler_exempt(self):
        stats = PlayerMatchStats(player_name="Test", role="BOWL", runs_scored=2, balls_faced=10)
        pts = compute_strike_rate_points(stats)
        assert pts == 0.0

    def test_less_than_10_balls_no_bonus(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", runs_scored=0, balls_faced=5)
        pts = compute_strike_rate_points(stats)
        assert pts == 0.0


class TestTotalFantasyPoints:
    def test_participation_bonus(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", in_playing_xi=True)
        pts = compute_fantasy_points(stats)
        assert pts >= 4.0  # at least participation bonus

    def test_not_in_xi(self):
        stats = PlayerMatchStats(player_name="Test", role="BAT", in_playing_xi=False)
        pts = compute_fantasy_points(stats)
        assert pts == 0.0

    def test_allrounder_full_game(self):
        """Test an all-rounder with batting + bowling + fielding contributions."""
        stats = PlayerMatchStats(
            player_name="Jadeja",
            role="AR",
            runs_scored=35, balls_faced=22, fours=3, sixes=1,
            is_out=True,
            wickets=2, lbw_bowled_wickets=1, overs_bowled=4.0,
            runs_conceded=28, maiden_overs=0,
            catches=1,
            in_playing_xi=True,
        )
        pts = compute_fantasy_points(stats)

        # Verify via breakdown
        bd = compute_fantasy_points_breakdown(stats)
        assert bd["total"] == pts
        assert bd["participation"] == 4.0
        assert bd["batting"] > 0
        assert bd["bowling"] > 0
        assert bd["fielding"] > 0

    def test_breakdown_sums_correctly(self):
        stats = PlayerMatchStats(
            player_name="Test", role="BAT",
            runs_scored=50, balls_faced=30, fours=5, sixes=2,
            in_playing_xi=True,
        )
        bd = compute_fantasy_points_breakdown(stats)
        expected = sum(v for k, v in bd.items() if k != "total")
        assert abs(bd["total"] - expected) < 0.01
