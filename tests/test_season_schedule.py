"""Tests for the season schedule module."""

import pytest
from pathlib import Path

from src.season.schedule import Schedule, ScheduleMatch


class TestSchedule:
    """Test suite for Schedule class."""

    def test_load_schedule(self):
        """Test loading schedule from JSON."""
        schedule = Schedule.load()

        assert schedule.season == "2026"
        assert len(schedule) == 20  # First 20 matches from Part 1
        assert schedule.get_max_match_number() == 20

    def test_get_match(self):
        """Test getting a specific match."""
        schedule = Schedule.load()

        match = schedule.get_match(1)
        assert match is not None
        assert match.home == "Royal Challengers Bengaluru"
        assert match.away == "Sunrisers Hyderabad"
        assert match.venue == "Bengaluru"

        # Non-existent match
        assert schedule.get_match(999) is None

    def test_get_upcoming(self):
        """Test getting upcoming matches."""
        schedule = Schedule.load()

        upcoming = schedule.get_upcoming(from_match=1, count=3)
        assert len(upcoming) == 3
        assert upcoming[0].match_number == 1
        assert upcoming[1].match_number == 2
        assert upcoming[2].match_number == 3

    def test_get_range(self):
        """Test getting match range."""
        schedule = Schedule.load()

        matches = schedule.get_range(5, 8)
        assert len(matches) == 4
        assert matches[0].match_number == 5
        assert matches[-1].match_number == 8

    def test_team_matches(self):
        """Test getting team matches."""
        schedule = Schedule.load()

        rcb_matches = schedule.team_matches("Royal Challengers Bengaluru", from_match=1)
        assert len(rcb_matches) >= 3  # RCB plays at least 3 in first 20

        # All should include RCB
        for match in rcb_matches:
            assert "Royal Challengers Bengaluru" in match.teams

    def test_player_match_count(self):
        """Test fixture density calculation."""
        schedule = Schedule.load()

        # RCB plays how many times in first 5 matches?
        count = schedule.player_match_count("Royal Challengers Bengaluru", from_match=1, window=5)
        assert isinstance(count, int)
        assert count >= 0

    def test_schedule_match_teams_property(self):
        """Test the teams property."""
        match = ScheduleMatch(
            match_number=1,
            date="2026-03-28",
            day="Sat",
            time="7:30 PM",
            home="Royal Challengers Bengaluru",
            away="Sunrisers Hyderabad",
            venue="Bengaluru",
        )

        assert match.teams == {"Royal Challengers Bengaluru", "Sunrisers Hyderabad"}
