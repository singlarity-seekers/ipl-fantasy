"""
IPL 2026 Schedule loader and query interface.

Provides schedule parsing, match lookup, and fixture density queries
for transfer optimization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScheduleMatch:
    """A single match in the IPL schedule."""

    match_number: int
    date: str  # YYYY-MM-DD
    day: str
    time: str
    home: str
    away: str
    venue: str

    @property
    def teams(self) -> set[str]:
        """Return both teams as a set."""
        return {self.home, self.away}


class Schedule:
    """IPL 2026 schedule manager."""

    def __init__(self, matches: list[ScheduleMatch], season: str = "2026"):
        self.season = season
        self._matches = {m.match_number: m for m in matches}
        self._matches_list = sorted(matches, key=lambda m: m.match_number)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Schedule:
        """Load schedule from JSON file."""
        if path is None:
            path = Path(__file__).resolve().parent.parent.parent / "data" / "schedule_2026.json"

        with open(path) as f:
            data = json.load(f)

        matches = [
            ScheduleMatch(
                match_number=m["match_number"],
                date=m["date"],
                day=m["day"],
                time=m["time"],
                home=m["home"],
                away=m["away"],
                venue=m["venue"],
            )
            for m in data.get("matches", [])
        ]

        return cls(matches, season=data.get("season", "2026"))

    def get_match(self, match_number: int) -> ScheduleMatch | None:
        """Get a match by number."""
        return self._matches.get(match_number)

    def get_upcoming(self, from_match: int, count: int = 5) -> list[ScheduleMatch]:
        """Get upcoming matches starting from a given match number."""
        upcoming = [m for m in self._matches_list if m.match_number >= from_match]
        return upcoming[:count]

    def get_range(self, start_match: int, end_match: int) -> list[ScheduleMatch]:
        """Get matches within a range (inclusive)."""
        return [m for m in self._matches_list if start_match <= m.match_number <= end_match]

    def team_matches(self, team: str, from_match: int = 1) -> list[ScheduleMatch]:
        """Get all remaining matches for a team."""
        return [m for m in self._matches_list if m.match_number >= from_match and team in m.teams]

    def player_match_count(self, team: str, from_match: int, window: int = 5) -> int:
        """
        Count how many matches a team plays in the next `window` matches.

        This is used for look-ahead valuation in transfer optimization.
        Higher fixture density = more opportunities for the player to score.
        """
        upcoming = self.get_upcoming(from_match, window)
        return sum(1 for m in upcoming if team in m.teams)

    def get_max_match_number(self) -> int:
        """Get the highest match number in the schedule."""
        if not self._matches_list:
            return 0
        return self._matches_list[-1].match_number

    def __len__(self) -> int:
        return len(self._matches_list)

    def __iter__(self):
        return iter(self._matches_list)
