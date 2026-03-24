"""
Season state persistence for transfer tracking.

Manages squad state across matches, tracks transfers used, boosters used,
and maintains a history of all transfers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import CONSTRAINTS


@dataclass
class TransferRecord:
    """Record of a single transfer."""

    match_number: int
    player_out: str
    player_in: str
    timestamp: str
    reason: str = ""  # e.g., "optimization", "injury", "manual"


@dataclass
class SeasonState:
    """
    Current state of the fantasy squad throughout the season.
    """

    squad: list[str]  # 11 player names
    transfers_used: int = 0
    boosters_used: dict[str, int] = field(default_factory=dict)
    current_match: int = 1
    history: list[TransferRecord] = field(default_factory=list)

    # Free Hit snapshot (saves pre-free-hit squad)
    free_hit_snapshot: list[str] | None = None
    free_hit_match: int | None = None

    # Season metadata
    season: str = "2026"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SeasonStateManager:
    """
    Manages persistence and operations on SeasonState.
    """

    LEAGUE_STAGE_BUDGET = 160
    PLAYOFF_BUDGET = 10
    LEAGUE_STAGE_END = 70

    def __init__(self, state_path: Path | str | None = None):
        if state_path is None:
            state_path = (
                Path(__file__).resolve().parent.parent.parent / "data" / "season_state.json"
            )
        self.state_path = Path(state_path)
        self._state: SeasonState | None = None

    def load(self) -> SeasonState:
        """Load state from disk or create default."""
        if not self.state_path.exists():
            # Return empty state - user must call init_squad
            return SeasonState(squad=[])

        with open(self.state_path) as f:
            data = json.load(f)

        # Deserialize history
        history = [TransferRecord(**record) for record in data.get("history", [])]

        self._state = SeasonState(
            squad=data.get("squad", []),
            transfers_used=data.get("transfers_used", 0),
            boosters_used=data.get("boosters_used", {}),
            current_match=data.get("current_match", 1),
            history=history,
            free_hit_snapshot=data.get("free_hit_snapshot"),
            free_hit_match=data.get("free_hit_match"),
            season=data.get("season", "2026"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
        return self._state

    def save(self, state: SeasonState | None = None) -> None:
        """Save state to disk."""
        if state is None:
            state = self._state
        if state is None:
            raise ValueError("No state to save")

        state.updated_at = datetime.now().isoformat()

        # Convert to dict for JSON serialization
        data = asdict(state)

        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)

        self._state = state

    def init_squad(
        self, players: list[str], match_number: int = 1, season: str = "2026"
    ) -> SeasonState:
        """Initialize a new squad at the start of the season."""
        if len(players) != CONSTRAINTS.squad_size:
            raise ValueError(
                f"Squad must have exactly {CONSTRAINTS.squad_size} players, got {len(players)}"
            )

        state = SeasonState(
            squad=players.copy(),
            transfers_used=0,
            boosters_used={},
            current_match=match_number,
            history=[],
            season=season,
        )

        self.save(state)
        return state

    def remaining_transfers(self, state: SeasonState | None = None) -> int:
        """Calculate remaining transfers based on match number."""
        if state is None:
            state = self._state or self.load()

        if state.current_match > self.LEAGUE_STAGE_END:
            # Playoffs: separate budget
            playoff_used = max(0, state.transfers_used - self.LEAGUE_STAGE_BUDGET)
            return self.PLAYOFF_BUDGET - playoff_used
        else:
            # League stage
            return self.LEAGUE_STAGE_BUDGET - state.transfers_used

    def can_use_booster(self, state: SeasonState, booster_name: str) -> bool:
        """Check if a booster can be used (respects max uses)."""
        from src.config import BOOSTERS

        if booster_name not in BOOSTERS:
            return False

        max_uses = BOOSTERS[booster_name]["max_uses"]
        used = state.boosters_used.get(booster_name, 0)
        return used < max_uses

    def use_booster(self, state: SeasonState, booster_name: str) -> SeasonState:
        """Record booster usage."""
        from src.config import BOOSTERS

        if not self.can_use_booster(state, booster_name):
            raise ValueError(f"Cannot use booster '{booster_name}' - max uses reached")

        state.boosters_used[booster_name] = state.boosters_used.get(booster_name, 0) + 1

        # Handle Free Hit special case: save current squad
        if booster_name == "free_hit":
            state.free_hit_snapshot = state.squad.copy()
            state.free_hit_match = state.current_match

        self.save(state)
        return state

    def apply_transfers(
        self,
        state: SeasonState,
        transfers_in: list[str],
        transfers_out: list[str],
        match_number: int,
        reason: str = "optimization",
    ) -> SeasonState:
        """
        Apply transfers to the squad.

        Args:
            state: Current season state
            transfers_in: Players to add
            transfers_out: Players to remove
            match_number: Match number for this transfer
            reason: Reason for the transfer

        Returns:
            Updated SeasonState

        Raises:
            ValueError: If transfer budget exceeded or invalid players
        """
        num_transfers = len(transfers_in)

        if num_transfers != len(transfers_out):
            raise ValueError(
                f"Transfers in ({num_transfers}) must equal transfers out ({len(transfers_out)})"
            )

        remaining = self.remaining_transfers(state)
        if num_transfers > remaining:
            raise ValueError(
                f"Transfer budget exceeded: {num_transfers} requested, {remaining} remaining"
            )

        # Validate transfers_out are in current squad
        current_squad_set = set(state.squad)
        for player in transfers_out:
            if player not in current_squad_set:
                raise ValueError(f"Player '{player}' not in current squad")

        # Validate transfers_in are NOT in current squad
        for player in transfers_in:
            if player in current_squad_set:
                raise ValueError(f"Player '{player}' already in squad")

        # Apply transfers
        new_squad = [p for p in state.squad if p not in transfers_out]
        new_squad.extend(transfers_in)

        if len(new_squad) != CONSTRAINTS.squad_size:
            raise ValueError(f"Squad size error after transfers: {len(new_squad)} players")

        state.squad = new_squad
        state.transfers_used += num_transfers
        state.current_match = match_number

        # Record in history
        for i, (out_player, in_player) in enumerate(zip(transfers_out, transfers_in)):
            record = TransferRecord(
                match_number=match_number,
                player_out=out_player,
                player_in=in_player,
                timestamp=datetime.now().isoformat(),
                reason=reason if i == 0 else f"{reason} (multi)",
            )
            state.history.append(record)

        # Handle Free Hit restore if applicable
        if state.free_hit_match == match_number and state.free_hit_snapshot is not None:
            # Restore pre-free-hit squad after the match
            state.squad = state.free_hit_snapshot
            state.free_hit_snapshot = None
            state.free_hit_match = None

        self.save(state)
        return state

    def set_squad(
        self, state: SeasonState, players: list[str], match_number: int | None = None
    ) -> SeasonState:
        """Manually set squad (for testing or manual overrides)."""
        if len(players) != CONSTRAINTS.squad_size:
            raise ValueError(f"Squad must have exactly {CONSTRAINTS.squad_size} players")

        old_squad_set = set(state.squad) if state.squad else set()
        new_squad_set = set(players)

        # Calculate implied transfers
        transfers_out = list(old_squad_set - new_squad_set)
        transfers_in = list(new_squad_set - old_squad_set)

        if transfers_in or transfers_out:
            # Record as manual transfer
            for out_player, in_player in zip(transfers_out, transfers_in):
                record = TransferRecord(
                    match_number=match_number or state.current_match,
                    player_out=out_player,
                    player_in=in_player,
                    timestamp=datetime.now().isoformat(),
                    reason="manual_set",
                )
                state.history.append(record)

            state.transfers_used += len(transfers_in)

        state.squad = players.copy()
        if match_number:
            state.current_match = match_number

        self.save(state)
        return state

    def get_state(self) -> SeasonState:
        """Get current state (loads if not already loaded)."""
        if self._state is None:
            return self.load()
        return self._state
