"""Tests for the season state management module."""

import json
import pytest
from pathlib import Path
import tempfile
import os

from src.season.state import SeasonState, SeasonStateManager, TransferRecord
from src.config import CONSTRAINTS


class TestTransferRecord:
    """Test suite for TransferRecord dataclass."""

    def test_transfer_record_creation(self):
        """Test creating a transfer record."""
        record = TransferRecord(
            match_number=5,
            player_out="Virat Kohli",
            player_in="Rohit Sharma",
            timestamp="2026-04-01T10:00:00",
            reason="optimization",
        )

        assert record.match_number == 5
        assert record.player_out == "Virat Kohli"
        assert record.player_in == "Rohit Sharma"
        assert record.reason == "optimization"


class TestSeasonState:
    """Test suite for SeasonState dataclass."""

    def test_season_state_creation(self):
        """Test creating season state."""
        squad = ["Player " + str(i) for i in range(11)]
        state = SeasonState(
            squad=squad,
            transfers_used=5,
            current_match=3,
        )

        assert len(state.squad) == 11
        assert state.transfers_used == 5
        assert state.current_match == 3
        assert state.season == "2026"

    def test_season_state_with_history(self):
        """Test season state with transfer history."""
        state = SeasonState(
            squad=["P" + str(i) for i in range(11)],
            history=[
                TransferRecord(2, "Old1", "New1", "2026-03-29T10:00:00"),
                TransferRecord(3, "Old2", "New2", "2026-04-01T10:00:00"),
            ],
        )

        assert len(state.history) == 2


class TestSeasonStateManager:
    """Test suite for SeasonStateManager."""

    def test_init_squad(self, tmp_path):
        """Test initializing a squad."""
        state_path = tmp_path / "test_state.json"
        manager = SeasonStateManager(state_path)

        squad = [f"Player {i}" for i in range(11)]
        state = manager.init_squad(squad, match_number=1)

        assert state.squad == squad
        assert state.transfers_used == 0
        assert state.current_match == 1
        assert state_path.exists()

    def test_init_squad_wrong_size(self, tmp_path):
        """Test initializing squad with wrong size raises error."""
        manager = SeasonStateManager(tmp_path / "test.json")

        with pytest.raises(ValueError, match="Squad must have exactly"):
            manager.init_squad(["P1", "P2", "P3"])  # Only 3 players

    def test_load_save_state(self, tmp_path):
        """Test loading and saving state."""
        state_path = tmp_path / "test_state.json"

        # Save state
        manager = SeasonStateManager(state_path)
        squad = [f"Player {i}" for i in range(11)]
        state1 = manager.init_squad(squad)

        # Load state
        manager2 = SeasonStateManager(state_path)
        state2 = manager2.load()

        assert state2.squad == state1.squad
        assert state2.transfers_used == state1.transfers_used

    def test_remaining_transfers_league_stage(self, tmp_path):
        """Test remaining transfers in league stage."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(
            squad=[f"P{i}" for i in range(11)],
            transfers_used=50,
            current_match=30,
        )

        remaining = manager.remaining_transfers(state)
        assert remaining == 160 - 50  # 110 remaining

    def test_remaining_transfers_playoffs(self, tmp_path):
        """Test remaining transfers in playoffs."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(
            squad=[f"P{i}" for i in range(11)],
            transfers_used=160,  # Used all league stage budget
            current_match=71,  # Playoffs
        )

        remaining = manager.remaining_transfers(state)
        assert remaining == 10  # Playoff budget

    def test_apply_transfers(self, tmp_path):
        """Test applying transfers."""
        manager = SeasonStateManager(tmp_path / "test.json")

        initial_squad = [f"Player {i}" for i in range(11)]
        state = manager.init_squad(initial_squad, match_number=1)

        # Apply 2 transfers
        new_state = manager.apply_transfers(
            state,
            transfers_in=["New1", "New2"],
            transfers_out=["Player 0", "Player 1"],
            match_number=2,
            reason="optimization",
        )

        assert new_state.transfers_used == 2
        assert len(new_state.history) == 2
        assert "New1" in new_state.squad
        assert "Player 0" not in new_state.squad

    def test_apply_transfers_budget_exceeded(self, tmp_path):
        """Test that exceeding transfer budget raises error."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(
            squad=[f"P{i}" for i in range(11)],
            transfers_used=159,
            current_match=50,
        )

        with pytest.raises(ValueError, match="Transfer budget exceeded"):
            manager.apply_transfers(
                state,
                transfers_in=["New1", "New2"],
                transfers_out=["P0", "P1"],
                match_number=51,
            )

    def test_apply_transfers_player_not_in_squad(self, tmp_path):
        """Test transferring out player not in squad raises error."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(
            squad=[f"P{i}" for i in range(11)],
        )

        with pytest.raises(ValueError, match="not in current squad"):
            manager.apply_transfers(
                state,
                transfers_in=["New1"],
                transfers_out=["NonExistentPlayer"],
                match_number=2,
            )

    def test_set_squad(self, tmp_path):
        """Test manually setting squad."""
        manager = SeasonStateManager(tmp_path / "test.json")

        initial_squad = [f"Player {i}" for i in range(11)]
        state = manager.init_squad(initial_squad, match_number=1)

        new_squad = [f"NewPlayer {i}" for i in range(11)]
        new_state = manager.set_squad(state, new_squad, match_number=2)

        assert new_state.squad == new_squad
        assert new_state.current_match == 2

    def test_can_use_booster(self, tmp_path):
        """Test booster availability check."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(
            squad=[f"P{i}" for i in range(11)],
            boosters_used={"triple_captain": 1, "free_hit": 0},
        )

        assert manager.can_use_booster(state, "triple_captain") == True  # 1/2 used
        assert manager.can_use_booster(state, "free_hit") == True  # 0/1 used
        assert manager.can_use_booster(state, "wildcard") == True  # 0/1 used

    def test_use_booster(self, tmp_path):
        """Test using a booster."""
        manager = SeasonStateManager(tmp_path / "test.json")

        state = SeasonState(squad=[f"P{i}" for i in range(11)])

        new_state = manager.use_booster(state, "triple_captain")

        assert new_state.boosters_used["triple_captain"] == 1

    def test_free_hit_snapshot(self, tmp_path):
        """Test Free Hit saves and restores squad."""
        manager = SeasonStateManager(tmp_path / "test.json")

        initial_squad = [f"Player {i}" for i in range(11)]
        state = manager.init_squad(initial_squad, match_number=1)

        # Use Free Hit
        state = manager.use_booster(state, "free_hit")

        # Snapshot should be saved
        assert state.free_hit_snapshot == initial_squad
        assert state.free_hit_match == 1

        # After applying transfers at same match, squad should restore
        state = manager.apply_transfers(
            state,
            transfers_in=["New1", "New2"],
            transfers_out=["Player 0", "Player 1"],
            match_number=1,  # Same match as free hit
        )

        # Squad should be restored to original
        assert state.squad == initial_squad
        assert state.free_hit_snapshot is None
