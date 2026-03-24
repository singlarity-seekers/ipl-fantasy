"""
Data ingestion for the IPL Fantasy system.

Loads the Kaggle IPL Complete Dataset (2008–2024) consisting of:
  - matches.csv  → match-level summaries
  - deliveries.csv → ball-by-ball data
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW

logger = logging.getLogger(__name__)


# ── Column name normalization ─────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names for consistency."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_matches(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load matches.csv from the raw data directory.

    Columns expected (after normalization):
        id, season, city, date, match_type, player_of_match,
        venue, team1, team2, toss_winner, toss_decision,
        winner, result, result_margin, target_runs, target_overs,
        super_over, method, umpire1, umpire2
    """
    data_dir = data_dir or DATA_RAW
    path = data_dir / "matches.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"matches.csv not found at {path}. "
            "Please download the IPL dataset from Kaggle and place it in data/raw/."
        )

    logger.info("Loading matches from %s", path)
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Loaded %d matches", len(df))
    return df


def load_deliveries(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load deliveries.csv from the raw data directory.

    Columns expected (after normalization):
        match_id, inning, over, ball, batter, bowler,
        non_striker, batsman_runs, extra_runs, total_runs,
        extras_type, is_wicket, player_dismissed,
        dismissal_kind, fielder
    """
    data_dir = data_dir or DATA_RAW
    path = data_dir / "deliveries.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"deliveries.csv not found at {path}. "
            "Please download the IPL dataset from Kaggle and place it in data/raw/."
        )

    logger.info("Loading deliveries from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)

    logger.info("Loaded %d deliveries", len(df))
    return df


def load_dataset(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both matches and deliveries DataFrames."""
    matches = load_matches(data_dir)
    deliveries = load_deliveries(data_dir)
    return matches, deliveries


# ── Merge helper ──────────────────────────────────────────────────────────────

def merge_match_context(
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge match-level context (venue, date, teams, toss) into
    the ball-by-ball deliveries DataFrame.

    Returns a new DataFrame with match metadata joined on match_id/id.
    """
    match_cols = [
        "id", "season", "city", "date", "venue",
        "team1", "team2", "toss_winner", "toss_decision",
        "winner",
    ]
    available_cols = [c for c in match_cols if c in matches.columns]

    merged = deliveries.merge(
        matches[available_cols],
        left_on="match_id",
        right_on="id",
        how="left",
    )
    # Drop duplicate id column
    if "id" in merged.columns and "match_id" in merged.columns:
        merged = merged.drop(columns=["id"])

    return merged


# ── Caching to Parquet ────────────────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    name: str,
    output_dir: Path | None = None,
) -> Path:
    """Save a DataFrame as a Parquet file for faster re-loading."""
    output_dir = output_dir or DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s → %s (%d rows)", name, path, len(df))
    return path


def load_processed(
    name: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a previously saved Parquet file."""
    output_dir = output_dir or DATA_PROCESSED
    path = output_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)
