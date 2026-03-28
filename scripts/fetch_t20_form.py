#!/usr/bin/env python3
"""
Fetch recent T20 form data (last 12 months) for IPL 2026 squad players.

Downloads ball-by-ball data from Cricsheet for all major T20 leagues,
filters to the last 12 months, and computes per-player aggregates.

Output: data/raw/t20_recent_form.csv

Leagues covered:
  - T20 Internationals
  - IPL, BBL, SA20, PSL, CPL, T20 Blast, ILT20, LPL, MLC, The Hundred

Usage:
    python scripts/fetch_t20_form.py
    python scripts/fetch_t20_form.py --months 6   # last 6 months only
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "data" / "tmp_t20"
OUTPUT_PATH = DATA_RAW / "t20_recent_form.csv"

# Cricsheet T20 JSON download URLs
T20_SOURCES = {
    "T20I": "https://cricsheet.org/downloads/t20s_json.zip",
    "IPL": "https://cricsheet.org/downloads/ipl_json.zip",
    "BBL": "https://cricsheet.org/downloads/bbl_json.zip",
    "SA20": "https://cricsheet.org/downloads/sat_json.zip",
    "PSL": "https://cricsheet.org/downloads/psl_json.zip",
    "CPL": "https://cricsheet.org/downloads/cpl_json.zip",
    "T20Blast": "https://cricsheet.org/downloads/ntb_json.zip",
    "ILT20": "https://cricsheet.org/downloads/ilt_json.zip",
    "LPL": "https://cricsheet.org/downloads/lpl_json.zip",
    "MLC": "https://cricsheet.org/downloads/mlc_json.zip",
    "Hundred": "https://cricsheet.org/downloads/hnd_json.zip",
    "SMAT": "https://cricsheet.org/downloads/sma_json.zip",
}


def download_zip(url: str, label: str) -> bytes:
    """Download a zip file from URL, return raw bytes."""
    print(f"  Downloading {label}...", end=" ", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ipl-fantasy/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"{len(data) / 1024 / 1024:.1f} MB")
    return data


def parse_matches_from_zip(zip_bytes: bytes, cutoff_date: str, league: str) -> list[dict]:
    """
    Parse Cricsheet JSON zip, extract per-player match stats
    for matches on or after cutoff_date.

    Returns list of dicts with player-level stats per match.
    """
    records = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]

        for fname in json_files:
            try:
                raw = zf.read(fname)
                match = json.loads(raw)
            except (json.JSONDecodeError, KeyError):
                continue

            info = match.get("info", {})

            # Get match date
            dates = info.get("dates", [])
            if not dates:
                continue
            match_date = dates[0]
            if match_date < cutoff_date:
                continue

            venue = info.get("venue", "")
            teams = info.get("teams", [])
            match_id = fname.replace(".json", "")

            # Build per-player stats from innings
            player_stats: dict[str, dict] = {}

            for innings in match.get("innings", []):
                batting_team = innings.get("team", "")

                for over_data in innings.get("overs", []):
                    over_num = over_data.get("over", 0)

                    for delivery in over_data.get("deliveries", []):
                        batter = delivery.get("batter", "")
                        bowler = delivery.get("bowler", "")
                        runs_batter = delivery.get("runs", {}).get("batter", 0)
                        runs_extras = delivery.get("runs", {}).get("extras", 0)
                        runs_total = delivery.get("runs", {}).get("total", 0)

                        # Initialize player entries
                        for name in [batter, bowler]:
                            if name and name not in player_stats:
                                player_stats[name] = {
                                    "player": name,
                                    "match_id": match_id,
                                    "date": match_date,
                                    "league": league,
                                    "venue": venue,
                                    "runs": 0,
                                    "balls_faced": 0,
                                    "fours": 0,
                                    "sixes": 0,
                                    "is_out": False,
                                    "balls_bowled": 0,
                                    "runs_conceded": 0,
                                    "wickets": 0,
                                    "catches": 0,
                                    "played": True,
                                }

                        # Batting stats
                        if batter in player_stats:
                            ps = player_stats[batter]
                            ps["runs"] += runs_batter
                            # Only count legal deliveries for balls faced
                            extras_type = delivery.get("extras", {})
                            if "wides" not in extras_type:
                                ps["balls_faced"] += 1
                            if runs_batter == 4:
                                ps["fours"] += 1
                            elif runs_batter == 6:
                                ps["sixes"] += 1

                        # Bowling stats
                        if bowler in player_stats:
                            ps = player_stats[bowler]
                            extras_type = delivery.get("extras", {})
                            if "wides" not in extras_type and "noballs" not in extras_type:
                                ps["balls_bowled"] += 1
                            elif "noballs" in extras_type:
                                ps["balls_bowled"] += 0  # no-balls don't count
                            ps["runs_conceded"] += runs_total - extras_type.get("legbyes", 0) - extras_type.get("byes", 0)

                        # Wickets
                        for wicket in delivery.get("wickets", []):
                            kind = wicket.get("kind", "")
                            dismissed = wicket.get("player_out", "")

                            if dismissed in player_stats:
                                player_stats[dismissed]["is_out"] = True

                            if kind not in ("run out", "retired hurt", "retired out", "obstructing the field"):
                                if bowler in player_stats:
                                    player_stats[bowler]["wickets"] += 1

                            # Catches
                            for fielder_info in wicket.get("fielders", []):
                                fielder = fielder_info.get("name", "")
                                if fielder and kind == "caught" and fielder in player_stats:
                                    player_stats[fielder]["catches"] += 1

            records.extend(player_stats.values())

    return records


def load_squad_names() -> set[str]:
    """Load all player names from squads_2026.json."""
    squads_path = PROJECT_ROOT / "data" / "squads_2026.json"
    with open(squads_path) as f:
        squads = json.load(f)

    names = set()
    for team_data in squads["teams"].values():
        for p in team_data["players"]:
            names.add(p["name"])
    return names


def load_aliases() -> dict[str, str]:
    """Load player alias mapping (ESPN name -> Cricsheet name)."""
    alias_path = PROJECT_ROOT / "data" / "player_aliases.json"
    with open(alias_path) as f:
        aliases = json.load(f)
    # Build reverse mapping too: cricsheet -> espn
    reverse = {v: k for k, v in aliases.items() if not k.startswith("_")}
    return aliases, reverse


def aggregate_form(records: list[dict], squad_names: set[str], aliases: dict, reverse_aliases: dict) -> pd.DataFrame:
    """
    Aggregate per-match records into per-player form stats.
    Only include players in the IPL 2026 squads (matching via aliases).
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    # Map Cricsheet names to squad names
    def resolve_name(cricsheet_name: str) -> str | None:
        # Direct match
        if cricsheet_name in squad_names:
            return cricsheet_name
        # Check reverse alias (cricsheet -> espn)
        if cricsheet_name in reverse_aliases:
            espn_name = reverse_aliases[cricsheet_name]
            if espn_name in squad_names:
                return espn_name
        return None

    df["squad_name"] = df["player"].apply(resolve_name)
    df = df.dropna(subset=["squad_name"])

    if df.empty:
        return pd.DataFrame()

    # Aggregate per player
    agg = df.groupby("squad_name").agg(
        matches=("match_id", "nunique"),
        total_runs=("runs", "sum"),
        total_balls_faced=("balls_faced", "sum"),
        total_fours=("fours", "sum"),
        total_sixes=("sixes", "sum"),
        dismissals=("is_out", "sum"),
        total_balls_bowled=("balls_bowled", "sum"),
        total_runs_conceded=("runs_conceded", "sum"),
        total_wickets=("wickets", "sum"),
        total_catches=("catches", "sum"),
        leagues=("league", lambda x: ",".join(sorted(set(x)))),
    ).reset_index()

    agg = agg.rename(columns={"squad_name": "player"})

    # Compute derived stats
    agg["batting_avg"] = agg.apply(
        lambda r: r["total_runs"] / max(r["dismissals"], 1), axis=1
    ).round(2)
    agg["strike_rate"] = agg.apply(
        lambda r: (r["total_runs"] / max(r["total_balls_faced"], 1)) * 100, axis=1
    ).round(2)
    agg["bowling_avg"] = agg.apply(
        lambda r: r["total_runs_conceded"] / max(r["total_wickets"], 1) if r["total_wickets"] > 0 else 0, axis=1
    ).round(2)
    agg["economy"] = agg.apply(
        lambda r: (r["total_runs_conceded"] / max(r["total_balls_bowled"], 1)) * 6 if r["total_balls_bowled"] > 0 else 0, axis=1
    ).round(2)
    overs = agg["total_balls_bowled"] / 6
    agg["overs_bowled"] = overs.round(1)

    return agg.sort_values("matches", ascending=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch recent T20 form data")
    parser.add_argument("--months", type=int, default=12, help="Look-back window in months (default: 12)")
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(days=args.months * 30)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    print(f"Fetching T20 data from {cutoff_str} to now ({args.months} months)")

    # Load squad names and aliases
    squad_names = load_squad_names()
    aliases, reverse_aliases = load_aliases()
    print(f"Tracking {len(squad_names)} squad players")

    # Download and parse each league
    all_records = []
    for league, url in T20_SOURCES.items():
        try:
            zip_bytes = download_zip(url, league)
            records = parse_matches_from_zip(zip_bytes, cutoff_str, league)
            all_records.extend(records)
            print(f"    → {len(records)} player-match records from {league}")
        except Exception as e:
            print(f"    → FAILED: {e}")
            continue

    print(f"\nTotal raw records: {len(all_records)}")

    # Aggregate
    form_df = aggregate_form(all_records, squad_names, aliases, reverse_aliases)

    if form_df.empty:
        print("No matching players found!")
        sys.exit(1)

    # Save
    form_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(form_df)} player form records to {OUTPUT_PATH}")
    print(f"\nTop 15 by matches played:")
    print(form_df[["player", "matches", "total_runs", "strike_rate", "total_wickets", "economy", "leagues"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
