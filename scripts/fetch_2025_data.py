#!/usr/bin/env python3
"""
Download IPL 2025 data from Cricsheet and convert to the same CSV format
as the existing Kaggle dataset (matches.csv + deliveries.csv).

Cricsheet provides ball-by-ball data in JSON format. This script:
  1. Downloads the IPL JSON zip from Cricsheet
  2. Parses each match JSON file
  3. Filters to 2025 season matches
  4. Converts to matches.csv + deliveries.csv format
  5. Appends to existing CSVs (or writes new ones)

Usage:
    python scripts/fetch_2025_data.py
    python scripts/fetch_2025_data.py --append   # append to existing CSVs
"""

from __future__ import annotations

import json
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

# Cricsheet IPL download URL (JSON format)
CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
TEMP_DIR = PROJECT_ROOT / "data" / "tmp_cricsheet"


def download_cricsheet_ipl() -> Path:
    """Download IPL JSON zip from Cricsheet."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "ipl_json.zip"

    if zip_path.exists():
        print(f"  Using cached {zip_path}")
        return zip_path

    print(f"  Downloading from {CRICSHEET_IPL_JSON_URL}...")
    urllib.request.urlretrieve(CRICSHEET_IPL_JSON_URL, zip_path)
    print(f"  Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


def parse_match_json(data: dict, match_id: int) -> tuple[dict | None, list[dict]]:
    """
    Parse a single Cricsheet JSON match into:
      - match_row: dict matching matches.csv columns
      - delivery_rows: list of dicts matching deliveries.csv columns
    """
    info = data.get("info", {})
    innings = data.get("innings", [])

    # Extract season
    season = info.get("season", "")
    if isinstance(season, list):
        season = "/".join(str(s) for s in season)

    # Match-level data
    teams = info.get("teams", [])
    if len(teams) < 2:
        return None, []

    outcome = info.get("outcome", {})
    winner = outcome.get("winner", "")
    result = outcome.get("result", "")
    result_margin = ""
    if "by" in outcome:
        by = outcome["by"]
        if "runs" in by:
            result = "runs"
            result_margin = by["runs"]
        elif "wickets" in by:
            result = "wickets"
            result_margin = by["wickets"]

    toss = info.get("toss", {})
    dates = info.get("dates", [])
    date_str = dates[0] if dates else ""

    match_row = {
        "id": match_id,
        "season": str(season),
        "city": info.get("city", ""),
        "date": date_str,
        "match_type": info.get("match_type", "T20"),
        "player_of_match": ", ".join(info.get("player_of_match", [])),
        "venue": info.get("venue", ""),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "winner": winner,
        "result": result,
        "result_margin": result_margin,
        "target_runs": "",
        "target_overs": "",
        "super_over": "Y" if info.get("super_over", False) else "N",
        "method": info.get("method", ""),
        "umpire1": "",
        "umpire2": "",
    }

    # Umpires
    umpires = info.get("umpires", [])
    if len(umpires) >= 1:
        match_row["umpire1"] = umpires[0]
    if len(umpires) >= 2:
        match_row["umpire2"] = umpires[1]

    # Delivery-level data
    delivery_rows = []
    for inning_data in innings:
        inning_num = inning_data.get("team", "")
        # Determine inning number from position
        inning_idx = innings.index(inning_data) + 1

        overs = inning_data.get("overs", [])
        for over_data in overs:
            over_num = over_data.get("over", 0)
            deliveries_list = over_data.get("deliveries", [])

            for ball_idx, delivery in enumerate(deliveries_list):
                batter = delivery.get("batter", "")
                bowler = delivery.get("bowler", "")
                non_striker = delivery.get("non_striker", "")

                runs = delivery.get("runs", {})
                batsman_runs = runs.get("batter", 0)
                extra_runs = runs.get("extras", 0)
                total_runs = runs.get("total", 0)

                extras = delivery.get("extras", {})
                extras_type = ""
                if extras:
                    extras_type = list(extras.keys())[0] if extras else ""

                # Wicket info
                wickets = delivery.get("wickets", [])
                is_wicket = 1 if wickets else 0
                player_dismissed = ""
                dismissal_kind = ""
                fielder = ""

                if wickets:
                    w = wickets[0]
                    player_dismissed = w.get("player_out", "")
                    dismissal_kind = w.get("kind", "")
                    fielders = w.get("fielders", [])
                    if fielders:
                        fielder = fielders[0].get("name", "")

                delivery_rows.append({
                    "match_id": match_id,
                    "inning": inning_idx,
                    "over": over_num,
                    "ball": ball_idx + 1,
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "batsman_runs": batsman_runs,
                    "extra_runs": extra_runs,
                    "total_runs": total_runs,
                    "extras_type": extras_type,
                    "is_wicket": is_wicket,
                    "player_dismissed": player_dismissed,
                    "dismissal_kind": dismissal_kind,
                    "fielder": fielder,
                })

    return match_row, delivery_rows


def convert_cricsheet_to_csv(
    zip_path: Path,
    season_filter: str | None = "2025",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all match JSONs from a Cricsheet zip and convert to DataFrames.

    Args:
        zip_path: Path to downloaded Cricsheet zip.
        season_filter: Only include matches from this season (None = all).

    Returns:
        (matches_df, deliveries_df)
    """
    all_matches = []
    all_deliveries = []

    print(f"  Parsing Cricsheet JSON files from {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        print(f"  Found {len(json_files)} JSON files")

        for i, fname in enumerate(json_files):
            try:
                with zf.open(fname) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            info = data.get("info", {})
            season = info.get("season", "")
            if isinstance(season, list):
                season = "/".join(str(s) for s in season)

            # Filter by season
            if season_filter and str(season) != season_filter:
                continue

            # Use filename as basis for match_id
            match_id_str = fname.replace(".json", "").split("/")[-1]
            try:
                match_id = int(match_id_str)
            except ValueError:
                match_id = 100000 + i

            match_row, delivery_rows = parse_match_json(data, match_id)
            if match_row:
                all_matches.append(match_row)
                all_deliveries.extend(delivery_rows)

    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)

    print(f"  Parsed {len(matches_df)} matches, {len(deliveries_df)} deliveries for season {season_filter or 'ALL'}")

    return matches_df, deliveries_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch IPL 2025 data from Cricsheet")
    parser.add_argument("--append", action="store_true", help="Append to existing CSVs")
    parser.add_argument("--season", default="2025", help="Season to filter (default: 2025)")
    parser.add_argument("--all-seasons", action="store_true", help="Download all seasons (overwrite existing)")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   Cricsheet → IPL CSV Converter")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/3] Downloading Cricsheet IPL data...")
    zip_path = download_cricsheet_ipl()

    # Step 2: Convert
    season = None if args.all_seasons else args.season
    print(f"\n[2/3] Converting to CSV format (season={season or 'ALL'})...")
    new_matches, new_deliveries = convert_cricsheet_to_csv(zip_path, season_filter=season)

    if new_matches.empty:
        print(f"\nWARNING: No matches found for season {season}. Check if data is available.")
        return

    # Step 3: Save
    print(f"\n[3/3] Saving CSVs to {DATA_RAW}...")

    if args.append and not args.all_seasons:
        # Append to existing CSVs
        matches_path = DATA_RAW / "matches.csv"
        deliveries_path = DATA_RAW / "deliveries.csv"

        if matches_path.exists():
            existing_matches = pd.read_csv(matches_path)
            # Avoid ID collisions
            max_id = existing_matches["id"].max() if "id" in existing_matches.columns else 0
            new_matches["id"] = range(int(max_id) + 1, int(max_id) + 1 + len(new_matches))
            new_deliveries["match_id"] = new_deliveries["match_id"].map(
                dict(zip(
                    new_matches.index,
                    new_matches["id"],
                ))
            ).fillna(new_deliveries["match_id"])

            combined_matches = pd.concat([existing_matches, new_matches], ignore_index=True)
            combined_matches.to_csv(matches_path, index=False)
            print(f"  Appended {len(new_matches)} matches → {matches_path} (total: {len(combined_matches)})")
        else:
            new_matches.to_csv(matches_path, index=False)
            print(f"  Wrote {len(new_matches)} matches → {matches_path}")

        if deliveries_path.exists():
            existing_deliveries = pd.read_csv(deliveries_path)
            combined_deliveries = pd.concat([existing_deliveries, new_deliveries], ignore_index=True)
            combined_deliveries.to_csv(deliveries_path, index=False)
            print(f"  Appended {len(new_deliveries)} deliveries → {deliveries_path} (total: {len(combined_deliveries)})")
        else:
            new_deliveries.to_csv(deliveries_path, index=False)
            print(f"  Wrote {len(new_deliveries)} deliveries → {deliveries_path}")
    else:
        # Write new files (or overwrite)
        suffix = f"_{season}" if season else ""
        matches_path = DATA_RAW / f"matches{suffix}.csv"
        deliveries_path = DATA_RAW / f"deliveries{suffix}.csv"

        new_matches.to_csv(matches_path, index=False)
        new_deliveries.to_csv(deliveries_path, index=False)
        print(f"  Wrote {len(new_matches)} matches → {matches_path}")
        print(f"  Wrote {len(new_deliveries)} deliveries → {deliveries_path}")

    print(f"\nDone! Season breakdown:")
    if "season" in new_matches.columns:
        print(new_matches.groupby("season").size().to_string())

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
