"""
IPL Fantasy CLI — command-line interface for team generation and analysis.

Usage:
    ipl-fantasy generate --team1 "MI" --team2 "CSK" --venue "Wankhede"
    ipl-fantasy forecast --player "Virat Kohli"
    ipl-fantasy backtest --season 2024
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(
    name="ipl-fantasy",
    help="IPL Fantasy Agentic System -- AI-powered IPL Fantasy team optimizer",
    add_completion=False,
)

console = Console()

# ── Team alias mapping (abbreviations → full names) ──────────────────────────
TEAM_ALIASES = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad",
    "DC": "Delhi Capitals",
    "GT": "Gujarat Titans",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "LSG": "Lucknow Super Giants",
    # Common alternate names
    "Mumbai": "Mumbai Indians",
    "Chennai": "Chennai Super Kings",
    "Bangalore": "Royal Challengers Bengaluru",
    "Bengaluru": "Royal Challengers Bengaluru",
    "Kolkata": "Kolkata Knight Riders",
    "Hyderabad": "Sunrisers Hyderabad",
    "Delhi": "Delhi Capitals",
    "Gujarat": "Gujarat Titans",
    "Punjab": "Punjab Kings",
    "Rajasthan": "Rajasthan Royals",
    "Lucknow": "Lucknow Super Giants",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQUADS_PATH = PROJECT_ROOT / "data" / "squads_2026.json"
CRICKET_DATA_2026_PATH = PROJECT_ROOT / "data" / "raw" / "cricket_data_2026.csv"


def _resolve_team_name(name: str) -> str:
    """Resolve team abbreviation or alias to full name."""
    return TEAM_ALIASES.get(name, TEAM_ALIASES.get(name.upper(), name))


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_squads(squads_path: Path | None = None) -> dict:
    """Load 2026 squads from JSON."""
    path = squads_path or SQUADS_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_cricket_data_2026(csv_path: Path | None = None) -> pd.DataFrame | None:
    """Load cricket_data_2026.csv with per-year player stats."""
    path = csv_path or CRICKET_DATA_2026_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Clean numeric columns
    for col in df.columns:
        if col not in ("Player_Name", "Year", "Highest_Score", "Best_Bowling_Match"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _build_player_pool_from_squads(
    team1: str,
    team2: str,
    squads: dict,
    cricket_data: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Build the player pool from squads_2026.json for the two teams.

    Only includes players from the official 2026 squads.
    Enriches with credit costs from squads + recent performance data.
    """
    teams = squads.get("teams", {})

    # Resolve team names
    team1_full = _resolve_team_name(team1)
    team2_full = _resolve_team_name(team2)

    pool = []
    for team_name in [team1_full, team2_full]:
        team_data = teams.get(team_name)
        if not team_data:
            continue

        for player in team_data["players"]:
            entry = {
                "name": player["name"],
                "role": player["role"],
                "team": team_name,
                "credit_cost": player.get("credit_cost", 7.0),
                "nationality": player.get("nationality", "Indian"),
            }

            # Enrich with recent performance from cricket_data_2026.csv
            if cricket_data is not None:
                player_data = cricket_data[cricket_data["Player_Name"] == player["name"]]
                if not player_data.empty:
                    # Use most recent year's data
                    recent = player_data.sort_values("Year").iloc[-1]
                    entry["recent_runs"] = float(recent.get("Runs_Scored", 0))
                    entry["recent_wickets"] = float(recent.get("Wickets_Taken", 0))
                    entry["recent_avg"] = float(recent.get("Batting_Average", 0))
                    entry["recent_sr"] = float(recent.get("Batting_Strike_Rate", 0))
                    entry["recent_economy"] = float(recent.get("Economy_Rate", 0))

            pool.append(entry)

    return pool


@app.command()
def generate(
    team1: str = typer.Option(
        ..., "--team1", "-t1", help="First team name or abbreviation (e.g. MI, CSK)"
    ),
    team2: str = typer.Option(..., "--team2", "-t2", help="Second team name or abbreviation"),
    venue: str = typer.Option("", "--venue", "-v", help="Match venue"),
    date: str = typer.Option("", "--date", "-d", help="Match date (YYYY-MM-DD)"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of lineups to generate"),
    n_sims: int = typer.Option(10_000, "--sims", "-n", help="Number of Monte Carlo simulations"),
    strategy: str = typer.Option(
        "safe", "--strategy", "-s", help="Captain strategy: safe/differential/contrarian"
    ),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM sidecar for context"),
    data_dir: str = typer.Option("", "--data-dir", help="Path to data/raw directory"),
    squads_file: str = typer.Option("", "--squads", help="Path to squads JSON file"),
    exclude: list[str] = typer.Option(
        [], "--exclude", "-x", help="Players to exclude (e.g. injured). Repeatable."
    ),
    booster: str = typer.Option(
        "",
        "--booster",
        "-b",
        help="Activate a booster: triple_captain/free_hit/foreign_stars/indian_warrior/double_points/wildcard",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Generate optimized IPL Fantasy lineups for a match (2026 squads only)."""
    _setup_logging(verbose)

    from src.agent.orchestrator import MatchInput, Orchestrator
    from src.captain.selector import CaptainStrategy
    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset

    # Resolve team names
    team1_full = _resolve_team_name(team1)
    team2_full = _resolve_team_name(team2)

    console.print(
        Panel(
            f"[bold cyan]{team1_full}[/] vs [bold cyan]{team2_full}[/] at [yellow]{venue or 'TBD'}[/]",
            title="IPL Fantasy Team Generator (2026)",
            border_style="green",
        )
    )

    # Load 2026 squads
    squads_path = Path(squads_file) if squads_file else None
    squads = _load_squads(squads_path)
    if not squads:
        console.print("[bold red]Could not load squads_2026.json[/]")
        raise typer.Exit(1)

    # Load cricket_data_2026.csv for enrichment
    cricket_data = _load_cricket_data_2026()

    # Build player pool from 2026 squads ONLY
    player_pool = _build_player_pool_from_squads(
        team1_full,
        team2_full,
        squads,
        cricket_data,
    )

    # Exclude injured / unavailable players
    if exclude:
        exclude_lower = [e.lower() for e in exclude]
        before = len(player_pool)
        player_pool = [p for p in player_pool if p["name"].lower() not in exclude_lower]
        removed = before - len(player_pool)
        if removed:
            console.print(f"[dim]Excluded {removed} player(s): {', '.join(exclude)}[/]")
        else:
            console.print(f"[yellow]Warning: none of the excluded names matched: {exclude}[/]")

    if not player_pool:
        console.print(f"[bold red]No players found for {team1_full} or {team2_full} in squads.[/]")
        available = list(squads.get("teams", {}).keys())
        console.print(f"[dim]Available teams: {', '.join(available)}[/]")
        raise typer.Exit(1)

    console.print(
        f"[dim]Player pool: {len(player_pool)} players "
        f"({sum(1 for p in player_pool if p['team'] == team1_full)} {team1_full}, "
        f"{sum(1 for p in player_pool if p['team'] == team2_full)} {team2_full})[/]"
    )

    # Load historical data for forecaster training
    with console.status("[bold green]Loading IPL dataset..."):
        data_path = Path(data_dir) if data_dir else None
        matches, deliveries = load_dataset(data_path)

    # Build scorecards
    with console.status("[bold green]Building player scorecards..."):
        scorecards = build_player_scorecards(deliveries, matches)

    # Initialize orchestrator
    with console.status("[bold green]Fitting forecaster..."):
        orchestrator = Orchestrator(use_llm=use_llm)
        orchestrator.fit_from_data(scorecards)

    # Map strategy
    strategy_map = {
        "safe": CaptainStrategy.SAFE,
        "differential": CaptainStrategy.DIFFERENTIAL,
        "contrarian": CaptainStrategy.CONTRARIAN,
    }
    cap_strategy = strategy_map.get(strategy, CaptainStrategy.SAFE)

    # Generate — using 2026 squad players only
    match_input = MatchInput(
        team1=team1_full,
        team2=team2_full,
        venue=venue,
        date=date or None,
        players=player_pool,
    )

    # Validate booster
    active_booster = booster if booster else None
    if active_booster:
        from src.config import BOOSTERS

        if active_booster not in BOOSTERS:
            console.print(f"[bold red]Unknown booster '{active_booster}'.[/]")
            console.print(f"[dim]Valid boosters: {', '.join(BOOSTERS.keys())}[/]")
            raise typer.Exit(1)
        console.print(
            f"[bold magenta]Booster active: {active_booster} — {BOOSTERS[active_booster]['description']}[/]"
        )

    with console.status(f"[bold green]Running {n_sims:,} simulations..."):
        result = orchestrator.generate_team(
            match=match_input,
            top_k=top_k,
            captain_strategy=cap_strategy,
            n_simulations=n_sims,
            booster=active_booster,
        )

    # Display results
    _display_results(result, top_k)


@app.command()
def forecast(
    player: str = typer.Option(..., "--player", "-p", help="Player name"),
    venue: str = typer.Option("", "--venue", "-v", help="Venue for conditioning"),
    data_dir: str = typer.Option("", "--data-dir", help="Path to data/raw directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Show forecast for a specific player."""
    _setup_logging(verbose)

    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset
    from src.forecast.models import PlayerForecaster

    with console.status("[bold green]Loading data..."):
        data_path = Path(data_dir) if data_dir else None
        matches, deliveries = load_dataset(data_path)
        scorecards = build_player_scorecards(deliveries, matches)

    forecaster = PlayerForecaster()
    forecaster.fit(scorecards)

    fc = forecaster.forecast(player, venue=venue or None)

    table = Table(title=f"Forecast: {fc.player} ({fc.role})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Expected Points", f"{fc.expected_points:.1f}")
    table.add_row("Std Dev", f"{fc.std_points:.1f}")
    table.add_row("Floor (5th)", f"{fc.floor_5:.1f}")
    table.add_row("10th Percentile", f"{fc.quantile_10:.1f}")
    table.add_row("Median (50th)", f"{fc.quantile_50:.1f}")
    table.add_row("90th Percentile", f"{fc.quantile_90:.1f}")
    table.add_row("Ceiling (95th)", f"{fc.ceiling_95:.1f}")

    console.print(table)


@app.command()
def backtest(
    season: int = typer.Option(2024, "--season", help="IPL season to backtest"),
    strategy: str = typer.Option("safe", "--strategy", "-s", help="Captain strategy"),
    data_dir: str = typer.Option("", "--data-dir", help="Path to data/raw directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Backtest the system on historical data."""
    _setup_logging(verbose)

    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset

    with console.status("[bold green]Loading data..."):
        data_path = Path(data_dir) if data_dir else None
        matches, deliveries = load_dataset(data_path)
        scorecards = build_player_scorecards(deliveries, matches)

    # Filter to season
    if "season" in scorecards.columns:
        season_data = scorecards[scorecards["season"] == season]
        console.print(f"[bold]Season {season}: {season_data['match_id'].nunique()} matches[/]")
    else:
        console.print("[yellow]Season column not found; using all data.[/]")
        season_data = scorecards

    if season_data.empty:
        console.print(f"[red]No data found for season {season}.[/]")
        raise typer.Exit(1)

    # Summary stats
    table = Table(title=f"Season {season} Summary")
    table.add_column("Stat", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Matches", str(season_data["match_id"].nunique()))
    table.add_row("Unique Players", str(season_data["player"].nunique()))
    table.add_row("Avg Fantasy Points", f"{season_data['fantasy_points'].mean():.1f}")
    table.add_row("Max Fantasy Points", f"{season_data['fantasy_points'].max():.1f}")
    table.add_row("Median Fantasy Points", f"{season_data['fantasy_points'].median():.1f}")

    # Top performers
    top = (
        season_data.groupby("player")["fantasy_points"].agg(["mean", "count", "max"]).reset_index()
    )
    top = top[top["count"] >= 5].sort_values("mean", ascending=False).head(10)

    table.add_section()
    table.add_row("[bold]Top 10 Players (min 5 matches)[/]", "")
    for _, row in top.iterrows():
        table.add_row(
            f"  {row['player']}",
            f"avg={row['mean']:.1f}, max={row['max']:.0f}, matches={row['count']:.0f}",
        )

    console.print(table)


@app.command()
def teams() -> None:
    """List all available 2026 IPL teams and their squads."""
    squads = _load_squads()
    if not squads:
        console.print("[bold red]Could not load squads_2026.json[/]")
        raise typer.Exit(1)

    for team_name, info in squads.get("teams", {}).items():
        n = len(info["players"])
        overseas = sum(1 for p in info["players"] if p.get("nationality") == "Overseas")
        console.print(
            f"[bold cyan]{info['abbreviation']:5s}[/] {team_name:30s} "
            f"[dim]{n} players ({overseas} overseas)[/] — C: {info['captain']}"
        )


# ── Squad Management Commands ─────────────────────────────────────────────────

squad_app = typer.Typer(help="Manage your season-long fantasy squad")


@squad_app.command("init")
def squad_init(
    match: int = typer.Option(1, "--match", "-m", help="Starting match number"),
    players: str | None = typer.Option(
        None,
        "--players",
        "-p",
        help="Comma-separated player names (optional - auto-generates if not provided)",
    ),
    team1: str | None = typer.Option(
        None, "--team1", "-t1", help="First team (auto-detected from schedule if not provided)"
    ),
    team2: str | None = typer.Option(
        None, "--team2", "-t2", help="Second team (auto-detected from schedule if not provided)"
    ),
    venue: str = typer.Option(
        "", "--venue", "-v", help="Match venue (auto-detected from schedule if not provided)"
    ),
    season: str = typer.Option("2026", "--season", "-s", help="Season year"),
    strategy: str = typer.Option(
        "safe",
        "--strategy",
        help="Captain strategy for auto-generation: safe/differential/contrarian",
    ),
    look_ahead: int = typer.Option(
        0,
        "--look-ahead",
        "-l",
        help="Number of future matches to consider for initial squad optimization (0=single match only)",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """
    Initialize your squad for the season.

    If --players is not provided, will auto-generate an optimal XI using the
    forecast and optimizer for the specified match.

    Use --look-ahead N to optimize for the next N matches (values players whose
    teams have busy upcoming schedules).
    """
    from src.season.state import SeasonStateManager
    from src.season.schedule import Schedule
    from src.agent.orchestrator import MatchInput, Orchestrator
    from src.captain.selector import CaptainStrategy
    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset
    from src.optimizer.fantasy_ilp import PlayerSlot

    _setup_logging(verbose)

    manager = SeasonStateManager()

    # If players provided manually, use those
    if players:
        player_list = [p.strip() for p in players.split(",")]
        try:
            state = manager.init_squad(player_list, match, season)
            console.print(f"[bold green]Squad initialized for Match {match}[/]")
            console.print(f"Players: {', '.join(state.squad)}")
            console.print(f"Transfers remaining: {manager.remaining_transfers(state)}")
        except ValueError as e:
            console.print(f"[bold red]Error: {e}[/]")
            raise typer.Exit(1)
        return

    # Auto-generate squad from match
    # Load schedule to get match info
    try:
        schedule = Schedule.load()
        match_info = schedule.get_match(match)
    except Exception:
        match_info = None

    # Resolve teams
    if team1 and team2:
        team1_full = _resolve_team_name(team1)
        team2_full = _resolve_team_name(team2)
    elif match_info:
        team1_full = match_info.home
        team2_full = match_info.away
        if not venue:
            venue = match_info.venue
    else:
        console.print(
            "[bold red]Error: Must provide either --players, --team1/--team2, or a valid match number in schedule[/]"
        )
        console.print("[dim]Examples:[/]")
        console.print("  ipl-fantasy squad init --players 'Rohit Sharma,Virat Kohli,...'")
        console.print("  ipl-fantasy squad init --match 1")
        console.print("  ipl-fantasy squad init --team1 MI --team2 CSK --venue 'Wankhede'")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold cyan]{team1_full}[/] vs [bold cyan]{team2_full}[/]\n"
            f"Auto-generating optimal XI for Match {match}...\n"
            + (
                f"[dim]Look-ahead: {look_ahead} matches (all 245 players considered)[/]"
                if look_ahead > 0
                else "[dim]Single match optimization (49 players from match teams)[/]"
            ),
            title="Initialize Squad",
            border_style="green",
        )
    )

    # Load data and generate squad
    with console.status("[bold green]Loading data..."):
        matches_df, deliveries = load_dataset()
        scorecards = build_player_scorecards(deliveries, matches_df)

    with console.status("[bold green]Generating optimal squad..."):
        squads = _load_squads()
        cricket_data = _load_cricket_data_2026()

        # For look-ahead mode, use ALL 245 players from all teams
        # For single match mode, only use the two teams playing
        if look_ahead > 0:
            # Build player pool from ALL teams for maximum flexibility
            all_teams = list(squads.get("teams", {}).keys())
            player_pool = []
            for team_name in all_teams:
                team_data = squads["teams"].get(team_name)
                if not team_data:
                    continue
                for player in team_data["players"]:
                    entry = {
                        "name": player["name"],
                        "role": player["role"],
                        "team": team_name,
                        "credit_cost": player.get("credit_cost", 7.0),
                        "nationality": player.get("nationality", "Indian"),
                    }
                    if cricket_data is not None:
                        player_data = cricket_data[cricket_data["Player_Name"] == player["name"]]
                        if not player_data.empty:
                            recent = player_data.sort_values("Year").iloc[-1]
                            entry["recent_runs"] = float(recent.get("Runs_Scored", 0))
                            entry["recent_wickets"] = float(recent.get("Wickets_Taken", 0))
                    player_pool.append(entry)
        else:
            # Single match mode: only players from the two teams playing
            player_pool = _build_player_pool_from_squads(
                team1_full,
                team2_full,
                squads,
                cricket_data,
            )

        orchestrator = Orchestrator(use_llm=False)
        orchestrator.fit_from_data(scorecards)

        match_input = MatchInput(
            team1=team1_full,
            team2=team2_full,
            venue=venue,
            players=player_pool,
        )

        strategy_map = {
            "safe": CaptainStrategy.SAFE,
            "differential": CaptainStrategy.DIFFERENTIAL,
            "contrarian": CaptainStrategy.CONTRARIAN,
        }

        # Use transfer optimizer with look-ahead if requested
        if look_ahead > 0:
            from src.optimizer.transfer_optimizer import TransferOptimizer

            schedule = Schedule.load()

            # Step 1: Run single-match optimization to get best C/VC for TODAY
            # This ensures we have strong captain options from teams playing match N
            single_match_pool = _build_player_pool_from_squads(
                team1_full, team2_full, squads, cricket_data
            )
            single_match_input = MatchInput(
                team1=team1_full,
                team2=team2_full,
                venue=venue,
                players=single_match_pool,
            )

            console.print("[dim]Optimizing for today's match (captain selection)...[/]")
            today_result = orchestrator.generate_team(
                match=single_match_input,
                top_k=1,
                captain_strategy=strategy_map.get(strategy, CaptainStrategy.SAFE),
            )
            today_lineup = today_result.best_lineup
            today_best_captain = today_lineup.captain
            today_best_vc = today_lineup.vice_captain
            today_best_players = {p.name for p in today_lineup.players}

            # Step 2: Run look-ahead optimization for long-term squad value
            console.print(f"[dim]Optimizing for next {look_ahead} matches (fixture density)...[/]")

            # Build player slots from simulation (all 245 players)
            player_names = [p["name"] for p in player_pool]
            roles = {p["name"]: p.get("role", "BAT") for p in player_pool}
            forecasts = orchestrator.forecaster.forecast_match(
                players=player_names,
                roles=roles,
                venue=venue,
            )
            sim_result = orchestrator.simulator.simulate_match(forecasts)

            # Build PlayerSlots for optimizer
            player_slots = []
            for p_info in player_pool:
                name = p_info["name"]
                sim_row = sim_result.summary[sim_result.summary["player"] == name]
                expected = float(sim_row["mean_fp"].iloc[0]) if not sim_row.empty else 20.0

                player_slots.append(
                    PlayerSlot(
                        name=name,
                        role=p_info.get("role", "BAT"),
                        team=p_info.get("team", team1_full),
                        credit_cost=p_info.get("credit_cost", 8.0),
                        expected_points=expected,
                        nationality=p_info.get("nationality", "Indian"),
                    )
                )

            # Use transfer optimizer with empty current squad (all 11 are "transfers")
            optimizer = TransferOptimizer(schedule=schedule)
            transfer_plan = optimizer.optimize(
                current_squad=[],  # Empty - we're building from scratch
                available_players=player_slots,
                max_transfers=11,  # Can pick any 11 players
                current_match=match,
                look_ahead=look_ahead,
            )

            future_squad_players = transfer_plan.new_squad
            future_expected_points = transfer_plan.expected_points

            # Step 3: Merge squads - ensure we have good C/VC options from today's match
            # Count how many players from today's match teams are in the future squad
            match_info = schedule.get_match(match)
            playing_teams = {match_info.home, match_info.away} if match_info else set()

            future_squad_match_players = [
                p for p in future_squad_players if p.team in playing_teams
            ]

            # If we have less than 3 players from today's match, add top performers
            final_squad = list(future_squad_players)

            if len(future_squad_match_players) < 3:
                # Need to add more players from today's match for good C/VC options
                players_to_add = 3 - len(future_squad_match_players)

                # Get top players from today's best lineup that aren't already in squad
                today_player_objects = {p.name: p for p in today_lineup.players}
                future_squad_names = {p.name for p in final_squad}

                # Prioritize: best C and VC from today, then other high performers
                priority_additions = []
                if today_best_captain and today_best_captain not in future_squad_names:
                    priority_additions.append(today_player_objects[today_best_captain])
                if today_best_vc and today_best_vc not in future_squad_names:
                    priority_additions.append(today_player_objects[today_best_vc])

                # Add other top performers from today if needed
                for p in today_lineup.players:
                    if len(priority_additions) >= players_to_add:
                        break
                    if p.name not in future_squad_names and p not in priority_additions:
                        priority_additions.append(p)

                # Replace lowest-value future players with today's priority players
                # Sort future squad by expected points, remove lowest
                future_squad_sorted = sorted(final_squad, key=lambda p: p.expected_points)
                for new_player in priority_additions:
                    if len(final_squad) >= 11:
                        # Remove lowest value player not from today's match
                        for i, p in enumerate(future_squad_sorted):
                            if p.team not in playing_teams:
                                final_squad.remove(p)
                                future_squad_sorted.pop(i)
                                break
                    final_squad.append(new_player)

            # Create lineup object
            from src.optimizer.fantasy_ilp import OptimizedLineup

            best_lineup = OptimizedLineup(
                players=final_squad,
                total_expected_points=sum(p.expected_points for p in final_squad),
                total_credits=sum(p.credit_cost for p in final_squad),
            )

            # Step 4: Select C/VC from players in final squad who are playing TODAY
            from src.captain.selector import CaptainSelector

            eligible_for_captain = [p for p in final_squad if p.team in playing_teams]
            selector = CaptainSelector()

            if len(eligible_for_captain) >= 2:
                eligible_names = [p.name for p in eligible_for_captain]
                pick = selector.select(
                    lineup_players=eligible_names,
                    sim_result=sim_result,
                    strategy=strategy_map.get(strategy, CaptainStrategy.SAFE),
                )
                best_lineup.captain = pick.captain
                best_lineup.vice_captain = pick.vice_captain
            elif len(eligible_for_captain) == 1:
                best_lineup.captain = eligible_for_captain[0].name
                # Pick VC from rest of eligible players
                other_eligible = [p for p in eligible_for_captain if p.name != best_lineup.captain]
                if other_eligible:
                    best_lineup.vice_captain = other_eligible[0].name
            else:
                # Fallback to today's best C/VC if they're somehow not in squad
                best_lineup.captain = today_best_captain
                best_lineup.vice_captain = today_best_vc
                console.print(f"[yellow]Warning: Using fallback captain selection[/]")

        else:
            # Single match optimization (original behavior)
            result = orchestrator.generate_team(
                match=match_input,
                top_k=1,
                captain_strategy=strategy_map.get(strategy, CaptainStrategy.SAFE),
            )
            best_lineup = result.best_lineup

    # Extract players from best lineup
    player_names = [p.name for p in best_lineup.players]

    # Initialize squad
    try:
        state = manager.init_squad(player_names, match, season)

        console.print(f"[bold green]Squad initialized for Match {match}![/]")
        console.print()

        # Show the generated squad
        table = Table(title="Auto-Generated Squad")
        table.add_column("Role", style="cyan", width=6)
        table.add_column("Player", style="white")
        table.add_column("Team", style="dim")
        table.add_column("Credits", justify="right")

        for p in sorted(
            best_lineup.players, key=lambda x: ("WK", "BAT", "AR", "BOWL").index(x.role)
        ):
            table.add_row(p.role, p.name, p.team, f"{p.credit_cost:.1f}")

        console.print(table)
        console.print()
        console.print(f"[bold]Captain:[/] {best_lineup.captain}")
        console.print(f"[bold]Vice-Captain:[/] {best_lineup.vice_captain}")
        console.print(f"[bold]Expected Points:[/] {best_lineup.total_expected_points:.1f}")
        console.print(f"[bold]Transfers remaining:[/] {manager.remaining_transfers(state)}")

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/]")
        raise typer.Exit(1)


@squad_app.command("show")
def squad_show() -> None:
    """Display your current squad."""
    from src.season.state import SeasonStateManager

    manager = SeasonStateManager()
    state = manager.load()

    if not state.squad:
        console.print("[yellow]No squad initialized. Run: ipl-fantasy squad init[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Current Squad (Match {state.current_match})[/]")
    for i, player in enumerate(state.squad, 1):
        console.print(f"  {i:2d}. {player}")


@squad_app.command("set")
def squad_set(
    players: str = typer.Option(..., "--players", "-p", help="Comma-separated player names"),
    match: int = typer.Option(None, "--match", "-m", help="Match number (defaults to current)"),
) -> None:
    """Manually set your squad (resets transfer tracking)."""
    from src.season.state import SeasonStateManager

    player_list = [p.strip() for p in players.split(",")]

    manager = SeasonStateManager()
    state = manager.load()

    try:
        state = manager.set_squad(state, player_list, match)
        console.print(f"[bold green]Squad updated[/]")
        console.print(f"Players: {', '.join(state.squad)}")
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/]")
        raise typer.Exit(1)


@squad_app.command("status")
def squad_status() -> None:
    """Show squad status: transfers remaining, boosters available."""
    from src.season.state import SeasonStateManager
    from src.config import BOOSTERS, TRANSFER

    manager = SeasonStateManager()
    state = manager.load()

    if not state.squad:
        console.print("[yellow]No squad initialized. Run: ipl-fantasy squad init[/]")
        raise typer.Exit(1)

    remaining = manager.remaining_transfers(state)
    stage = "Playoffs" if state.current_match > TRANSFER.league_stage_end else "League Stage"

    table = Table(title=f"Squad Status — Match {state.current_match} ({stage})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Transfers Used", str(state.transfers_used))
    table.add_row("Transfers Remaining", str(remaining))
    table.add_row("Squad Size", str(len(state.squad)))
    table.add_row("Total History", str(len(state.history)))

    # Boosters
    table.add_section()
    table.add_row("[bold]Boosters Used[/]", "")
    for name, info in BOOSTERS.items():
        used = state.boosters_used.get(name, 0)
        max_uses = info["max_uses"]
        status = "✓" if used < max_uses else "✗"
        table.add_row(f"  {name}", f"{used}/{max_uses} {status}")

    console.print(table)


@squad_app.command("history")
def squad_history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records to show"),
) -> None:
    """Show transfer history."""
    from src.season.state import SeasonStateManager

    manager = SeasonStateManager()
    state = manager.load()

    if not state.history:
        console.print("[dim]No transfers recorded yet.[/]")
        return

    table = Table(title="Transfer History")
    table.add_column("Match", style="cyan", justify="right")
    table.add_column("Out", style="red")
    table.add_column("In", style="green")
    table.add_column("Reason", style="dim")

    for record in state.history[-limit:]:
        table.add_row(
            str(record.match_number),
            record.player_out,
            record.player_in,
            record.reason,
        )

    console.print(table)
    if len(state.history) > limit:
        console.print(f"[dim]... and {len(state.history) - limit} more records[/]")


app.add_typer(squad_app, name="squad")

# ── Transfer Planning Commands ───────────────────────────────────────────────


@app.command()
def plan(
    match: int = typer.Option(..., "--match", "-m", help="Match number to plan for"),
    max_transfers: int = typer.Option(
        3, "--max-transfers", "-t", help="Maximum transfers for this match"
    ),
    look_ahead: int = typer.Option(
        0, "--look-ahead", "-l", help="Number of future matches to consider (0=disabled)"
    ),
    strategy: str = typer.Option(
        "safe", "--strategy", "-s", help="Captain strategy: safe/differential/contrarian"
    ),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM sidecar for context"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Plan optimal transfers for a match."""
    _setup_logging(verbose)

    from src.agent.orchestrator import MatchInput, Orchestrator
    from src.captain.selector import CaptainStrategy
    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset
    from src.season.schedule import Schedule
    from src.season.state import SeasonStateManager

    # Load current squad
    manager = SeasonStateManager()
    state = manager.load()

    if not state.squad:
        console.print("[bold red]No squad initialized. Run: ipl-fantasy squad init[/]")
        raise typer.Exit(1)

    # Load schedule and get match info
    schedule = Schedule.load()
    match_info = schedule.get_match(match)

    if not match_info:
        console.print(f"[bold red]Match {match} not found in schedule[/]")
        raise typer.Exit(1)

    # Check transfer budget
    remaining = manager.remaining_transfers(state)
    actual_max = min(max_transfers, remaining)

    console.print(
        Panel(
            f"[bold cyan]{match_info.home}[/] vs [bold cyan]{match_info.away}[/] at [yellow]{match_info.venue}[/]\n"
            f"Match {match} | {match_info.date} {match_info.time}\n"
            f"Transfers remaining: {remaining} | Planning up to {actual_max} transfers",
            title="Transfer Planning",
            border_style="green",
        )
    )

    if remaining <= 0:
        console.print("[bold red]No transfers remaining![/]")
        raise typer.Exit(1)

    # Load squads for player pool (all 245 players, not just match teams)
    squads = _load_squads()
    cricket_data = _load_cricket_data_2026()

    # Build player pool from ALL teams (squad may have players from any team)
    all_teams = list(squads.get("teams", {}).keys())
    player_pool = []
    for team_name in all_teams:
        team_data = squads["teams"].get(team_name)
        if not team_data:
            continue
        for player in team_data["players"]:
            entry = {
                "name": player["name"],
                "role": player["role"],
                "team": team_name,
                "credit_cost": player.get("credit_cost", 7.0),
                "nationality": player.get("nationality", "Indian"),
            }
            if cricket_data is not None:
                player_data = cricket_data[cricket_data["Player_Name"] == player["name"]]
                if not player_data.empty:
                    recent = player_data.sort_values("Year").iloc[-1]
                    entry["recent_runs"] = float(recent.get("Runs_Scored", 0))
                    entry["recent_wickets"] = float(recent.get("Wickets_Taken", 0))
            player_pool.append(entry)

    # Load historical data and run pipeline
    with console.status("[bold green]Loading data..."):
        matches_df, deliveries = load_dataset()
        scorecards = build_player_scorecards(deliveries, matches_df)

    with console.status("[bold green]Planning transfers..."):
        orchestrator = Orchestrator(use_llm=use_llm)
        orchestrator.fit_from_data(scorecards)

        match_input = MatchInput(
            team1=match_info.home,
            team2=match_info.away,
            venue=match_info.venue,
            date=match_info.date,
            players=player_pool,
        )

        strategy_map = {
            "safe": CaptainStrategy.SAFE,
            "differential": CaptainStrategy.DIFFERENTIAL,
            "contrarian": CaptainStrategy.CONTRARIAN,
        }

        transfer_plan = orchestrator.plan_transfers(
            match=match_input,
            current_squad=state.squad,
            max_transfers=actual_max,
            current_match=match,
            look_ahead=look_ahead,
            schedule=schedule,
        )

    # Display results
    console.print()
    console.print(
        Panel(
            f"[bold green]Expected Points:[/] {transfer_plan.expected_points:.1f}\n"
            f"[bold]Transfers:[/] {transfer_plan.num_transfers} | "
            f"[bold]Keeping:[/] {len(transfer_plan.kept_players)}/11 players",
            title="Recommended Transfers",
            border_style="gold1",
        )
    )

    if transfer_plan.transfers_out:
        console.print("[bold red]OUT:[/]")
        for player in transfer_plan.transfers_out:
            console.print(f"  - {player}")

    if transfer_plan.transfers_in:
        console.print("[bold green]IN:[/]")
        for player in transfer_plan.transfers_in:
            # Find player details
            for p in player_pool:
                if p["name"] == player:
                    console.print(f"  + {player} ({p['role']}, {p['team']}, {p['credit_cost']}cr)")
                    break

    # Show new squad
    console.print("\n[bold]New Squad:[/]")
    for i, player in enumerate(transfer_plan.new_squad, 1):
        tag = ""
        if player.name in transfer_plan.transfers_in:
            tag = " [green](NEW)"
        console.print(f"  {i:2d}. {player.name}{tag}")

    # Save plan for apply command
    plan_data = {
        "match_number": match,
        "transfers_in": transfer_plan.transfers_in,
        "transfers_out": transfer_plan.transfers_out,
        "expected_points": transfer_plan.expected_points,
        "timestamp": datetime.now().isoformat(),
    }
    plan_path = PROJECT_ROOT / "data" / "pending_transfers.json"
    with open(plan_path, "w") as f:
        json.dump(plan_data, f, indent=2)

    console.print(f"\n[dim]Run 'ipl-fantasy apply --match {match}' to commit these transfers[/]")


@app.command()
def apply(
    match: int = typer.Option(..., "--match", "-m", help="Match number to apply transfers for"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Skip confirmation prompt"),
) -> None:
    """Apply planned transfers to your squad."""
    from src.season.state import SeasonStateManager
    from src.season.schedule import Schedule

    # Load current state
    manager = SeasonStateManager()
    state = manager.load()

    if not state.squad:
        console.print("[bold red]No squad initialized. Run: ipl-fantasy squad init[/]")
        raise typer.Exit(1)

    # Check for pending transfers (stored in a plan file)
    plan_path = manager.state_path.parent / "pending_transfers.json"

    if not plan_path.exists():
        console.print("[yellow]No pending transfer plan found. Run 'ipl-fantasy plan' first.[/]")
        console.print("[dim]Alternatively, use --players to specify transfers manually[/]")
        raise typer.Exit(1)

    with open(plan_path) as f:
        plan_data = json.load(f)

    if plan_data.get("match_number") != match:
        console.print(
            f"[bold red]Pending plan is for match {plan_data.get('match_number')}, not {match}[/]"
        )
        raise typer.Exit(1)

    transfers_in = plan_data.get("transfers_in", [])
    transfers_out = plan_data.get("transfers_out", [])

    # Show plan
    console.print(
        Panel(
            f"Match {match}: {len(transfers_in)} transfers",
            title="Apply Transfers",
            border_style="yellow",
        )
    )

    if transfers_out:
        console.print("[bold red]OUT:[/]")
        for player in transfers_out:
            console.print(f"  - {player}")

    if transfers_in:
        console.print("[bold green]IN:[/]")
        for player in transfers_in:
            console.print(f"  + {player}")

    # Confirm
    if not confirm:
        response = console.input("\nApply these transfers? [y/N]: ")
        if response.lower() != "y":
            console.print("[dim]Cancelled[/]")
            raise typer.Exit(0)

    # Apply
    try:
        new_state = manager.apply_transfers(
            state,
            transfers_in=transfers_in,
            transfers_out=transfers_out,
            match_number=match,
            reason="planned",
        )

        # Clean up plan file
        plan_path.unlink()

        console.print(f"[bold green]Transfers applied for Match {match}![/]")
        console.print(f"Transfers used: {new_state.transfers_used}/160")
        console.print(f"Transfers remaining: {manager.remaining_transfers(new_state)}")

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/]")
        raise typer.Exit(1)


@app.command()
def toss(
    team1_xi: str = typer.Option(
        ...,
        "--team1-xi",
        "-t1",
        help="Comma-separated playing XI for team 1 (e.g., 'Rohit Sharma,Sharma,Ishan Kishan,...')",
    ),
    team2_xi: str = typer.Option(
        ..., "--team2-xi", "-t2", help="Comma-separated playing XI for team 2"
    ),
    match: int = typer.Option(..., "--match", "-m", help="Match number"),
    team1: str = typer.Option(..., "--team1", help="Team 1 name or abbreviation (e.g., MI)"),
    team2: str = typer.Option(..., "--team2", help="Team 2 name or abbreviation (e.g., CSK)"),
    venue: str = typer.Option("", "--venue", "-v", help="Match venue"),
    strategy: str = typer.Option(
        "safe", "--strategy", "-s", help="Captain strategy: safe/differential/contrarian"
    ),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM sidecar for context"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """
    Optimize lineup AFTER toss with confirmed playing XIs.
    
    This gives much better predictions since we only select from players
    who are actually playing (not the full 25-player squad).
    
    Example:
        ipl-fantasy toss --match 5 --team1 MI --team2 CSK \\
            --team1-xi "Rohit Sharma,Suryakumar Yadav,Ishan Kishan,..." \\
            --team2-xi "Ruturaj Gaikwad,Shivam Dube,Ravindra Jadeja,..."
    """
    _setup_logging(verbose)

    from src.agent.orchestrator import MatchInput, Orchestrator
    from src.captain.selector import CaptainStrategy
    from src.data.features import build_player_scorecards
    from src.data.ingest import load_dataset
    from src.season.schedule import Schedule
    from src.season.state import SeasonStateManager

    team1_full = _resolve_team_name(team1)
    team2_full = _resolve_team_name(team2)

    # Parse playing XIs
    team1_players = [p.strip() for p in team1_xi.split(",")]
    team2_players = [p.strip() for p in team2_xi.split(",")]

    if len(team1_players) != 11 or len(team2_players) != 11:
        console.print(f"[bold red]Error: Each team must have exactly 11 players[/]")
        console.print(f"Team 1: {len(team1_players)} players provided")
        console.print(f"Team 2: {len(team2_players)} players provided")
        raise typer.Exit(1)

    # Get venue from schedule if not provided
    if not venue:
        try:
            schedule = Schedule.load()
            match_info = schedule.get_match(match)
            if match_info:
                venue = match_info.venue
        except Exception:
            pass

    console.print(
        Panel(
            f"[bold cyan]{team1_full}[/] vs [bold cyan]{team2_full}[/] at [yellow]{venue or 'TBD'}[/]\n"
            f"Match {match} | Post-toss optimization\n"
            f"Player pool: {len(team1_players)} + {len(team2_players)} = 22 players (confirmed playing)",
            title="Post-Toss Optimization",
            border_style="green",
        )
    )

    # Build player pool from confirmed playing XI only
    squads = _load_squads()
    cricket_data = _load_cricket_data_2026()

    player_pool = []
    playing_names = set(team1_players + team2_players)

    # Look up player details from squads
    for team_name, team_data in squads.get("teams", {}).items():
        if team_name not in (team1_full, team2_full):
            continue

        for player in team_data["players"]:
            if player["name"] in playing_names:
                entry = {
                    "name": player["name"],
                    "role": player["role"],
                    "team": team_name,
                    "credit_cost": player.get("credit_cost", 7.0),
                    "nationality": player.get("nationality", "Indian"),
                }

                # Enrich with recent stats
                if cricket_data is not None:
                    player_data = cricket_data[cricket_data["Player_Name"] == player["name"]]
                    if not player_data.empty:
                        recent = player_data.sort_values("Year").iloc[-1]
                        entry["recent_runs"] = float(recent.get("Runs_Scored", 0))
                        entry["recent_wickets"] = float(recent.get("Wickets_Taken", 0))

                player_pool.append(entry)

    console.print(f"[dim]Found {len(player_pool)}/22 players in squad database[/]")

    # Warn about missing players
    found_names = {p["name"] for p in player_pool}
    missing = playing_names - found_names
    if missing:
        console.print(
            f"[yellow]Warning: {len(missing)} players not found in 2026 squads:[/] {', '.join(missing)}"
        )

    # Run optimization
    with console.status("[bold green]Loading historical data..."):
        matches_df, deliveries = load_dataset()
        scorecards = build_player_scorecards(deliveries, matches_df)

    with console.status("[bold green]Optimizing with confirmed playing XI..."):
        orchestrator = Orchestrator(use_llm=use_llm)
        orchestrator.fit_from_data(scorecards)

        match_input = MatchInput(
            team1=team1_full,
            team2=team2_full,
            venue=venue,
            players=player_pool,
        )

        strategy_map = {
            "safe": CaptainStrategy.SAFE,
            "differential": CaptainStrategy.DIFFERENTIAL,
            "contrarian": CaptainStrategy.CONTRARIAN,
        }

        result = orchestrator.generate_team(
            match=match_input,
            top_k=3,
            captain_strategy=strategy_map.get(strategy, CaptainStrategy.SAFE),
        )

    # Display results
    _display_results(result, top_k=3)

    # Compare with current squad if available
    manager = SeasonStateManager()
    state = manager.load()

    if state.squad:
        current_set = set(state.squad)
        recommended_set = {p.name for p in result.best_lineup.players}

        common = current_set & recommended_set
        need_to_add = recommended_set - current_set
        need_to_remove = current_set - recommended_set

        console.print()
        console.print(
            Panel(
                f"[bold green]Already in squad:[/] {len(common)}/11 players\n"
                f"[bold yellow]Need to add:[/] {len(need_to_add)} players\n"
                f"[bold red]Should remove:[/] {len(need_to_remove)} players",
                title="Squad Comparison",
                border_style="blue",
            )
        )

        if need_to_add:
            console.print("[bold yellow]Players to add:[/]")
            for p in need_to_add:
                console.print(f"  + {p}")

        if need_to_remove:
            console.print("[bold red]Players to remove:[/]")
            for p in need_to_remove:
                console.print(f"  - {p}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _display_results(result, top_k: int) -> None:
    """Display pipeline results in a rich table."""
    from src.agent.orchestrator import PipelineResult

    console.print()

    # Best lineup
    best = result.best_lineup
    pick = result.best_captain_pick

    cap_mult = "3x" if best.booster == "triple_captain" else "2x"
    budget_line = (
        f"[bold]Credits Used:[/] {best.total_credits:.1f} (no limit — Free Hit)"
        if best.booster == "free_hit"
        else f"[bold]Credits Used:[/] {best.total_credits:.1f} / 100"
    )
    booster_line = f"\n[bold magenta]Booster:[/] {best.booster}" if best.booster else ""

    console.print(
        Panel(
            f"[bold green]Captain:[/] {pick.captain} ({cap_mult})\n"
            f"[bold yellow]Vice-Captain:[/] {pick.vice_captain} (1.5x)\n"
            f"[bold]Strategy:[/] {pick.strategy.value}\n"
            f"[bold]Expected Total:[/] {best.total_expected_points:.1f} pts\n"
            f"{budget_line}{booster_line}",
            title="Recommended Lineup",
            border_style="gold1",
        )
    )

    # Players table
    table = Table(title="Players")
    table.add_column("Role", style="cyan", width=6)
    table.add_column("Player", style="white")
    table.add_column("Team", style="dim")
    table.add_column("Credits", justify="right")
    table.add_column("E[pts]", justify="right", style="green")
    table.add_column("Tag", style="yellow")

    for p in sorted(best.players, key=lambda x: ("WK", "BAT", "AR", "BOWL").index(x.role)):
        tag = ""
        if p.name == pick.captain:
            tag = "(C)"
        elif p.name == pick.vice_captain:
            tag = "(VC)"

        table.add_row(
            p.role,
            p.name,
            p.team,
            f"{p.credit_cost:.1f}",
            f"{p.expected_points:.1f}",
            tag,
        )

    console.print(table)

    # Constraint check
    violations = best.validate()
    if violations:
        console.print(f"[bold red]WARNING: Constraint violations: {violations}[/]")
    else:
        console.print("[bold green]OK -- All IPL Fantasy constraints satisfied[/]")

    # Bandit recommendation
    rec = result.bandit_recommendation
    console.print(
        f"\n[dim]Bandit confidence: {rec.get('confidence', 0):.2f} "
        f"| Contest: {rec.get('contest_type', 'unknown')}[/]"
    )


if __name__ == "__main__":
    app()
