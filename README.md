# IPL Fantasy Agentic System

AI-powered IPL Fantasy team optimizer using probabilistic forecasting, Monte Carlo simulation, and constrained optimization.

## Architecture

```
Historical IPL Data (278K deliveries) + T20 Form (12 leagues) + SMAT (BCCI API)
    ↓
Forecaster (career avg + venue + matchup + T20 form + snub boost)
    ↓
Monte Carlo Simulation (10K sims → point distributions)
    ↓
ILP Optimizer (budget/role/team/overseas constraints + boosters)
    ↓
Captain Selector (P(50+) + per-player IPL history + role bonus)
    ↓
Reward Reranker + Thompson Sampling Bandit → Final Lineup
```

## Quick Start

```bash
# Install
uv sync --extra dev

# Generate a lineup for a single match
uv run ipl-fantasy generate --team1 SRH --team2 RCB --venue "M Chinnaswamy Stadium" \
  -x "Pat Cummins" --booster foreign_stars

# Initialize season-long squad with 3-match look-ahead
uv run ipl-fantasy squad init --match 1 --look-ahead 3 -x "Pat Cummins"

# Plan transfers for next match
uv run ipl-fantasy plan --match 2 --max-transfers 3 --look-ahead 3

# Apply transfers
uv run ipl-fantasy apply --match 2 --confirm

# Post-toss optimization with confirmed playing XIs
uv run ipl-fantasy toss --match 1 --team1 RCB --team2 SRH \
  --team1-xi "Salt,Kohli,Patidar,..." --team2-xi "Head,Klaasen,..."

# Squad management
uv run ipl-fantasy squad show
uv run ipl-fantasy squad status
uv run ipl-fantasy squad history

# Player forecast
uv run ipl-fantasy forecast --player "Virat Kohli" --venue "M Chinnaswamy Stadium"

# Backtest
uv run ipl-fantasy backtest --season 2024
```

## Modules

| Module | Description |
|--------|-------------|
| `src/scoring/` | IPL Fantasy points calculator (exact rules from config) |
| `src/data/` | Dataset ingestion + feature engineering (scorecards, venue splits) |
| `src/forecast/` | Probabilistic forecasting: IPL history + venue + matchup + T20 form + snub boost |
| `src/simulation/` | Monte Carlo match simulation (N=10,000) |
| `src/optimizer/` | ILP lineup optimizer (with boosters) + transfer optimizer (with look-ahead) |
| `src/captain/` | C/VC selector: P(50+) + per-player IPL 50+ rate + role bonus |
| `src/reranker/` | Reward model + Thompson Sampling bandit |
| `src/llm/` | LLM sidecar for news/injury context (Gemini/OpenAI) |
| `src/agent/` | Agentic orchestrator (full pipeline) |
| `src/season/` | Schedule loader + squad state persistence + transfer tracking |
| `cli/` | Typer CLI with Rich output |

## Forecasting

The forecaster blends multiple signals:

| Layer | Weight | Description |
|-------|--------|-------------|
| IPL career avg | 40% | Historical fantasy points from all IPL seasons |
| Recent IPL form | 60% | Last 5 IPL matches (blended with career) |
| Venue adjustment | 30% | Player-specific venue history (min 5 matches) or venue multiplier |
| Opposition matchup | +/-10% | Batter SR vs bowling team (min 3 overs faced) |
| T20 form | 20% | Recent form across 12 T20 leagues (35% discount for SMAT-only) |
| T20 WC snub boost | +8% | Players dropped from India's 2026 T20 World Cup squad |

## Captain Selection

Captain choice has the biggest impact on fantasy points (2x multiplier). The selector scores each player:

```
score = 0.40 x E[2x pts] + 0.25 x ceiling + 0.15 x consistency
      + 0.20 x P(50+ in sims) + per-player IPL history bonus
```

All-rounders with high IPL 50+ rates (Russell 68%, Narine 52%) get significant captain bonuses. Backtested: captain scored 30+ in **90%** of matches.

## Season-Long Transfer Optimizer

Manages the 160-transfer budget across 70 league matches:

- **Fixture density weighting** — values players whose teams play more upcoming matches
- **Transfer penalty** (λ=5) — penalizes each transfer in the ILP objective
- **Look-ahead** — optimizes across next N matches, not just today
- **Booster support** — Triple Captain, Free Hit, Foreign Stars, Indian Warrior, Double Points, Wildcard
- **State persistence** — tracks squad, transfers, boosters, history in JSON

## Data

| File | Description |
|------|-------------|
| `data/squads_2026.json` | 249 players, 10 teams, official credit costs from iplt20.com |
| `data/player_aliases.json` | 249 ESPN→Cricsheet name mappings (verified) |
| `data/schedule_2026.json` | First 20 matches (from official IPL PDF) |
| `data/raw/t20_recent_form.csv` | T20 form: 175 players across 12 leagues + SMAT |
| `data/raw/matches.csv` | 1,169 historical IPL matches (2007-2025) |
| `data/raw/deliveries.csv` | 278K ball-by-ball records |
| `data/fantasy_league_rules_2026.md` | Official 2026 rules reference |

## Tests

```bash
uv run pytest tests/ -v    # 83 tests, all passing
```

## Backtest Results (IPL 2024)

Tested on last 10 matches of IPL 2024:
- **67% of optimal points** captured on average
- **Captain scored 30+** in 90% of matches
- MAE: 24.6 pts per player
