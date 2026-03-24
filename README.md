# IPL Fantasy Agentic System

AI-powered IPL Fantasy team optimizer using probabilistic forecasting, Monte Carlo simulation, and constrained optimization.

## Architecture

```
Data Pipeline → Forecaster → Monte Carlo Sim → ILP Optimizer → Captain Selector → Reward Reranker → Bandit → Output
                     ↑                                                                                    ↑
                LLM Sidecar ──────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download IPL dataset from Kaggle and place in data/raw/
# https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

# Generate a team for a single match
ipl-fantasy generate --team1 "Mumbai Indians" --team2 "Chennai Super Kings" --venue "Wankhede Stadium"

# Initialize season-long squad (auto-generates optimal XI for match 1)
ipl-fantasy squad init --match 1

# Initialize with look-ahead optimization (considers next 5 matches for fixture density)
ipl-fantasy squad init --match 1 --look-ahead 5

# Plan transfers for upcoming match
ipl-fantasy plan --match 5 --max-transfers 3 --look-ahead 3

# Apply planned transfers
ipl-fantasy apply --match 5 --confirm

# Post-toss optimization (confirmed playing XIs)
ipl-fantasy toss --match 5 --team1 MI --team2 CSK \
    --team1-xi "Rohit Sharma,..." --team2-xi "Ruturaj Gaikwad,..."

# Check squad status
ipl-fantasy squad status

# View transfer history
ipl-fantasy squad history

# Player forecast
ipl-fantasy forecast --player "Virat Kohli"

# Backtest a season
ipl-fantasy backtest --season 2024
```

## Modules

| Module | Description |
|--------|-------------|
| `src/scoring/` | IPL Fantasy points calculator (exact rules) |
| `src/data/` | Kaggle dataset ingestion + feature engineering |
| `src/forecast/` | Probabilistic player forecasting (statistical + XGBoost) |
| `src/simulation/` | Monte Carlo match simulation (N=10,000) |
| `src/optimizer/` | ILP lineup optimizer + **transfer optimizer with fixture density** |
| `src/captain/` | Simulation-aware C/VC selector (safe/differential/contrarian) |
| `src/reranker/` | Reward model + Thompson Sampling/UCB bandit |
| `src/llm/` | LLM sidecar for news/injury context (Gemini/OpenAI) |
| `src/agent/` | Agentic orchestrator (full pipeline) |
| `src/season/` | **NEW: Schedule loader + squad state management** |
| `cli/` | Typer CLI with Rich output |

## Tests

```bash
# Run all tests (83 tests)
pytest tests/ -v
```

## LLM Sidecar

Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` environment variable, then:

```bash
ipl-fantasy generate --team1 "MI" --team2 "CSK" --venue "Wankhede" --llm
```

## Season-Long Transfer Optimizer

The new transfer optimizer helps manage your 160-transfer budget across 70 league matches:

- **Fixture Density Weighting**: Values players higher if their teams have busy upcoming schedules
- **Transfer Penalties**: ILP formulation penalizes each transfer to minimize churn
- **Look-ahead Optimization**: Plans transfers considering next N matches, not just today
- **State Persistence**: Tracks squad, transfers used, boosters, and transfer history

### Example Workflow

```bash
# 1. Initialize squad for Match 1 (with look-ahead)
ipl-fantasy squad init --match 1 --look-ahead 5

# 2. Before each match, plan transfers
ipl-fantasy plan --match 5 --max-transfers 3 --look-ahead 3

# 3. After toss, use confirmed XIs for final optimization
ipl-fantasy toss --match 5 --team1-xi "..." --team2-xi "..."

# 4. Apply the transfers
ipl-fantasy apply --match 5 --confirm

# 5. Check status anytime
ipl-fantasy squad status
```

## Data Files

- `data/squads_2026.json` - 10 IPL teams, 245 players with credit costs
- `data/schedule_2026.json` - First 20 matches (parsed from official IPL PDF)
- `data/player_aliases.json` - ESPN → Cricsheet name mappings
- `data/raw/matches.csv` - Historical match data (2008-2025)
- `data/raw/deliveries.csv` - Ball-by-ball delivery data (278K+ records)
