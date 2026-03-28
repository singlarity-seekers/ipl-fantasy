# AGENTS.md — Agent Instructions for IPL Fantasy

> Context and instructions for AI coding agents working on this project.

## Key Documents

| Document | What it covers |
|----------|---------------|
| [README.md](README.md) | Architecture, quick start, module table, CLI reference |
| [pyproject.toml](pyproject.toml) | Dependencies, entry points, build config |
| [data/fantasy_league_rules_2026.md](data/fantasy_league_rules_2026.md) | Official 2026 rules — squad constraints, transfers, boosters |
| [data/squads_2026.json](data/squads_2026.json) | 2026 IPL squads — 249 players, 10 teams, official credit costs from iplt20.com |
| [data/player_aliases.json](data/player_aliases.json) | ESPN name → Cricsheet name mapping (249 entries, verified) |
| [data/schedule_2026.json](data/schedule_2026.json) | Parsed IPL 2026 schedule (first 20 matches) |
| [data/raw/t20_recent_form.csv](data/raw/t20_recent_form.csv) | Recent T20 form (175 players, 12 leagues + SMAT) |

## Project Conventions

- **Entry point:** `cli/main.py` → installed as `ipl-fantasy` via pyproject.toml
- **Pipeline:** `orchestrator.py` chains: forecast → simulate → optimize → captain → rerank → bandit
- **Player pool:** Always from `data/squads_2026.json`, never from historical CSV
- **Forecaster trains** on historical IPL data but **predicts** for 2026 squad players only
- **Name resolution:** `src/forecast/models.py` uses `player_aliases.json` (ESPN → Cricsheet) before history lookup
- **Team abbreviations:** MI, CSK, RCB, KKR, SRH, DC, GT, PBKS, RR, LSG — resolved in `cli/main.py`

## Architecture

```
Historical Data (278K deliveries, 1169 matches)
    ↓
PlayerForecaster.fit(scorecards, deliveries)
    ├── Per-player history (career + recent 5 matches)
    ├── Venue-specific history (37 venues, player-venue pairs)
    ├── Batter-vs-team matchups (406 matchups)
    ├── T20 form data (175 players, 12 leagues)
    └── T20 WC snub boost (+8% for dropped players)
    ↓
MonteCarloSimulator (10K simulations)
    ↓
IPLFantasyOptimizer (ILP with budget/role/team/overseas constraints + boosters)
    ↓
CaptainSelector (P(50+) + per-player IPL history + role bonus)
    ↓
RewardModel + LineupBandit (Thompson Sampling)
    ↓
Output: OptimizedLineup with C/VC
```

## Forecasting Layers

The forecaster blends multiple signals (in order of application):

1. **IPL career average** (40%) + **recent 5-match form** (60%)
2. **Venue adjustment** — player-venue history (30% weight, min 5 matches) or venue multiplier fallback
3. **Opposition matchup** — batter SR vs bowling team (±10% for dominant/struggling matchups)
4. **T20 form** — 20% weight blend from non-IPL T20 leagues (with 35% domestic discount for SMAT-only)
5. **T20 WC snub boost** — +8% for players dropped from India's 2026 squad
6. **Cold-start** — if no IPL history: use T20 form estimate; if no form: role-based priors

## Known Gotchas

1. **Don't modify historical CSVs** — `matches.csv` and `deliveries.csv` are append-only
2. **3 squad players not in IPL Fantasy portal** — Muzarabani, Aman Rao, Dasun Shanaka
3. **Credit costs are static** — real credits change per match. Update `squads_2026.json` when new data available
4. **Schedule only has 20/70+ matches** — update `schedule_2026.json` when BCCI releases Part 2
5. **T20 form CSV needs periodic refresh** — run `uv run python scripts/fetch_t20_form.py --months 12`
6. **BCCI SMAT API** — JSONP format at `https://scores.bcci.tv/domesticstats/syed-mushtaq-ali-trophy-elite-2025-26/stats/player/full/mostRuns.js`
7. **Venue names have variants** — normalized via `_VENUE_ALIASES` in `models.py`. New venues may need adding.

## Agent Behavior Rules

1. **Say "I don't know" when you don't know.** Do not guess factual information (schedules, fixtures, dates, venues). Check authoritative sources in the project first.
2. **IPL 2026 schedule:** Consult `data/schedule_2026.json` or the PDF — do not rely on general knowledge.
3. **Injured players:** Use `--exclude` / `-x` flag on `generate`, `squad init`, and `plan` commands.

## Environment

- **Python 3.11+** required
- **Install:** `uv sync --extra dev`
- **Run:** `uv run <command>` — no manual venv needed
- **Tests:** `uv run pytest tests/ -v` — 83 tests, all passing
- **LLM sidecar** (optional): set `GOOGLE_API_KEY` or `OPENAI_API_KEY`
- **No external DB** — all data in CSV/JSON files under `data/`

## File Map

```
data/
├── raw/
│   ├── matches.csv, deliveries.csv     # Historical IPL (2007-2025)
│   ├── cricket_data_2026.csv           # Per-year player stats
│   ├── t20_recent_form.csv             # T20 form (12 leagues + SMAT)
│   └── ...IPL Schedule 2026_Part 1.pdf # Official schedule PDF
├── scraped/                            # Raw ODT files with official credits
├── squads_2026.json                    # 249 players, official credits
├── player_aliases.json                 # 249 name mappings (verified)
├── schedule_2026.json                  # First 20 matches parsed
└── fantasy_league_rules_2026.md        # Official 2026 rules

src/
├── scoring/fantasy.py                  # Fantasy point calculation
├── forecast/models.py                  # Forecaster (history + venue + matchup + form + snub)
├── forecast/cold_start.py              # Cold-start priors
├── simulation/monte_carlo.py           # 10K Monte Carlo sims
├── optimizer/fantasy_ilp.py            # ILP optimizer with boosters
├── optimizer/transfer_optimizer.py     # Transfer-aware ILP with look-ahead
├── captain/selector.py                 # C/VC: P(50+) + per-player history
├── reranker/                           # Reward model + Thompson Sampling
├── agent/orchestrator.py               # Full pipeline orchestrator
├── season/schedule.py                  # Schedule loader
├── season/state.py                     # Squad state persistence
├── data/ingest.py                      # CSV loading
├── data/features.py                    # Scorecard builder
├── llm/sidecar.py                      # LLM context
└── config.py                           # Constraints, scoring, boosters, transfers

cli/main.py                             # All CLI commands
scripts/fetch_t20_form.py               # T20 form data fetcher
scripts/fetch_2025_data.py              # Cricsheet→CSV converter
```

## Pending Work

1. Full IPL 2026 schedule (Part 2+ from BCCI)
2. Streamlit dashboard
3. Live API for match-day inputs
4. Transfer optimizer unit tests
5. `scripts/parse_schedule.py` for automated schedule updates
