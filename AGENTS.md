# AGENTS.md — Agent Instructions for IPL Fantasy

> This file provides context and instructions for AI coding agents working on this project.  
> It references existing docs — **do not duplicate** what's already covered elsewhere.

## Key Documents

| Document | What it covers |
|----------|---------------|
| [README.md](README.md) | Architecture diagram, quick start, module table, test/LLM usage |
| [HANDOFF.md](HANDOFF.md) | Detailed progress, what's done vs pending, **known gotchas**, file map, next steps |
| [pyproject.toml](pyproject.toml) | Dependencies, entry points, build config |
| [data/squads_2026.json](data/squads_2026.json) | 2026 IPL squads (10 teams, 245 players) — source of truth for player pool |
| [data/player_aliases.json](data/player_aliases.json) | ESPN name → Cricsheet name mapping (245 entries) |
| [data/fantasy_league_rules_2026.md](data/fantasy_league_rules_2026.md) | IPL Fantasy League 2026 rules — squad constraints, transfers, boosters, tiebreakers |
| [data/raw/...IPL Schedule 2026_Part 1.pdf](data/raw/1773233174530_TATA%20IPL%20Schedule%202026_Part%201.pdf) | Official IPL 2026 match schedule (dates, teams, venues) |

**Start by reading `HANDOFF.md`** — it has the full context, gotchas, and verification commands.

## Project Conventions

- **Entry point:** `cli/main.py` → installed as `ipl-fantasy` via pyproject.toml
- **Pipeline flow:** `orchestrator.py` chains: forecast → simulate → optimize → captain → rerank → bandit
- **Player pool:** Always sourced from `data/squads_2026.json`, never from historical CSV
- **Forecaster trains** on historical data (`data/raw/matches.csv` + `deliveries.csv`) but **predicts** for 2026 squad players only
- **Name resolution:** `src/forecast/models.py` loads `data/player_aliases.json` to map ESPN full names → Cricsheet initials before history lookup
- **Team names:** CLI accepts abbreviations (MI, CSK, RCB, etc.) — resolved in `cli/main.py` via `TEAM_ALIASES`

## Critical Context

1. **The `deliveries.csv` is 278K rows** — loading takes 3-5s. Don't add unnecessary re-reads. Consider Parquet conversion if performance is an issue.

2. **Name mapping is the #1 priority** — Without correct aliases, star players (Bumrah, Kohli, Rohit) get cold-start priors instead of their real stats. See `HANDOFF.md` § "Known Gotchas" for specific wrong mappings.

3. **Don't modify historical CSVs** — `matches.csv` and `deliveries.csv` are append-only. The 2025 season was appended via `scripts/fetch_2025_data.py`.

4. **Tests:** `pytest tests/ -v` — 61 tests existed before the recent `models.py` alias changes. Re-run to verify nothing broke.

## Agent Behavior Rules

1. **Say "I don't know" when you don't know.** Do not guess or fabricate factual information (match schedules, team lineups, dates, venues, etc.). If you are not certain of a fact, say so and point the user to the authoritative source.

2. **IPL 2026 schedule:** The official schedule is in `data/raw/1773233174530_TATA IPL Schedule 2026_Part 1.pdf`. Always consult this file for match dates, teams, and venues — do not rely on general knowledge or assumptions about IPL openers/fixtures.

## Environment

- **Python 3.11+** required
- **Install:** `uv sync --extra dev` (preferred) or `pip install -e ".[dev]"`
- **Run commands:** Use `uv run <command>` — no manual venv activation needed
- **LLM sidecar** (optional): set `GOOGLE_API_KEY` or `OPENAI_API_KEY`
- **No external DB** — everything is CSV/JSON files in `data/`
