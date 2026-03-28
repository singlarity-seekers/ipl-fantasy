"""
Agentic orchestrator — the central pipeline that ties all modules together.

COMPLETE PIPELINE (for single match):
-------------------------------------
1. Ingest match context (teams, venue, date)
2. (Optional) LLM sidecar → pull news, generate adjustments
3. Feature pipeline → build player features from historical data
4. Forecast → probabilistic predictions (mean ± std for each player)
5. Monte Carlo → 10,000 simulations using those distributions
6. Optimizer → ILP generates top-K valid lineups within constraints
7. Captain selector → assigns C/VC per lineup based on distributions
8. Reranker → scores & ranks lineups via reward model (floor, ceiling, diversity)
9. Bandit → Thompson Sampling selects final lineup
10. Output → CLI / Dashboard

TRANSFER PLANNING (season-long mode):
------------------------------------
The orchestrator also supports transfer-aware optimization:
- plan_transfers() uses TransferOptimizer instead of IPLFantasyOptimizer
- Considers fixture density over next N matches
- Penalizes each transfer in objective function
- Respects 160-transfer season budget

This is called by the CLI 'plan' command to recommend transfers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.captain.selector import CaptainPick, CaptainSelector, CaptainStrategy
from src.config import SIM_CONFIG
from src.forecast.models import PlayerForecaster
from src.optimizer.fantasy_ilp import IPLFantasyOptimizer, OptimizedLineup, PlayerSlot
from src.reranker.bandit import ContestType, LineupBandit
from src.reranker.reward_model import RewardModel, extract_lineup_features
from src.simulation.monte_carlo import MonteCarloSimulator, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class MatchInput:
    """Input specification for a match to generate teams for."""

    team1: str
    team2: str
    venue: str
    date: str | None = None
    # Player pool with credit costs
    players: list[dict] = field(default_factory=list)
    # Each dict: {"name": str, "role": str, "team": str, "credit_cost": float}
    news_context: str = ""  # raw news/injury text for LLM


@dataclass
class PipelineResult:
    """Complete output of the agentic pipeline."""

    lineups: list[OptimizedLineup]
    captain_picks: list[CaptainPick]
    simulation_result: SimulationResult
    selected_lineup_index: int
    bandit_recommendation: dict
    llm_context: dict | None = None

    @property
    def best_lineup(self) -> OptimizedLineup:
        return self.lineups[self.selected_lineup_index]

    @property
    def best_captain_pick(self) -> CaptainPick:
        return self.captain_picks[self.selected_lineup_index]


class Orchestrator:
    """
    Central agentic orchestrator that runs the full pipeline:
    forecast → simulate → optimize → captain → rerank → select.
    """

    def __init__(
        self,
        forecaster: PlayerForecaster | None = None,
        simulator: MonteCarloSimulator | None = None,
        optimizer: IPLFantasyOptimizer | None = None,
        captain_selector: CaptainSelector | None = None,
        reward_model: RewardModel | None = None,
        bandit: LineupBandit | None = None,
        use_llm: bool = False,
        llm_provider: str = "gemini",
    ):
        self.forecaster = forecaster or PlayerForecaster()
        self.simulator = simulator or MonteCarloSimulator()
        self.optimizer = optimizer or IPLFantasyOptimizer()
        self.captain_selector = captain_selector or CaptainSelector()
        self.reward_model = reward_model or RewardModel()
        self.bandit = bandit or LineupBandit(contest_type=ContestType.MEGA_CONTEST)
        self.use_llm = use_llm
        self.llm_provider = llm_provider

        self._llm_sidecar = None

    def fit_from_data(
        self,
        scorecards: pd.DataFrame,
        deliveries: pd.DataFrame | None = None,
    ) -> Orchestrator:
        """
        Fit the forecaster from historical scorecards.

        Args:
            scorecards: DataFrame from features.build_player_scorecards().
            deliveries: Optional ball-by-ball data for matchup analysis.
        """
        self.forecaster.fit(scorecards, deliveries=deliveries)
        # Pass player history to captain selector for per-player 50+ rate scoring
        self.captain_selector.set_player_history(self.forecaster._player_history)
        logger.info("Orchestrator fitted on %d scorecard records", len(scorecards))
        return self

    def generate_team(
        self,
        match: MatchInput,
        top_k: int = 5,
        captain_strategy: CaptainStrategy = CaptainStrategy.SAFE,
        n_simulations: int | None = None,
        booster: str | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for a match.

        Args:
            match: MatchInput with teams, venue, player pool.
            top_k: Number of diverse lineups to generate.
            captain_strategy: Captain selection strategy.
            n_simulations: Override simulation count.

        Returns:
            PipelineResult with lineups, captains, and recommendation.
        """
        logger.info(
            "Generating team for %s vs %s at %s",
            match.team1,
            match.team2,
            match.venue,
        )

        # ── Step 1: LLM Context (optional) ─────────────────────────────
        llm_context = None
        adjustments = {}
        if self.use_llm:
            llm_context = self._run_llm(match)
            if llm_context:
                # Build adjustment map
                for adj in llm_context.get("player_adjustments", []):
                    adjustments[adj["player"]] = adj["adjustment_factor"]

        # ── Step 2: Forecast ────────────────────────────────────────────
        player_names = [p["name"] for p in match.players]
        roles = {p["name"]: p.get("role", "BAT") for p in match.players}
        player_teams = {p["name"]: p.get("team", match.team1) for p in match.players}

        forecasts = self.forecaster.forecast_match(
            players=player_names,
            roles=roles,
            venue=match.venue,
            player_teams=player_teams,
            team1=match.team1,
            team2=match.team2,
        )

        # Apply LLM adjustments
        if adjustments:
            for f in forecasts:
                if f.player in adjustments:
                    factor = adjustments[f.player]
                    f.expected_points *= factor
                    f.std_points *= abs(factor)
                    logger.info("LLM adjusted %s: factor=%.2f", f.player, factor)

        # ── Step 3: Monte Carlo Simulation ──────────────────────────────
        if n_simulations:
            self.simulator.n_simulations = n_simulations

        sim_result = self.simulator.simulate_match(forecasts)

        # ── Step 4: Build player slots for optimizer ────────────────────
        player_slots = []
        for p_info in match.players:
            name = p_info["name"]
            sim_row = sim_result.summary[sim_result.summary["player"] == name]

            expected = float(sim_row["mean_fp"].iloc[0]) if not sim_row.empty else 20.0
            ceiling = float(sim_row["ceiling_95"].iloc[0]) if not sim_row.empty else 40.0
            consistency = float(sim_row["consistency"].iloc[0]) if not sim_row.empty else 0.5

            player_slots.append(
                PlayerSlot(
                    name=name,
                    role=p_info.get("role", "BAT"),
                    team=p_info.get("team", match.team1),
                    credit_cost=p_info.get("credit_cost", 8.0),
                    expected_points=expected,
                    ceiling_95=ceiling,
                    consistency=consistency,
                    nationality=p_info.get("nationality", "Indian"),
                )
            )

        # ── Step 5: Optimize → Top-K lineups ───────────────────────────
        if booster:
            self.optimizer.booster = booster
            logger.info("Booster active: %s", booster)

        lineups = self.optimizer.generate_top_k(
            players=player_slots,
            k=top_k,
        )

        if not lineups:
            raise RuntimeError("Optimizer could not generate any valid lineups")

        # ── Step 6: Captain / VC selection ──────────────────────────────
        captain_picks = []
        for lineup in lineups:
            lineup_names = [p.name for p in lineup.players]
            pick = self.captain_selector.select(
                lineup_players=lineup_names,
                sim_result=sim_result,
                strategy=captain_strategy,
            )
            lineup.captain = pick.captain
            lineup.vice_captain = pick.vice_captain
            captain_picks.append(pick)

        # ── Step 7: Rerank via reward model ─────────────────────────────
        lineup_features = [
            extract_lineup_features(
                lineup,
                sim_summary=sim_result.summary,
            )
            for lineup in lineups
        ]

        ranked = self.reward_model.rank_lineups(lineup_features)

        # Reorder lineups by reward model score
        reordered_lineups = [lineups[idx] for idx, _ in ranked]
        reordered_picks = [captain_picks[idx] for idx, _ in ranked]

        # ── Step 8: Bandit selection ────────────────────────────────────
        self.bandit.initialize_arms(ranked)
        selected_arm = self.bandit.select()
        recommendation = self.bandit.get_recommendation()

        result = PipelineResult(
            lineups=reordered_lineups,
            captain_picks=reordered_picks,
            simulation_result=sim_result,
            selected_lineup_index=selected_arm,
            bandit_recommendation=recommendation,
            llm_context={"context": llm_context.__dict__} if llm_context else None,
        )

        logger.info(
            "Pipeline complete. Selected lineup %d (E[pts]=%.1f, C=%s, VC=%s)",
            selected_arm,
            result.best_lineup.total_expected_points,
            result.best_lineup.captain,
            result.best_lineup.vice_captain,
        )

        return result

    def plan_transfers(
        self,
        match: MatchInput,
        current_squad: list[str],
        max_transfers: int,
        current_match: int = 1,
        look_ahead: int = 0,
        schedule=None,
    ) -> "TransferPlan":
        """
        Plan optimal transfers for a match using the transfer-aware optimizer.

        Args:
            match: MatchInput with teams, venue, player pool
            current_squad: List of 11 player names currently in squad
            max_transfers: Maximum transfers allowed for this window
            current_match: Current match number (for fixture density)
            look_ahead: Number of future matches to consider (0 = disabled)
            schedule: Optional Schedule object for fixture density

        Returns:
            TransferPlan with recommended transfers
        """
        from src.optimizer.transfer_optimizer import TransferOptimizer, TransferPlan
        from src.season.schedule import Schedule

        logger.info(
            "Planning transfers for Match %d (%s vs %s), max=%d",
            current_match,
            match.team1,
            match.team2,
            max_transfers,
        )

        # Step 1: Run forecast pipeline (reuses existing logic)
        player_names = [p["name"] for p in match.players]
        roles = {p["name"]: p.get("role", "BAT") for p in match.players}

        forecasts = self.forecaster.forecast_match(
            players=player_names,
            roles=roles,
            venue=match.venue,
        )

        # Step 2: Monte Carlo simulation
        sim_result = self.simulator.simulate_match(forecasts)

        # Step 3: Build player slots
        player_slots = []
        for p_info in match.players:
            name = p_info["name"]
            sim_row = sim_result.summary[sim_result.summary["player"] == name]

            expected = float(sim_row["mean_fp"].iloc[0]) if not sim_row.empty else 20.0
            ceiling = float(sim_row["ceiling_95"].iloc[0]) if not sim_row.empty else 40.0
            consistency = float(sim_row["consistency"].iloc[0]) if not sim_row.empty else 0.5

            player_slots.append(
                PlayerSlot(
                    name=name,
                    role=p_info.get("role", "BAT"),
                    team=p_info.get("team", match.team1),
                    credit_cost=p_info.get("credit_cost", 8.0),
                    expected_points=expected,
                    ceiling_95=ceiling,
                    consistency=consistency,
                    nationality=p_info.get("nationality", "Indian"),
                )
            )

        # Step 4: Transfer optimization
        optimizer = TransferOptimizer(schedule=schedule)

        transfer_plan = optimizer.optimize(
            current_squad=current_squad,
            available_players=player_slots,
            max_transfers=max_transfers,
            current_match=current_match,
            look_ahead=look_ahead,
        )

        logger.info(
            "Transfer plan: %d transfers, E[pts]=%.1f, keeping %d players",
            transfer_plan.num_transfers,
            transfer_plan.expected_points,
            len(transfer_plan.kept_players),
        )

        return transfer_plan

    def _run_llm(self, match: MatchInput):
        """Run LLM sidecar for context extraction."""
        try:
            if self._llm_sidecar is None:
                from src.llm.sidecar import LLMSidecar

                self._llm_sidecar = LLMSidecar(provider=self.llm_provider)

            match_info = f"{match.team1} vs {match.team2} at {match.venue}"
            if match.date:
                match_info += f" on {match.date}"

            return self._llm_sidecar.analyze_match(
                match_info=match_info,
                news_context=match.news_context,
            )
        except Exception as e:
            logger.warning("LLM sidecar failed: %s", e)
            return None
