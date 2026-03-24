"""
LLM Sidecar for contextual intelligence.

Injects real-world context (injuries, weather, pitch reports, team news)
into the forecasting pipeline via structured LLM outputs.

Provider-agnostic: supports Google Gemini and OpenAI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlayerAdjustment:
    """LLM-generated adjustment for a player's forecast."""

    player: str
    adjustment_factor: float  # multiplier (e.g., 0.7 = -30%, 1.2 = +20%)
    confidence: float  # 0–1, how confident the LLM is
    reason: str
    source: str  # e.g., "injury report", "pitch report", "weather"


@dataclass
class MatchContext:
    """Structured match context extracted by the LLM."""

    venue: str
    weather: str
    pitch_type: str  # batting-friendly, bowling-friendly, balanced
    toss_impact: str
    key_absences: list[str]
    player_adjustments: list[PlayerAdjustment]
    narrative: str  # human-readable summary


SYSTEM_PROMPT = """You are an expert IPL cricket analyst. Given a match preview and news,
extract structured insights for fantasy cricket team selection.

You must respond with valid JSON matching this schema:
{
  "venue": "stadium name",
  "weather": "conditions description",
  "pitch_type": "batting-friendly | bowling-friendly | balanced",
  "toss_impact": "brief analysis",
  "key_absences": ["player1", "player2"],
  "player_adjustments": [
    {
      "player": "player name",
      "adjustment_factor": 0.8,
      "confidence": 0.7,
      "reason": "why this adjustment",
      "source": "injury report"
    }
  ],
  "narrative": "2-3 sentence match preview summary"
}

Rules:
- adjustment_factor: 1.0 = no change, <1.0 = downgrade, >1.0 = upgrade
- confidence: 0.0 to 1.0
- Only include adjustments for players with meaningful changes
- Be conservative with adjustments (0.7–1.3 range)
"""


class LLMSidecar:
    """
    LLM-powered context engine for enriching player forecasts
    with real-world information.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            provider: "gemini" or "openai".
            model: Model name override.
            api_key: API key (falls back to env vars).
        """
        self.provider = provider
        self.api_key = api_key
        self._client = None

        if provider == "gemini":
            self.model = model or "gemini-2.0-flash"
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_client(self):
        """Lazy-initialize the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "gemini":
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.error("google-generativeai not installed")
                raise
        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed")
                raise

        return self._client

    def analyze_match(
        self,
        match_info: str,
        news_context: str = "",
    ) -> MatchContext:
        """
        Analyze match context and generate player adjustments.

        Args:
            match_info: Match description (teams, venue, date).
            news_context: Additional news/injury reports.

        Returns:
            MatchContext with structured insights.
        """
        prompt = f"""Match Information:
{match_info}

Recent News & Updates:
{news_context}

Analyze this match and provide structured insights for fantasy team selection."""

        try:
            raw_response = self._call_llm(prompt)
            return self._parse_response(raw_response)
        except Exception as e:
            logger.error("LLM analysis failed: %s", e)
            return self._fallback_context(match_info)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return raw text response."""
        client = self._get_client()

        if self.provider == "gemini":
            response = client.generate_content(
                f"{SYSTEM_PROMPT}\n\n{prompt}",
                generation_config={"response_mime_type": "application/json"},
            )
            return response.text
        elif self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> MatchContext:
        """Parse LLM JSON response into MatchContext."""
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)

        adjustments = [
            PlayerAdjustment(
                player=adj["player"],
                adjustment_factor=float(adj.get("adjustment_factor", 1.0)),
                confidence=float(adj.get("confidence", 0.5)),
                reason=adj.get("reason", ""),
                source=adj.get("source", "llm"),
            )
            for adj in data.get("player_adjustments", [])
        ]

        return MatchContext(
            venue=data.get("venue", ""),
            weather=data.get("weather", ""),
            pitch_type=data.get("pitch_type", "balanced"),
            toss_impact=data.get("toss_impact", ""),
            key_absences=data.get("key_absences", []),
            player_adjustments=adjustments,
            narrative=data.get("narrative", ""),
        )

    def _fallback_context(self, match_info: str) -> MatchContext:
        """Return a neutral context when LLM is unavailable."""
        return MatchContext(
            venue="Unknown",
            weather="Normal",
            pitch_type="balanced",
            toss_impact="Standard advantage",
            key_absences=[],
            player_adjustments=[],
            narrative=f"No LLM context available for: {match_info[:100]}",
        )

    def apply_adjustments(
        self,
        context: MatchContext,
        player_expected: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply LLM-generated adjustments to player expected points.

        Returns updated {player: adjusted_expected_points}.
        """
        adjusted = dict(player_expected)

        for adj in context.player_adjustments:
            if adj.player in adjusted:
                original = adjusted[adj.player]
                # Weight adjustment by confidence
                effective_factor = 1.0 + (adj.adjustment_factor - 1.0) * adj.confidence
                adjusted[adj.player] = original * effective_factor
                logger.info(
                    "Adjusted %s: %.1f → %.1f (factor=%.2f, confidence=%.2f, reason=%s)",
                    adj.player, original, adjusted[adj.player],
                    adj.adjustment_factor, adj.confidence, adj.reason,
                )

        return adjusted
