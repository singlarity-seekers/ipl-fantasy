# Away-Ground Penalty

## Concept

Apply a performance adjustment based on whether a player is playing at their team's **home ground** vs **away ground**. Home teams typically have advantages in:

- **Pitch familiarity**: Local players understand how the pitch behaves over 20 overs
- **Crowd support**: Home crowd energy (though less relevant in IPL with neutral venues)  
- **Travel fatigue**: Away teams may arrive late with less preparation time
- **Historical performance**: Teams like RCB dominate at Chinnaswamy, MI at Wankhede

## Why It Matters

Current forecaster treats all venues equally. But:
- Kohli averages **52 points** at Chinnaswamy vs **38 points** at other venues
- Rohit Sharma scores **20% more** at Wankhede than neutral venues
- Bowlers with local knowledge (e.g., Ashwin at Chepauk) exploit pitch conditions better

This could significantly improve prediction accuracy for captain/VC selection.

## Implementation Approach

### Option 1: Binary Home/Away Penalty

```python
# In src/forecast/models.py
def forecast(
    self, 
    player: str, 
    venue: str,
    home_team: str | None = None,  # NEW
    away_team: str | None = None,  # NEW
) -> PlayerForecast:
    base_forecast = self._calculate_base_forecast(player, venue)
    
    # Apply home/away adjustment
    player_team = self._get_player_team(player)
    if home_team and player_team == home_team:
        adjustment = 1.07  # +7% home advantage
    elif away_team and player_team == away_team:
        adjustment = 0.95  # -5% away disadvantage
    else:
        adjustment = 1.0
    
    return PlayerForecast(
        expected_points=base_forecast.expected_points * adjustment,
        std_points=base_forecast.std_points * adjustment,  # uncertainty scales too
        ...
    )
```

**Penalties/Bonuses:**
- Home: +5-10% expected points
- Away: -5-8% expected points  
- Neutral (playoffs): ±0%

### Option 2: Venue-Specific Player History (Preferred)

Rather than binary home/away, use historical performance at specific venues:

```python
# Calculate venue-specific stats from historical data
venue_history = {
    "Virat Kohli": {
        "Chinnaswamy Stadium": {"mean": 52.3, "std": 18.1, "matches": 45},
        "Wankhede Stadium": {"mean": 38.2, "std": 22.4, "matches": 12},
        "Eden Gardens": {"mean": 41.5, "std": 20.8, "matches": 8},
    }
}

# Use venue-specific if sufficient sample size (≥5 matches)
if venue in venue_history[player] and venue_history[player][venue]["matches"] >= 5:
    use_venue_specific_stats()
else:
    fall_back_to_global_stats()
```

## Challenges

### 1. IPL Venue Complexity

- **"Home" is fluid**: DC plays home games in Vizag (not Delhi), PBKS in new Chandigarh stadium
- **Neutral venues**: Playoffs often at venues neither team calls home
- **New stadiums**: Some 2026 venues have minimal historical IPL data
- **Venue changes**: Teams shift "home" bases frequently in modern IPL

### 2. Sample Size Issues

- Need **≥5 matches** per (player, venue) pair for reliable stats
- A player like Ruturaj Gaikwad has played at Chinnaswamy 30+ times → reliable
- But a new 2026 player at a new venue (e.g., New Chandigarh) → insufficient data
- Cold-start problem for rookies and new venues

### 3. Interaction with Existing Adjustments

The system already has:
- Venue adjustments in Monte Carlo (`simulation/monte_carlo.py:99`)
- Player history blending (career avg + recent form)
- Role-based cold-start priors

Adding home/away could **double-count** venue effects if not careful.

### 4. Data Verification Required

Must verify for each 2026 venue:
- How many historical matches exist?
- Which teams call it "home"?
- Is it truly a home advantage or just familiarity?

## Data Requirements

From `data/raw/matches.csv`:
- Venue names (must normalize: "MA Chidambaram" vs "Chepauk Stadium")
- Home/away team assignments per match
- Win rates by venue (validate if home advantage exists)

From `data/raw/deliveries.csv`:
- Player performance per venue (runs, wickets, fantasy points)
- Minimum sample sizes per (player, venue)

## Decision Matrix

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Binary Home/Away** | Simple, easy to implement | IPL venues are complex; ignores individual player variance | ❌ Not recommended |
| **Venue-Specific History** | Captures real player-venue fit | Requires large sample sizes; sparse for new venues | ⚠️ Conditional use |
| **Team Home Advantage** | Aggregated stats more reliable | Loses individual player nuance | ⚠️ Backup option |

## Recommendation

**Defer implementation until data verification complete.**

**Steps before building:**

1. **Analyze venue coverage**: 
   ```python
   # Count matches per venue in historical data
   venue_counts = matches_df.groupby('venue').size()
   # Identify which 2026 venues have <20 historical matches
   ```

2. **Calculate home advantage magnitude**:
   ```python
   # Is home win rate actually >50% in IPL?
   home_win_rate = matches_df[matches_df['winner'] == matches_df['team1']].shape[0] / len(matches_df)
   # If ~50%, home advantage is minimal in IPL
   ```

3. **Evaluate per-player venue variance**:
   ```python
   # Do players actually vary significantly by venue?
   kohli_by_venue = deliveries_df[deliveries_df['batsman'] == 'V Kohli'].groupby('venue')['fantasy_points'].agg(['mean', 'std', 'count'])
   # If variance <15%, not worth the complexity
   ```

**If data supports it:**
- Implement **venue-specific conditioning** (Option 2)
- Only apply when ≥5 historical matches at venue
- Fall back to global stats for cold-start cases

**If data is insufficient:**
- Do not implement — complexity not worth minimal accuracy gain
- Focus instead on improving player form tracking or weather adjustments

## Alternative: Simpler Venue Boost

Instead of home/away, simpler approach:

```python
# Check if player has historically performed well at this venue
def venue_boost(player, venue):
    history = get_player_venue_history(player, venue)
    if history["count"] >= 5 and history["mean"] > global_mean * 1.2:
        return 1.10  # 10% boost for "favorite" venue
    return 1.0
```

This captures "comfort venues" without home/away complexity.

## Conclusion

**Status:** IMPLEMENTED (2026-03-28)

**What was built:**
- Venue-specific player history (30% weight, min 5 matches at venue) — Option 2 from above
- Venue multiplier fallback (37 venues) for players without venue-specific data
- Venue name normalization (`_VENUE_ALIASES` in `src/forecast/models.py`)
- Opposition matchup adjustment (batter SR vs bowling team, ±10%)
- Integrated into `PlayerForecaster.fit()` and `forecast()` methods

**Data analysis confirmed:**
- Venue effects range from +7% (Arun Jaitley, Delhi) to -6% (Subrata Roy Sahara)
- Chinnaswamy premium: +4-5% for batters
- Player-venue variance is massive: Kohli 78.9 at Chinnaswamy vs 27.4 at worst (51.5 gap)
- 406 batter-vs-team matchups computed with min 18 balls faced
