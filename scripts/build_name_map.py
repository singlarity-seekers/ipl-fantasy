import pandas as pd
import json
import re
from pathlib import Path

def get_cricsheet_players():
    # Only load the relevant columns to save memory and time
    df = pd.read_csv('data/raw/deliveries.csv', usecols=['batter', 'bowler', 'non_striker'])
    return set(df['batter'].dropna().unique()) | set(df['bowler'].dropna().unique()) | set(df['non_striker'].dropna().unique())

def get_espn_players():
    with open('data/squads_2026.json') as f:
        squads = json.load(f)
    return [p['name'] for t in squads['teams'].values() for p in t['players']]

cricsheet = get_cricsheet_players()
espn = get_espn_players()

mapping = {}
unmapped = []

# 1. Exact matches
for espn_name in espn.copy():
    if espn_name in cricsheet:
        mapping[espn_name] = espn_name
        espn.remove(espn_name)

# 2. Heuristic matching: e.g. "Rohit Sharma" -> "RG Sharma", "Virat Kohli" -> "V Kohli"
# Rule: Last name must match exactly. First initial must match.
for espn_name in espn.copy():
    parts = espn_name.split()
    if len(parts) >= 2:
        first_initial = parts[0][0].upper()
        # For names like "Dewald Brevis", last name is "Brevis".
        last_name = parts[-1]
        
        candidates = []
        for c in cricsheet:
            if c.endswith(last_name):
                c_parts = c.split()
                if len(c_parts) > 1 and c_parts[0][0].upper() == first_initial:
                    candidates.append(c)
        
        if len(candidates) == 1:
            mapping[espn_name] = candidates[0]
            espn.remove(espn_name)
        elif len(candidates) > 1:
            unmapped.append((espn_name, f"Multiple candidates: {candidates}"))
        else:
            # Check edge cases like "Quinton de Kock" -> "Q de Kock"
            target_suffix = " ".join(parts[1:])
            candidates = [c for c in cricsheet if c.endswith(target_suffix) and c.startswith(first_initial)]
            if len(candidates) == 1:
                mapping[espn_name] = candidates[0]
                espn.remove(espn_name)

print(f"Mapped {len(mapping)} players.")
print(f"Unmapped {len(espn)} players.")
for u in unmapped:
    print(f"Conflicted: {u}")
if espn:
    print("Some completely unmapped:")
    for e in espn[:15]:
        print(f"  {e}")
    
# Save raw map to let us inspect
with open('data/player_aliases.json', 'w') as f:
    json.dump(mapping, f, indent=2)
