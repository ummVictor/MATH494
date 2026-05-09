import os
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# SIMPLE TERMINAL PREDICTION TOOL
# ------------------------------------------------------------
# Before using this script:
# 1) Run final_presentation.py first.
# 2) Make sure BUNDLE_PATH points to the saved joblib file.
# ------------------------------------------------------------

BUNDLE_PATH = r"C:\Users\victo\Desktop\GIT\MATH494\final_presentation_outputs\saved_models\player_prediction_bundle.joblib"

TARGET_LABELS = {
    "next_pts": "Predicted next-season PTS",
    "next_reb": "Predicted next-season REB",
    "next_ast": "Predicted next-season AST",
    "next_ts_pct": "Predicted next-season TS%",
}



def format_next_season_label(season_value):
    if pd.isna(season_value):
        return "next season"
    season_str = str(season_value)
    parts = season_str.split("-")
    try:
        start = int(parts[0])
        next_start = start + 1
        next_end = str(next_start + 1)[-2:]
        return f"{next_start}-{next_end}"
    except Exception:
        return "next season"



def choose_player_name(input_name, available_names):
    exact = [name for name in available_names if name.lower() == input_name.lower()]
    if exact:
        return exact[0]

    contains = [name for name in available_names if input_name.lower() in name.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        print("\nMultiple close matches found:")
        for idx, name in enumerate(contains[:10], start=1):
            print(f"  {idx}. {name}")
        pick = input("Choose a number, or press Enter to cancel: ").strip()
        if pick.isdigit():
            pick_num = int(pick)
            if 1 <= pick_num <= min(10, len(contains)):
                return contains[pick_num - 1]
        return None

    close = get_close_matches(input_name, available_names, n=5, cutoff=0.6)
    if close:
        print("\nDid you mean one of these?")
        for idx, name in enumerate(close, start=1):
            print(f"  {idx}. {name}")
        pick = input("Choose a number, or press Enter to cancel: ").strip()
        if pick.isdigit():
            pick_num = int(pick)
            if 1 <= pick_num <= len(close):
                return close[pick_num - 1]
    return None



def predict_for_player(player_name, bundle):
    latest_rows = bundle["latest_rows"]
    available_names = sorted(latest_rows["player_name"].dropna().unique().tolist())

    chosen = choose_player_name(player_name, available_names)
    if chosen is None:
        print("\nPlayer not found.")
        return

    row = latest_rows[latest_rows["player_name"] == chosen].sort_values("season_start").tail(1).copy()
    if row.empty:
        print("\nNo usable row found for that player.")
        return

    player_row = row.iloc[0]
    current_season = player_row.get("season", "Unknown")
    predicted_season = format_next_season_label(current_season)

    print("\n" + "=" * 60)
    print(f"Player: {chosen}")
    print(f"Most recent season in data: {current_season}")
    print(f"Predicted season: {predicted_season}")
    print("-" * 60)
    print("Current-season stats used as input")
    print(f"PTS: {player_row.get('pts', np.nan):.2f}")
    print(f"REB: {player_row.get('reb', np.nan):.2f}")
    print(f"AST: {player_row.get('ast', np.nan):.2f}")
    print(f"TS%: {player_row.get('ts_pct', np.nan):.3f}")
    print(f"Age: {player_row.get('age', np.nan):.1f}")
    print(f"Team: {player_row.get('team_abbreviation', 'Unknown')}")
    print("-" * 60)

    for target, info in bundle["targets"].items():
        model = info["model"]
        pred = float(model.predict(row[info["features"]])[0])
        if target == "next_ts_pct":
            print(f"{TARGET_LABELS[target]}: {pred:.3f}")
        else:
            print(f"{TARGET_LABELS[target]}: {pred:.2f}")

    print("-" * 60)
    print("Note: These predictions use only historical stats and context features in the dataset.")
    print("They do not account for injuries, role changes, trades, coaching changes, or off-court factors.")
    print("=" * 60 + "\n")



def main():
    if not os.path.exists(BUNDLE_PATH):
        print("Saved model bundle not found.")
        print("Run final_presentation.py first, then update BUNDLE_PATH if needed.")
        return

    bundle = joblib.load(BUNDLE_PATH)

    print("NBA Next-Season Prediction Tool")
    print("Type a player name, or type 'quit' to exit.\n")

    while True:
        user_input = input("Player name: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        predict_for_player(user_input, bundle)


if __name__ == "__main__":
    main()
