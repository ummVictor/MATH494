

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) CONFIG
# =========================
DATA_PATH = "C:\\Users\\victo\\Desktop\\GIT\\MATH494\\all_seasons.csv"  # <-- your path
OUTPUT_DIR = Path("presentation1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Change if you want higher-res images
FIG_DPI = 200

# Minimum games filter (rough proxy for "extremely low-minute players")
MIN_GAMES = 10

# Variables we’ll use often
CORE_NUMERIC_COLS = [
    "age", "gp", "pts", "reb", "ast",
    "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"
]


# =========================
# 1) LOAD + BASIC CLEANING
# =========================
def parse_season_start_year(season_str: str) -> int:
    """
    Convert "1996-97" -> 1996 (start year).
    """
    if pd.isna(season_str):
        return np.nan
    s = str(season_str)
    # Most common format in this dataset: "1996-97"
    try:
        return int(s.split("-")[0])
    except Exception:
        return np.nan


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop common index column if present
    for col in ["Unnamed: 0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Standardize column names (optional; dataset already good)
    # df.columns = [c.strip().lower() for c in df.columns]

    # Parse season start year
    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in the dataset.")
    df["season_start"] = df["season"].apply(parse_season_start_year).astype("Int64")

    # Ensure numeric columns are numeric
    for col in CORE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_position_proxy_from_height(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset doesn't include true 'position'. For Presentation 1, we create a
    simple proxy from height (cm) for faceted plots:
      - Guard:  < 195 cm
      - Wing:   195–205 cm
      - Big:    > 205 cm
    """
    if "player_height" not in df.columns:
        df["pos_proxy"] = "Unknown"
        return df

    h = df["player_height"]
    df["pos_proxy"] = pd.cut(
        h,
        bins=[-np.inf, 195, 205, np.inf],
        labels=["Guard (proxy)", "Wing (proxy)", "Big (proxy)"]
    ).astype(str)
    return df


def basic_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roughly remove "extremely low-minute players" using games played (gp)
    since minutes aren't in this dataset.
    """
    if "gp" in df.columns:
        df = df[df["gp"].fillna(0) >= MIN_GAMES].copy()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag (previous-season) features per player for later modeling.
    This supports Slide 5 talking point (lag variables).
    """
    df = df.sort_values(["player_name", "season_start"]).copy()

    lag_cols = ["pts", "reb", "ast", "ts_pct", "usg_pct", "net_rating"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("player_name")[col].shift(1)

    # Year-to-year improvement metrics
    if "pts" in df.columns:
        df["d_pts"] = df["pts"] - df["pts_lag1"]
    if "ts_pct" in df.columns:
        df["d_ts"] = df["ts_pct"] - df["ts_pct_lag1"]

    return df


# =========================
# 2) SLIDE 4 SUMMARY STATS
# =========================
def print_dataset_overview(df: pd.DataFrame) -> None:
    season_min = int(df["season_start"].min())
    season_max = int(df["season_start"].max())
    n_rows = df.shape[0]
    n_players = df["player_name"].nunique()
    n_seasons = df["season"].nunique()

    print("\n=== Slide 4: Dataset Overview ===")
    print(f"Time span: {season_min} to {season_max} (season start years)")
    print(f"Rows (player-seasons): {n_rows:,}")
    print(f"Unique players: {n_players:,}")
    print(f"Unique seasons: {n_seasons:,}")
    print("\nColumns:", list(df.columns))


# =========================
# 3) PLOTTING HELPERS
# =========================
def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_preprocess_flowchart_placeholder() -> None:
    """
    Optional slide visual: simple flow diagram (not a true flowchart tool),
    but a clean "pipeline" graphic you can include.
    """
    steps = [
        "Load CSV",
        "Drop index cols",
        f"Filter gp >= {MIN_GAMES}",
        "Create season_start",
        "Create pos_proxy (height)",
        "Create lag features",
        "EDA plots"
    ]

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.axis("off")

    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.5

    for i, (x, step) in enumerate(zip(x_positions, steps)):
        ax.text(x, y, step, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35"))
        if i < len(steps) - 1:
            ax.annotate("",
                        xy=(x_positions[i+1]-0.06, y),
                        xytext=(x+0.06, y),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Preprocessing Pipeline (Presentation 1)", fontsize=13)
    savefig(OUTPUT_DIR / "slide5_preprocessing_pipeline.png")


# =========================
# 4) SLIDE 6: AGE CURVES
# =========================
def plot_age_curve_pts(df: pd.DataFrame) -> None:
    """
    Smoothed (rolling mean) curve of scoring vs age.
    Also includes curves by pos_proxy to mimic "by position".
    """
    d = df.dropna(subset=["age", "pts"]).copy()
    d["age_int"] = d["age"].round().astype(int)

    # Overall mean by age
    overall = d.groupby("age_int", as_index=False)["pts"].mean().sort_values("age_int")

    plt.figure(figsize=(10, 5))
    plt.plot(overall["age_int"], overall["pts"], label="Overall (mean PTS)")

    # Rolling mean to smooth (window=3)
    overall_smooth = overall.copy()
    overall_smooth["pts_smooth"] = overall_smooth["pts"].rolling(3, center=True, min_periods=1).mean()
    plt.plot(overall_smooth["age_int"], overall_smooth["pts_smooth"], linewidth=2, label="Overall (smoothed)")

    plt.title("Performance vs Age (Points per Game)")
    plt.xlabel("Age")
    plt.ylabel("Points per Game (PTS)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide6_age_curve_pts_overall.png")

    # Faceted-style by proxy position (same plot, multiple lines)
    if "pos_proxy" in d.columns:
        plt.figure(figsize=(10, 5))
        for group, sub in d.groupby("pos_proxy"):
            g = sub.groupby("age_int", as_index=False)["pts"].mean().sort_values("age_int")
            g["pts_smooth"] = g["pts"].rolling(3, center=True, min_periods=1).mean()
            plt.plot(g["age_int"], g["pts_smooth"], label=str(group))

        plt.title("Performance vs Age (Smoothed PTS) by Position Proxy (Height)")
        plt.xlabel("Age")
        plt.ylabel("Smoothed Points per Game")
        plt.legend()
        plt.grid(True, alpha=0.25)
        savefig(OUTPUT_DIR / "slide6_age_curve_pts_by_pos_proxy.png")


# =========================
# 5) SLIDE 7: CORRELATION HEATMAP
# =========================
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation heatmap for numeric variables.
    Uses matplotlib only (no seaborn) per tool guidance.
    """
    cols = [c for c in CORE_NUMERIC_COLS if c in df.columns]
    d = df[cols].dropna().copy()

    corr = d.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap (Key Numeric Variables)")
    savefig(OUTPUT_DIR / "slide7_correlation_heatmap.png")


# =========================
# 6) SLIDE 8: IMPROVEMENT TRAJECTORIES
# =========================
def plot_improvement_distribution(df: pd.DataFrame) -> None:
    """
    Distribution of year-to-year change in points (d_pts).
    """
    if "d_pts" not in df.columns:
        return

    d = df.dropna(subset=["d_pts"]).copy()
    plt.figure(figsize=(10, 5))
    plt.hist(d["d_pts"], bins=50)
    plt.title("Year-to-Year Change in Scoring (ΔPTS = PTS - PTS_lag1)")
    plt.xlabel("ΔPTS (points per game)")
    plt.ylabel("Count of player-seasons")
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide8_improvement_hist_dpts.png")


def plot_age_vs_improvement(df: pd.DataFrame) -> None:
    """
    Scatter plot: age vs year-to-year improvement in points (d_pts).
    """
    if "d_pts" not in df.columns:
        return

    d = df.dropna(subset=["age", "d_pts"]).copy()
    # Light downsample for readability if huge
    if len(d) > 5000:
        d = d.sample(5000, random_state=7)

    plt.figure(figsize=(10, 5))
    plt.scatter(d["age"], d["d_pts"], s=10, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.title("Age vs Year-to-Year Improvement (ΔPTS)")
    plt.xlabel("Age")
    plt.ylabel("ΔPTS (points per game)")
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide8_age_vs_improvement_scatter.png")


def plot_example_player_trajectories(df: pd.DataFrame, players: list[str] | None = None) -> None:
    """
    Plot a few player scoring trajectories across seasons.
    If players not provided, picks 3 well-known names IF they exist.
    """
    if players is None:
        candidates = ["LeBron James", "Stephen Curry", "Kevin Durant", "Kobe Bryant", "Tim Duncan"]
        existing = [p for p in candidates if p in set(df["player_name"])]
        players = existing[:3] if len(existing) >= 3 else list(df["player_name"].value_counts().head(3).index)

    d = df[df["player_name"].isin(players)].dropna(subset=["season_start", "pts"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(10, 5))
    for p, sub in d.groupby("player_name"):
        sub = sub.sort_values("season_start")
        plt.plot(sub["season_start"], sub["pts"], marker="o", label=p)

    plt.title("Example Player Scoring Trajectories (PTS by Season)")
    plt.xlabel("Season Start Year")
    plt.ylabel("Points per Game")
    plt.legend()
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide8_example_player_trajectories.png")


# =========================
# 7) SLIDE 9: ERA SHIFTS
# =========================
def plot_league_pts_over_time(df: pd.DataFrame) -> None:
    """
    League average points per game over time.
    """
    d = df.dropna(subset=["season_start", "pts"]).copy()
    by_year = d.groupby("season_start", as_index=False)["pts"].mean().sort_values("season_start")

    plt.figure(figsize=(10, 5))
    plt.plot(by_year["season_start"], by_year["pts"], marker="o")
    plt.title("League Average Points per Game Over Time")
    plt.xlabel("Season Start Year")
    plt.ylabel("Average PTS (player-season mean)")
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide9_league_pts_over_time.png")


def plot_league_ts_over_time(df: pd.DataFrame) -> None:
    """
    League average true shooting percentage over time (if available).
    """
    if "ts_pct" not in df.columns:
        return

    d = df.dropna(subset=["season_start", "ts_pct"]).copy()
    by_year = d.groupby("season_start", as_index=False)["ts_pct"].mean().sort_values("season_start")

    plt.figure(figsize=(10, 5))
    plt.plot(by_year["season_start"], by_year["ts_pct"], marker="o")
    plt.title("League Average True Shooting % Over Time")
    plt.xlabel("Season Start Year")
    plt.ylabel("Average TS%")
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide9_league_ts_over_time.png")


# =========================
# 8) SLIDE 2 MOTIVATION VISUAL (OPTIONAL)
# =========================
def plot_motivation_visual_proxy(df: pd.DataFrame) -> None:
    """
    Your dataset does NOT contain salary/contract values. For Slide 2’s visual,
    you have two options:
      A) Add a salary dataset later (recommended)
      B) Use a performance distribution as a "value proxy" visual for now

    This produces (B): distribution of Points per Game as a simple proxy.
    """
    d = df.dropna(subset=["pts"]).copy()
    plt.figure(figsize=(10, 5))
    plt.hist(d["pts"], bins=40)
    plt.title("Motivation Visual (Proxy): Distribution of Player Scoring (PTS)")
    plt.xlabel("Points per Game (PTS)")
    plt.ylabel("Count of player-seasons")
    plt.grid(True, alpha=0.25)
    savefig(OUTPUT_DIR / "slide2_motivation_proxy_pts_distribution.png")


# =========================
# 9) MAIN
# =========================
def main() -> None:
    df = load_data(DATA_PATH)
    df = add_position_proxy_from_height(df)
    df = basic_filtering(df)
    df = add_lag_features(df)

    # Print Slide 4 stats
    print_dataset_overview(df)

    # Slide 2 visual (proxy)
    plot_motivation_visual_proxy(df)

    # Slide 5 optional pipeline graphic
    plot_preprocess_flowchart_placeholder()

    # Slide 6: age curves
    plot_age_curve_pts(df)

    # Slide 7: correlation heatmap
    plot_correlation_heatmap(df)

    # Slide 8: improvement patterns
    plot_improvement_distribution(df)
    plot_age_vs_improvement(df)
    plot_example_player_trajectories(df)

    # Slide 9: era shifts
    plot_league_pts_over_time(df)
    plot_league_ts_over_time(df)

    # Save a cleaned dataset for later presentations (Presentation 2+)
    cleaned_path = OUTPUT_DIR / "all_seasons_cleaned_for_modeling.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned dataset for next steps: {cleaned_path.resolve()}")

    print(f"\nAll figures saved to: {OUTPUT_DIR.resolve()}")
    print("You can drag the PNG files directly into your Presentation 1 slides.")


if __name__ == "__main__":
    main()
    
    
    