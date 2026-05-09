import os
import warnings
from difflib import get_close_matches

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# FINAL PRESENTATION: ADVANCED MODELS, INSIGHTS, INTERPRETATION
# ============================================================
# What this script does:
# 1) Builds baseline + advanced models for next-season PTS.
# 2) Compares models side-by-side using test and CV metrics.
# 3) Creates explainability outputs:
#       - best linear coefficients
#       - permutation importance for best advanced model
#       - partial dependence plots
# 4) Builds optional player archetypes using clustering.
# 5) Trains and saves models for a terminal prediction tool.
#
# Update DATA_PATH to your file location before running.
# ============================================================

DATA_PATH = r"C:\Users\victo\Desktop\GIT\MATH494\presentation1_outputs\all_seasons_cleaned_for_modeling.csv"
OUTPUT_DIR = r"C:\Users\victo\Desktop\GIT\MATH494\final_presentation_outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MAIN_TARGET = "next_pts"
ALL_PREDICTION_TARGETS = ["next_pts", "next_reb", "next_ast", "next_ts_pct"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------
# DATA PREPARATION HELPERS
# ------------------------------------------------------------
def parse_season_start_year(season_str):
    if pd.isna(season_str):
        return np.nan
    try:
        return int(str(season_str).split("-")[0])
    except Exception:
        return np.nan



def add_position_proxy(df: pd.DataFrame) -> pd.DataFrame:
    if "pos_proxy" in df.columns:
        return df
    if "player_height" not in df.columns:
        df["pos_proxy"] = "Unknown"
        return df

    df = df.copy()
    df["pos_proxy"] = pd.cut(
        df["player_height"],
        bins=[-np.inf, 195, 205, np.inf],
        labels=["Guard (proxy)", "Wing (proxy)", "Big (proxy)"],
    ).astype(str)
    return df



def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_name", "season_start"]).copy()
    lag_cols = ["pts", "reb", "ast", "ts_pct", "usg_pct", "net_rating"]
    for col in lag_cols:
        lag_name = f"{col}_lag1"
        if col in df.columns and lag_name not in df.columns:
            df[lag_name] = df.groupby("player_name")[col].shift(1)
    if "pts" in df.columns and "pts_lag1" in df.columns and "d_pts" not in df.columns:
        df["d_pts"] = df["pts"] - df["pts_lag1"]
    if "ts_pct" in df.columns and "ts_pct_lag1" in df.columns and "d_ts" not in df.columns:
        df["d_ts"] = df["ts_pct"] - df["ts_pct_lag1"]
    return df



def add_next_season_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_name", "season_start"]).copy()
    targets = ["pts", "reb", "ast", "ts_pct"]
    for stat in targets:
        next_col = f"next_{stat}"
        if stat in df.columns and next_col not in df.columns:
            df[next_col] = df.groupby("player_name")[stat].shift(-1)
    if "next_pts" in df.columns and "pts" in df.columns and "improved_next_pts" not in df.columns:
        df["improved_next_pts"] = (df["next_pts"] > df["pts"]).astype(int)
    return df



def ensure_prepared_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    for col in ["Unnamed: 0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_candidates = [
        "age", "player_height", "player_weight", "gp", "pts", "reb", "ast",
        "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct",
        "pts_lag1", "reb_lag1", "ast_lag1", "ts_pct_lag1", "usg_pct_lag1", "net_rating_lag1",
        "next_pts", "next_reb", "next_ast", "next_ts_pct",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "season_start" not in df.columns and "season" in df.columns:
        df["season_start"] = df["season"].apply(parse_season_start_year)

    df = add_position_proxy(df)
    df = add_lag_features(df)
    df = add_next_season_targets(df)
    return df



def get_feature_lists(df: pd.DataFrame):
    numeric_features = [
        "age",
        "player_height",
        "player_weight",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
        "pts_lag1",
        "reb_lag1",
        "ast_lag1",
        "ts_pct_lag1",
        "usg_pct_lag1",
        "net_rating_lag1",
    ]

    categorical_features = [
        "team_abbreviation",
        "college",
        "country",
        "draft_year",
        "draft_round",
        "draft_number",
        "season",
        "pos_proxy",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features



def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
    )



def build_model_factories(preprocessor):
    ridge_alphas = np.logspace(-3, 3, 50)
    lasso_alphas = np.logspace(-3, 1, 50)

    return {
        "Linear Regression": lambda: Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", LinearRegression())]
        ),
        "Ridge Regression": lambda: Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", RidgeCV(alphas=ridge_alphas, cv=5))]
        ),
        "LASSO Regression": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", LassoCV(alphas=lasso_alphas, cv=5, random_state=RANDOM_STATE, max_iter=20000)),
            ]
        ),
        "Random Forest": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": lambda: Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------
def evaluate_models_for_target(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    numeric_features, categorical_features = get_feature_lists(work)
    all_features = numeric_features + categorical_features

    X = work[all_features]
    y = work[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    factories = build_model_factories(preprocessor)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}

    for name, factory in factories.items():
        print(f"Training {name} for {target}...")
        model = factory()
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        cv_rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
        cv_r2 = cross_val_score(model, X, y, scoring="r2", cv=cv)

        results.append(
            {
                "Model": name,
                "Target": target,
                "Train_RMSE": train_rmse,
                "Test_RMSE": test_rmse,
                "Test_MAE": test_mae,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "CV_RMSE_Mean": cv_rmse.mean(),
                "CV_RMSE_SD": cv_rmse.std(),
                "CV_R2_Mean": cv_r2.mean(),
                "CV_R2_SD": cv_r2.std(),
                "Overfit_Gap_RMSE": test_rmse - train_rmse,
                "Overfit_Gap_R2": train_r2 - test_r2,
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(["CV_RMSE_Mean", "Test_RMSE"])
    return results_df, fitted_models, X_train, X_test, y_train, y_test, all_features, numeric_features, categorical_features


# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------
def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: str):
    order = df.sort_values(y_col, ascending=True if "RMSE" in y_col or "MAE" in y_col or "Gap" in y_col else False)
    plt.figure(figsize=(10, 6))
    plt.bar(order[x_col], order[y_col])
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def save_predicted_vs_actual(model, X_test, y_test, title: str, out_path: str):
    preds = model.predict(X_test)
    line_min = min(np.min(y_test), np.min(preds))
    line_max = max(np.max(y_test), np.max(preds))
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.45)
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# EXPLAINABILITY OUTPUTS
# ------------------------------------------------------------
def save_linear_coefficients(best_linear_model, out_dir: str):
    preprocessor = best_linear_model.named_steps["preprocessor"]
    estimator = best_linear_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.ravel(estimator.coef_)

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)
    coef_df.to_csv(os.path.join(out_dir, "best_linear_coefficients.csv"), index=False)

    plot_df = coef_df.head(20).sort_values("Coefficient")
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["Feature"], plot_df["Coefficient"])
    plt.title("Top Linear Model Coefficients (by absolute magnitude)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_linear_coefficients.png"), dpi=300)
    plt.close()

    return coef_df



def save_permutation_importance(best_advanced_model, X_test, y_test, out_dir: str):
    result = permutation_importance(
        best_advanced_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_SD": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    importance_df.to_csv(os.path.join(out_dir, "best_advanced_permutation_importance.csv"), index=False)

    plot_df = importance_df.head(15).sort_values("Importance_Mean")
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["Feature"], plot_df["Importance_Mean"])
    plt.title("Permutation Importance - Best Advanced Model")
    plt.xlabel("Mean decrease in model performance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_advanced_permutation_importance.png"), dpi=300)
    plt.close()

    return importance_df



def save_partial_dependence_plots(best_advanced_model, X_test, preferred_features, out_dir: str):
    selected = []
    for feature in preferred_features:
        if feature in X_test.columns and feature not in selected:
            selected.append(feature)
        if len(selected) == 3:
            break

    for feature in selected:
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(best_advanced_model, X_test, [feature], ax=ax)
        ax.set_title(f"Partial Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pdp_{feature}.png"), dpi=300)
        plt.close()

    return selected


# ------------------------------------------------------------
# PLAYER ARCHETYPES (OPTIONAL BUT HIGH IMPACT)
# ------------------------------------------------------------
def build_player_archetypes(df: pd.DataFrame, out_dir: str, n_clusters: int = 4):
    player_level = (
        df.groupby("player_name")
        .agg(
            seasons=("season", "nunique"),
            gp_total=("gp", "sum"),
            player_height=("player_height", "median"),
            player_weight=("player_weight", "median"),
            pts=("pts", "mean"),
            reb=("reb", "mean"),
            ast=("ast", "mean"),
            ts_pct=("ts_pct", "mean"),
            usg_pct=("usg_pct", "mean"),
            net_rating=("net_rating", "mean"),
        )
        .reset_index()
    )

    cluster_features = [
        "player_height",
        "player_weight",
        "pts",
        "reb",
        "ast",
        "ts_pct",
        "usg_pct",
        "net_rating",
    ]
    cluster_features = [c for c in cluster_features if c in player_level.columns]

    cluster_work = player_level.copy()
    cluster_work[cluster_features] = cluster_work[cluster_features].apply(pd.to_numeric, errors="coerce")
    cluster_work = cluster_work.dropna(subset=cluster_features).copy()

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(cluster_work[cluster_features])

    # Optional support for explaining why k=4 was used
    k_search = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        k_search.append({"k": k, "silhouette": score})
    k_search_df = pd.DataFrame(k_search)
    k_search_df.to_csv(os.path.join(out_dir, "cluster_silhouette_scores.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(k_search_df["k"], k_search_df["silhouette"], marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_silhouette_scores.png"), dpi=300)
    plt.close()

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    cluster_work["cluster"] = kmeans.fit_predict(X_cluster)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_cluster)
    cluster_work["pca1"] = coords[:, 0]
    cluster_work["pca2"] = coords[:, 1]

    plt.figure(figsize=(9, 7))
    for cluster_id in sorted(cluster_work["cluster"].unique()):
        sub = cluster_work[cluster_work["cluster"] == cluster_id]
        plt.scatter(sub["pca1"], sub["pca2"], alpha=0.6, label=f"Cluster {cluster_id}")
    plt.title("Player Archetypes (KMeans + PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "player_archetypes_pca.png"), dpi=300)
    plt.close()

    profiles = cluster_work.groupby("cluster")[cluster_features].mean().round(3)
    profiles.to_csv(os.path.join(out_dir, "player_archetype_profiles.csv"))

    # Save example players from each cluster
    examples = (
        cluster_work.sort_values(["cluster", "seasons", "gp_total"], ascending=[True, False, False])
        .groupby("cluster")
        .head(8)
        [["cluster", "player_name", "seasons", "gp_total", "pts", "reb", "ast"]]
    )
    examples.to_csv(os.path.join(out_dir, "player_archetype_examples.csv"), index=False)

    # Age curves by cluster: merge stable player cluster onto season-level rows
    merged = df.merge(cluster_work[["player_name", "cluster"]], on="player_name", how="inner")
    merged = merged.dropna(subset=["age", "pts"]).copy()
    merged["age_int"] = merged["age"].round().astype(int)

    plt.figure(figsize=(10, 6))
    for cluster_id, sub in merged.groupby("cluster"):
        curve = sub.groupby("age_int", as_index=False)["pts"].mean().sort_values("age_int")
        curve["pts_smooth"] = curve["pts"].rolling(3, center=True, min_periods=1).mean()
        plt.plot(curve["age_int"], curve["pts_smooth"], label=f"Cluster {cluster_id}")
    plt.title("Age Curves by Player Archetype")
    plt.xlabel("Age")
    plt.ylabel("Smoothed Points Per Game")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "player_archetype_age_curves.png"), dpi=300)
    plt.close()

    return cluster_work, profiles


# ------------------------------------------------------------
# PREDICTION TOOL ARTIFACTS
# ------------------------------------------------------------
def fit_best_model_for_target(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    numeric_features, categorical_features = get_feature_lists(work)
    all_features = numeric_features + categorical_features
    X = work[all_features]
    y = work[target]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    factories = build_model_factories(preprocessor)
    keep = ["Ridge Regression", "Random Forest", "Gradient Boosting"]

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_name = None
    best_score = np.inf
    best_model = None

    for name in keep:
        model = factories[name]()
        score = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv).mean()
        if score < best_score:
            best_score = score
            best_name = name
            best_model = model

    best_model.fit(X, y)
    return {
        "model": best_model,
        "model_name": best_name,
        "cv_rmse": best_score,
        "features": all_features,
    }



def save_prediction_bundle(df: pd.DataFrame):
    latest_rows = (
        df.sort_values(["player_name", "season_start"])
        .groupby("player_name")
        .tail(1)
        .copy()
    )

    model_bundle = {
        "targets": {},
        "latest_rows": latest_rows,
        "random_state": RANDOM_STATE,
    }

    summary_rows = []
    for target in ALL_PREDICTION_TARGETS:
        print(f"Building saved prediction model for {target}...")
        bundle = fit_best_model_for_target(df, target)
        model_bundle["targets"][target] = bundle
        summary_rows.append(
            {
                "Target": target,
                "Chosen_Model": bundle["model_name"],
                "CV_RMSE": bundle["cv_rmse"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Target")
    summary_df.to_csv(os.path.join(MODELS_DIR, "prediction_model_summary.csv"), index=False)
    latest_rows.to_csv(os.path.join(MODELS_DIR, "latest_player_rows.csv"), index=False)
    joblib.dump(model_bundle, os.path.join(MODELS_DIR, "player_prediction_bundle.joblib"))


# ------------------------------------------------------------
# AUTO-WRITTEN TALKING POINTS
# ------------------------------------------------------------
def write_summary_notes(results_df: pd.DataFrame, best_linear_name: str, best_advanced_name: str, best_improvements: dict):
    best_row = results_df.iloc[0]
    baseline_row = results_df[results_df["Model"] == "Linear Regression"].iloc[0]
    adv_row = results_df[results_df["Model"] == best_advanced_name].iloc[0]

    rmse_gain = baseline_row["CV_RMSE_Mean"] - adv_row["CV_RMSE_Mean"]
    r2_gain = adv_row["CV_R2_Mean"] - baseline_row["CV_R2_Mean"]

    lines = []
    lines.append("FINAL PRESENTATION TALKING POINTS")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Best overall model for {MAIN_TARGET}: {best_row['Model']}")
    lines.append(f"Best linear model: {best_linear_name}")
    lines.append(f"Best advanced model: {best_advanced_name}")
    lines.append("")
    lines.append("Model-comparison interpretation")
    lines.append(f"- Best advanced model CV RMSE improvement over plain linear regression: {rmse_gain:.3f}")
    lines.append(f"- Best advanced model CV R^2 improvement over plain linear regression: {r2_gain:.3f}")
    lines.append(f"- Best advanced model overfit gap (RMSE): {adv_row['Overfit_Gap_RMSE']:.3f}")
    lines.append("")
    lines.append("How to explain performance gains")
    if rmse_gain > 0.25:
        lines.append("- Advanced models provide a meaningful reduction in prediction error, which suggests nonlinear interactions matter.")
    elif rmse_gain > 0.05:
        lines.append("- Advanced models improve prediction modestly. The gain is real, but interpretability tradeoffs should be discussed.")
    else:
        lines.append("- Advanced models only slightly improve prediction. That suggests linear structure already captures much of the signal.")
    lines.append("")
    lines.append("Explainability reminders")
    lines.append("- Use linear coefficients to explain direction and rough relative importance.")
    lines.append("- Use permutation importance to show which features matter most for the best advanced model.")
    lines.append("- Use partial dependence plots to show how predicted next-season scoring changes as one feature varies.")
    lines.append("")
    lines.append("Ethics and limitations")
    lines.append("- The model does not observe injuries, locker-room context, coaching changes, or role changes.")
    lines.append("- Survivorship bias means older players in the dataset are often the better ones who stayed in the league.")
    lines.append("- These models should support decisions, not replace human judgment in contracts or roster moves.")
    lines.append("")
    lines.append("Saved prediction models")
    for target, info in best_improvements.items():
        lines.append(f"- {target}: {info['model_name']} (CV RMSE {info['cv_rmse']:.3f})")

    with open(os.path.join(OUTPUT_DIR, "final_presentation_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("Loading and preparing data...")
    df = ensure_prepared_dataset(DATA_PATH)

    print("\nEvaluating final-presentation models for next-season points...")
    results_df, fitted_models, X_train, X_test, y_train, y_test, all_features, numeric_features, categorical_features = evaluate_models_for_target(df, MAIN_TARGET)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "final_model_comparison.csv"), index=False)

    # Side-by-side comparison outputs
    save_bar_plot(results_df, "Model", "Test_RMSE", "Test RMSE by Model", "Test RMSE", os.path.join(OUTPUT_DIR, "test_rmse_comparison.png"))
    save_bar_plot(results_df, "Model", "Test_R2", "Test R^2 by Model", "Test R^2", os.path.join(OUTPUT_DIR, "test_r2_comparison.png"))
    save_bar_plot(results_df, "Model", "CV_RMSE_Mean", "Cross-Validated RMSE by Model", "Mean CV RMSE", os.path.join(OUTPUT_DIR, "cv_rmse_comparison_final.png"))
    save_bar_plot(results_df, "Model", "CV_R2_Mean", "Cross-Validated R^2 by Model", "Mean CV R^2", os.path.join(OUTPUT_DIR, "cv_r2_comparison_final.png"))
    save_bar_plot(results_df, "Model", "Overfit_Gap_RMSE", "Overfitting Check: Test RMSE - Train RMSE", "RMSE Gap", os.path.join(OUTPUT_DIR, "overfit_gap_rmse.png"))

    linear_candidates = ["Linear Regression", "Ridge Regression", "LASSO Regression"]
    advanced_candidates = ["Random Forest", "Gradient Boosting"]

    best_linear_name = results_df[results_df["Model"].isin(linear_candidates)].iloc[0]["Model"]
    best_advanced_name = results_df[results_df["Model"].isin(advanced_candidates)].iloc[0]["Model"]

    best_linear_model = fitted_models[best_linear_name]
    best_advanced_model = fitted_models[best_advanced_name]

    save_predicted_vs_actual(
        best_linear_model,
        X_test,
        y_test,
        f"{best_linear_name}: Predicted vs Actual Next-Season PTS",
        os.path.join(OUTPUT_DIR, "best_linear_predicted_vs_actual.png"),
    )
    save_predicted_vs_actual(
        best_advanced_model,
        X_test,
        y_test,
        f"{best_advanced_name}: Predicted vs Actual Next-Season PTS",
        os.path.join(OUTPUT_DIR, "best_advanced_predicted_vs_actual.png"),
    )

    coef_df = save_linear_coefficients(best_linear_model, OUTPUT_DIR)
    importance_df = save_permutation_importance(best_advanced_model, X_test, y_test, OUTPUT_DIR)

    preferred_features = []
    for feat in importance_df["Feature"].tolist():
        if feat in numeric_features:
            preferred_features.append(feat)
        if len(preferred_features) == 3:
            break
    if len(preferred_features) < 3:
        for fallback in ["age", "pts", "usg_pct", "ts_pct", "ast_pct", "net_rating"]:
            if fallback in numeric_features and fallback not in preferred_features:
                preferred_features.append(fallback)
            if len(preferred_features) == 3:
                break
    save_partial_dependence_plots(best_advanced_model, X_test, preferred_features, OUTPUT_DIR)

    print("\nBuilding player archetypes...")
    cluster_work, profiles = build_player_archetypes(df, OUTPUT_DIR, n_clusters=4)

    print("\nSaving prediction-tool models...")
    save_prediction_bundle(df)

    prediction_summary = pd.read_csv(os.path.join(MODELS_DIR, "prediction_model_summary.csv"))
    prediction_map = {
        row["Target"]: {"model_name": row["Chosen_Model"], "cv_rmse": row["CV_RMSE"]}
        for _, row in prediction_summary.iterrows()
    }
    write_summary_notes(results_df, best_linear_name, best_advanced_name, prediction_map)

    print("\nDone.")
    print(f"All final-presentation outputs saved to: {OUTPUT_DIR}")
    print(f"Prediction tool bundle saved to: {os.path.join(MODELS_DIR, 'player_prediction_bundle.joblib')}")


if __name__ == "__main__":
    main()
