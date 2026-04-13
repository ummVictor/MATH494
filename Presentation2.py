# ============================================
# PRESENTATION 2: BASELINE & CLASSICAL MODELING
# NBA Player Performance / Next-Season Prediction
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# -----------------------------
# 1. SETTINGS
# -----------------------------
DATA_PATH = r"C:\Users\victo\Desktop\GIT\MATH494\presentation1_outputs\all_seasons_cleaned_for_modeling.csv"
OUTPUT_DIR = r"C:\Users\victo\Desktop\GIT\MATH494\presentation2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# 3. CREATE NEXT-SEASON OUTCOMES
# -----------------------------
df = df.sort_values(["player_name", "season_start"]).reset_index(drop=True)

df["next_pts"] = df.groupby("player_name")["pts"].shift(-1)
df["next_reb"] = df.groupby("player_name")["reb"].shift(-1)
df["next_ast"] = df.groupby("player_name")["ast"].shift(-1)
df["next_ts_pct"] = df.groupby("player_name")["ts_pct"].shift(-1)

df["improved_next_pts"] = (df["next_pts"] > df["pts"]).astype(int)

model_df = df.dropna(subset=["next_pts"]).copy()

# -----------------------------
# 4. FEATURE SELECTION
# -----------------------------
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

numeric_features = [c for c in numeric_features if c in model_df.columns]
categorical_features = [c for c in categorical_features if c in model_df.columns]
all_features = numeric_features + categorical_features

print("\nNumeric features used:")
print(numeric_features)

print("\nCategorical features used:")
print(categorical_features)

# -----------------------------
# 5. HELPER FUNCTION FOR STATSMODELS
# -----------------------------
def make_statsmodels_matrix(data, target_col, numeric_cols, categorical_cols):
    """
    Build a fully numeric matrix for statsmodels.
    Handles:
    - missing values
    - categorical dummy encoding
    - bool -> int conversion
    - object -> numeric coercion
    """
    use_cols = [target_col] + numeric_cols + categorical_cols
    temp = data[use_cols].copy()

    # Fill numeric
    for col in numeric_cols:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            temp[col] = temp[col].fillna(temp[col].median())

    # Fill categorical
    for col in categorical_cols:
        if col in temp.columns:
            temp[col] = temp[col].astype(str).fillna("Missing")

    # Dummy encode categorical variables
    temp = pd.get_dummies(temp, columns=categorical_cols, drop_first=True)

    # Convert bool columns to int
    bool_cols = temp.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        temp[bool_cols] = temp[bool_cols].astype(int)

    # Force everything to numeric
    for col in temp.columns:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    # Drop rows with any remaining missing values
    temp = temp.dropna().copy()

    y = temp[target_col].astype(float)
    X = temp.drop(columns=[target_col]).astype(float)
    X = sm.add_constant(X, has_constant="add")

    return X, y, temp

# -----------------------------
# 6. REGRESSION DATASET
# -----------------------------
reg_df = model_df[all_features + ["next_pts"]].copy()

X_reg = reg_df[all_features]
y_reg = reg_df["next_pts"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# -----------------------------
# 7. PREPROCESSING PIPELINE
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 8. REGRESSION MODELS
# -----------------------------
linreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

ridge_alphas = np.logspace(-3, 3, 50)
lasso_alphas = np.logspace(-3, 1, 50)

ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RidgeCV(alphas=ridge_alphas, cv=5))
])

lasso_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LassoCV(alphas=lasso_alphas, cv=5, random_state=RANDOM_STATE, max_iter=20000))
])

regression_models = {
    "Linear Regression": linreg_pipeline,
    "Ridge Regression": ridge_pipeline,
    "LASSO Regression": lasso_pipeline
}

# -----------------------------
# 9. FIT + EVALUATE REGRESSION MODELS
# -----------------------------
regression_results = []

for name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    preds = model.predict(X_test_reg)

    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    mae = mean_absolute_error(y_test_reg, preds)
    r2 = r2_score(y_test_reg, preds)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse = -cross_val_score(
        model, X_reg, y_reg,
        scoring="neg_root_mean_squared_error",
        cv=cv
    )
    cv_r2 = cross_val_score(
        model, X_reg, y_reg,
        scoring="r2",
        cv=cv
    )

    regression_results.append({
        "Model": name,
        "Test_RMSE": rmse,
        "Test_MAE": mae,
        "Test_R2": r2,
        "CV_RMSE_Mean": cv_rmse.mean(),
        "CV_RMSE_SD": cv_rmse.std(),
        "CV_R2_Mean": cv_r2.mean(),
        "CV_R2_SD": cv_r2.std()
    })

regression_results_df = pd.DataFrame(regression_results).sort_values("Test_RMSE")
print("\n=== REGRESSION RESULTS ===")
print(regression_results_df)
regression_results_df.to_csv(os.path.join(OUTPUT_DIR, "regression_model_results.csv"), index=False)

# -----------------------------
# 10. OLS MODEL FOR INTERPRETATION + DIAGNOSTICS
# -----------------------------
ols_numeric = [
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
    "net_rating_lag1"
]
ols_numeric = [c for c in ols_numeric if c in model_df.columns]

ols_categorical = ["pos_proxy"]
ols_categorical = [c for c in ols_categorical if c in model_df.columns]

X_ols, y_ols, ols_temp = make_statsmodels_matrix(
    data=model_df,
    target_col="next_pts",
    numeric_cols=ols_numeric,
    categorical_cols=ols_categorical
)

print("\nOLS design matrix shape:", X_ols.shape)
print("OLS dtypes summary:")
print(X_ols.dtypes.value_counts())

ols_model = sm.OLS(y_ols, X_ols).fit()

print("\n=== OLS SUMMARY ===")
print(ols_model.summary())

with open(os.path.join(OUTPUT_DIR, "ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(ols_model.summary().as_text())

# -----------------------------
# 11. VIF (MULTICOLLINEARITY CHECK)
# -----------------------------
vif_df = pd.DataFrame()
vif_df["Feature"] = X_ols.columns
vif_df["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]
vif_df = vif_df.sort_values("VIF", ascending=False)

print("\n=== VIF TABLE ===")
print(vif_df.head(20))
vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif_table.csv"), index=False)

# -----------------------------
# 12. REGRESSION FIGURES
# -----------------------------
linreg_pipeline.fit(X_train_reg, y_train_reg)
linreg_preds = linreg_pipeline.predict(X_test_reg)
residuals = y_test_reg - linreg_preds

# Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=linreg_preds, alpha=0.6)
line_min = min(y_test_reg.min(), linreg_preds.min())
line_max = max(y_test_reg.max(), linreg_preds.max())
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
plt.xlabel("Actual Next-Season PTS")
plt.ylabel("Predicted Next-Season PTS")
plt.title("Linear Regression: Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual_linear.png"), dpi=300)
plt.close()

# Residuals vs Fitted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=linreg_preds, y=residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_fitted.png"), dpi=300)
plt.close()

# Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residual")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_distribution.png"), dpi=300)
plt.close()

# Q-Q Plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "qq_plot_residuals.png"), dpi=300)
plt.close()

# CV RMSE comparison
plt.figure(figsize=(9, 6))
sns.barplot(data=regression_results_df, x="Model", y="CV_RMSE_Mean")
plt.ylabel("Mean CV RMSE")
plt.title("Cross-Validated RMSE by Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_rmse_comparison.png"), dpi=300)
plt.close()

# CV R2 comparison
plt.figure(figsize=(9, 6))
sns.barplot(data=regression_results_df, x="Model", y="CV_R2_Mean")
plt.ylabel("Mean CV R²")
plt.title("Cross-Validated R² by Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_r2_comparison.png"), dpi=300)
plt.close()

# OLS coefficient plot
coef_df = pd.DataFrame({
    "Feature": ols_model.params.index,
    "Coefficient": ols_model.params.values
})
coef_df = coef_df[coef_df["Feature"] != "const"].copy()
coef_df["abs_coef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=coef_df, x="Coefficient", y="Feature")
plt.title("Top 15 OLS Coefficients by Absolute Magnitude")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ols_top_coefficients.png"), dpi=300)
plt.close()

# VIF plot
vif_plot_df = vif_df[vif_df["Feature"] != "const"].head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=vif_plot_df, x="VIF", y="Feature")
plt.title("Top 15 VIF Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vif_plot.png"), dpi=300)
plt.close()

# -----------------------------
# 13. CLASSIFICATION DATASET
# -----------------------------
clf_df = model_df[all_features + ["improved_next_pts"]].copy()
clf_df = clf_df.dropna(subset=["improved_next_pts"])

X_clf = clf_df[all_features]
y_clf = clf_df["improved_next_pts"].astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
)

# -----------------------------
# 14. LOGISTIC REGRESSION MODEL
# -----------------------------
logit_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=5000))
])

logit_pipeline.fit(X_train_clf, y_train_clf)
clf_preds = logit_pipeline.predict(X_test_clf)
clf_probs = logit_pipeline.predict_proba(X_test_clf)[:, 1]

accuracy = accuracy_score(y_test_clf, clf_preds)
precision = precision_score(y_test_clf, clf_preds)
recall = recall_score(y_test_clf, clf_preds)
f1 = f1_score(y_test_clf, clf_preds)
auc = roc_auc_score(y_test_clf, clf_probs)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(logit_pipeline, X_clf, y_clf, cv=cv, scoring="roc_auc")
cv_acc = cross_val_score(logit_pipeline, X_clf, y_clf, cv=cv, scoring="accuracy")

classification_results_df = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Test_Accuracy": accuracy,
    "Test_Precision": precision,
    "Test_Recall": recall,
    "Test_F1": f1,
    "Test_ROC_AUC": auc,
    "CV_Accuracy_Mean": cv_acc.mean(),
    "CV_Accuracy_SD": cv_acc.std(),
    "CV_AUC_Mean": cv_auc.mean(),
    "CV_AUC_SD": cv_auc.std()
}])

print("\n=== CLASSIFICATION RESULTS ===")
print(classification_results_df)
classification_results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_model_results.csv"), index=False)

# -----------------------------
# 15. CLASSIFICATION FIGURES
# -----------------------------
cm = confusion_matrix(y_test_clf, clf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_confusion_matrix.png"), dpi=300)
plt.close()

RocCurveDisplay.from_predictions(y_test_clf, clf_probs)
plt.title("Logistic Regression ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_roc_curve.png"), dpi=300)
plt.close()

# -----------------------------
# 16. STATSMODELS LOGIT FOR INTERPRETATION
# -----------------------------
logit_numeric = [
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
    "net_rating_lag1"
]
logit_numeric = [c for c in logit_numeric if c in model_df.columns]

logit_categorical = ["pos_proxy"]
logit_categorical = [c for c in logit_categorical if c in model_df.columns]

X_sm_logit, y_sm_logit, logit_temp = make_statsmodels_matrix(
    data=model_df,
    target_col="improved_next_pts",
    numeric_cols=logit_numeric,
    categorical_cols=logit_categorical
)

print("\nLogit design matrix shape:", X_sm_logit.shape)
print("Logit dtypes summary:")
print(X_sm_logit.dtypes.value_counts())

sm_logit_model = sm.Logit(y_sm_logit, X_sm_logit).fit(disp=False, maxiter=200)

with open(os.path.join(OUTPUT_DIR, "logistic_summary.txt"), "w", encoding="utf-8") as f:
    f.write(sm_logit_model.summary().as_text())

odds_ratios = pd.DataFrame({
    "Feature": sm_logit_model.params.index,
    "Coefficient": sm_logit_model.params.values,
    "Odds_Ratio": np.exp(sm_logit_model.params.values)
}).sort_values("Odds_Ratio", ascending=False)

odds_ratios.to_csv(os.path.join(OUTPUT_DIR, "logistic_odds_ratios.csv"), index=False)

or_plot_df = odds_ratios[odds_ratios["Feature"] != "const"].copy()
or_plot_df["abs_coef"] = or_plot_df["Coefficient"].abs()
or_plot_df = or_plot_df.sort_values("abs_coef", ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=or_plot_df, x="Odds_Ratio", y="Feature")
plt.title("Top Logistic Regression Odds Ratios")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_odds_ratio_plot.png"), dpi=300)
plt.close()

print(f"\nAll outputs saved to: {OUTPUT_DIR}")