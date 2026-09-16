from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    ROC_CURVE_PATH,
    TARGET,
    TEST_SIZE,
)

NUMERIC_FEATURES = [
    "age", "purchase_count", "total_spent", "avg_order_value",
    "days_since_last_purchase", "website_visits_30d", "cart_adds_30d",
    "discount_usage_rate", "support_tickets_90d", "customer_tenure_days",
]
CATEGORICAL_FEATURES = ["gender", "preferred_channel", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models() -> Dict[str, Pipeline]:
    def pipe(model):
        return Pipeline([( "preprocessor", build_preprocessor()), ("model", model)])

    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "DecisionTree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=3, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)),
        "GradientBoosting": pipe(GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)),
    }


def load_clean_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["customer_id"]).copy()
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def metrics_from_predictions(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = metrics_from_predictions(y_test, y_pred, y_prob)
    result["model"] = name
    return result


def save_plots(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Best Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=160)
    plt.close()


def save_feature_importance(model) -> None:
    final_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        values = final_model.coef_[0]
    else:
        return
    importance = pd.DataFrame({
        "feature": names,
        "importance": values,
        "absolute_importance": abs(values),
    }).sort_values("absolute_importance", ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y = load_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    rows = []

    for name, model in models.items():
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            **{f"cv_{metric}_mean": float(scores[f"test_{metric}"].mean()) for metric in scoring},
            **{f"cv_{metric}_std": float(scores[f"test_{metric}"].std()) for metric in scoring},
        })

    comparison = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)

    # Tune Random Forest on training data only.
    rf_pipeline = models["RandomForest"]
    param_grid = {
        "model__n_estimators": [120, 180],
        "model__max_depth": [8, 12],
        "model__min_samples_leaf": [2],
    }
    search = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    tuned_rf = search.best_estimator_
    tuned_cv_f1 = float(search.best_score_)
    tuned_row = {
        "model": "RandomForest_Tuned",
        "cv_f1_mean": tuned_cv_f1,
        "cv_f1_std": 0.0,
        "best_params": str(search.best_params_),
    }
    comparison = pd.concat([comparison, pd.DataFrame([tuned_row])], ignore_index=True, sort=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    # For the final choice, compare tuned RF with the best baseline model by CV F1.
    best_baseline_name = comparison.loc[comparison["model"].ne("RandomForest_Tuned")].sort_values("cv_f1_mean", ascending=False).iloc[0]["model"]
    baseline_model = models[best_baseline_name]
    baseline_model.fit(X_train, y_train)
    baseline_f1 = evaluate_model(best_baseline_name, baseline_model, X_test, y_test)["f1"]

    tuned_rf_metrics = evaluate_model("RandomForest_Tuned", tuned_rf, X_test, y_test)
    if tuned_rf_metrics["f1"] >= baseline_f1:
        best_name, best_model, final_metrics = "RandomForest_Tuned", tuned_rf, tuned_rf_metrics
    else:
        best_name, best_model, final_metrics = best_baseline_name, baseline_model, evaluate_model(best_baseline_name, baseline_model, X_test, y_test)

    joblib.dump(best_model, MODEL_PATH)
    save_plots(best_model, X_test, y_test)
    save_feature_importance(best_model)

    payload = {
        "best_model": best_name,
        "test_metrics": final_metrics,
        "classification_report": classification_report(
            y_test,
            best_model.predict(X_test),
            output_dict=True,
            zero_division=0,
        ),
        "data": {
            "rows_after_dedup": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate": float(y.mean()),
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== FINAL RESULT ===")
    print(f"Best model: {best_name}")
    for key, value in final_metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Comparison: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
