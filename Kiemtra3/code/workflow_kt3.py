"""Dữ liệu, cách chia và pipeline dùng chung cho Kiểm tra 3."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Loan_approval_data_2025.csv"
SUMMARY_PATH = ROOT / "models" / "kt3_experiment_summary.json"
SEED = 42
CAT_COLUMNS = ("occupation_status", "product_type", "loan_intent")
NEW_COLUMNS = (
    "debt_service_ratio",
    "savings_to_loan_ratio",
    "credit_risk_score",
    "dti_interest_burden",
    "employment_stability",
)

MODEL_NAMES = {
    "baseline": "Logistic Regression (KT2)",
    "logistic_fe": "Logistic Regression + FE",
    "cart": "Decision Tree (CART)",
    "svm": "SVM tuyến tính (SGD hinge)",
    "rf": "Random Forest",
    "hgb_raw": "HistGradientBoosting, đặc trưng gốc",
    "hgb_fe": "HistGradientBoosting + FE",
    "hgb_tuned": "HistGradientBoosting + FE, tối ưu",
    "stacking": "Stacking (RF + HGB + Logistic)",
}
MODEL_SHORT_NAMES = dict(zip(MODEL_NAMES, [
    "Logistic KT2", "Logistic + FE", "CART", "SVM tuyến tính", "Random Forest",
    "HGB (gốc)", "HGB + FE", "HGB tối ưu", "Stacking",
]))
HGB_GRID = {
    "clf__learning_rate": [0.05, 0.08, 0.12],
    "clf__max_depth": [6, 8],
    "clf__max_iter": [150, 200],
    "clf__l2_regularization": [0.0, 0.5],
}
SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}


@dataclass(frozen=True)
class DataSplit:
    X: pd.DataFrame
    y: pd.Series


def load_splits():
    """Dùng chính cách chia hai bước của KT2, với chỉ số dòng của CSV gốc."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["customer_id", "loan_status"])
    y = data["loan_status"]
    X_train, X_other, y_train, y_other = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_other, y_other, test_size=0.50, random_state=SEED, stratify=y_other
    )
    return DataSplit(X_train, y_train), DataSplit(X_val, y_val), DataSplit(X_test, y_test)


def stratified_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def engineer_features(X):
    """Tính theo từng hồ sơ, không học tham số từ dữ liệu hay nhãn."""
    features = X.copy()
    features["debt_service_ratio"] = (
        (X["current_debt"] + X["loan_amount"]) / (X["annual_income"] + 1.0)
    )
    features["savings_to_loan_ratio"] = X["savings_assets"] / (X["loan_amount"] + 1.0)
    features["credit_risk_score"] = (
        3.0 * X["defaults_on_file"]
        + 1.5 * X["delinquencies_last_2yrs"]
        + X["derogatory_marks"]
    )
    features["dti_interest_burden"] = X["debt_to_income_ratio"] * X["interest_rate"]
    features["employment_stability"] = (
        X["years_employed"] / (X["age"] - 18.0).clip(lower=1.0)
    )
    return features


def make_pipeline(classifier, columns, *, add_features=True):
    numeric = [column for column in columns if column not in CAT_COLUMNS]
    steps = []
    if add_features:
        numeric.extend(NEW_COLUMNS)
        steps.append(("features", FunctionTransformer(engineer_features)))
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
         list(CAT_COLUMNS)),
    ])
    steps.extend([("prep", preprocessor), ("clf", classifier)])
    return Pipeline(steps)


def make_hgb(columns, *, add_features=True, **overrides):
    params = {"max_depth": 6, "max_iter": 150, "learning_rate": 0.08, "random_state": SEED}
    params.update(overrides)
    return make_pipeline(HistGradientBoostingClassifier(**params), columns, add_features=add_features)


def make_candidates(columns):
    return {
        "baseline": make_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
            columns, add_features=False,
        ),
        "logistic_fe": make_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=SEED), columns
        ),
        "cart": make_pipeline(
            DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                                   min_samples_leaf=10, random_state=SEED), columns
        ),
        "svm": make_pipeline(
            SGDClassifier(loss="hinge", alpha=0.0001, max_iter=1000, random_state=SEED), columns
        ),
        "rf": make_pipeline(
            RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10,
                                   random_state=SEED, n_jobs=1), columns
        ),
        "hgb_raw": make_hgb(columns, add_features=False),
        "hgb_fe": make_hgb(columns),
    }


def make_stacking(candidates):
    # Mỗi mô hình tầng 1 có pipeline riêng, fit trong từng fold tạo dự đoán OOF.
    return StackingClassifier(
        estimators=[
            ("rf", candidates["rf"]),
            ("hgb", candidates["hgb_tuned"]),
            ("lr", candidates["logistic_fe"]),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=stratified_cv(3),
        n_jobs=1,
    )


def get_model_score(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def evaluate(y, prediction, score):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(accuracy_score(y, prediction)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Recall": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1-Macro": float(f1_score(y, prediction, average="macro")),
        "ROC-AUC": float(roc_auc_score(y, score)),
        "PR-AUC": float(average_precision_score(y, score)),
    }


def threshold_table(y, probabilities, thresholds=(0.40, 0.50, 0.60, 0.70)):
    rows = []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append({
            "Ngưỡng": threshold, "FP": int(fp), "FN": int(fn),
            "Specificity": float(tn / (tn + fp)), "Recall": float(tp / (tp + fn)),
        })
    return pd.DataFrame(rows)
