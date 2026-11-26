"""Chạy thí nghiệm KT3, từ CV trên Train tới đánh giá mô hình đã chốt trên Test."""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve, validation_curve
from threadpoolctl import threadpool_limits

from workflow_kt3 import (
    CAT_COLUMNS, DATA_PATH, HGB_GRID, MODEL_NAMES, NEW_COLUMNS, ROOT, SCORING, SEED,
    SUMMARY_PATH, engineer_features, evaluate, get_model_score, load_splits, make_candidates,
    make_stacking, stratified_cv, threshold_table,
)

matplotlib.use("Agg")

MODEL_FILES = {
    "hgb_tuned": "best_hist_gradient_boosting_pipeline.joblib",
    "rf": "random_forest_pipeline.joblib",
    "stacking": "stacking_pipeline.joblib",
}


def score_cv(estimator, train, cv, jobs):
    scores = cross_validate(
        estimator, train.X, train.y, scoring=SCORING, cv=cv, n_jobs=jobs,
        return_train_score=True, error_score="raise",
    )
    row = {"Train AUC": float(scores["train_roc_auc"].mean())}
    for metric, label in {
        "roc_auc": "ROC-AUC", "accuracy": "Accuracy", "f1_macro": "F1-Macro",
    }.items():
        fold_scores = scores[f"test_{metric}"]
        row[f"CV {label}"] = float(fold_scores.mean())
        row[f"CV {label} std"] = float(fold_scores.std())
    row["Fold ROC-AUC"] = scores["test_roc_auc"].tolist()
    return row


def tune_hgb(estimator, train, cv, jobs):
    search = GridSearchCV(
        estimator, HGB_GRID, scoring="roc_auc", cv=cv, n_jobs=jobs, error_score="raise"
    )
    search.fit(train.X, train.y)
    results = search.cv_results_
    grid_rows = [
        {
            "rank": int(rank), "params": params, "mean_roc_auc": float(mean),
            "std_roc_auc": float(std),
        }
        for rank, params, mean, std in zip(
            results["rank_test_score"], results["params"],
            results["mean_test_score"], results["std_test_score"],
        )
    ]
    return search.best_estimator_, search.best_params_, sorted(grid_rows, key=lambda row: row["rank"])


def measure_batch_ms(estimator, batch, repeats=7):
    """Độ trễ trung vị cho một lần predict trên toàn bộ batch."""
    estimator.predict(batch)
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        estimator.predict(batch)
        durations.append((perf_counter() - start) * 1000)
    return median(durations)


def paired_ablation(cv_by_model):
    comparisons = [
        ("Thêm đặc trưng cho Logistic", "baseline", "logistic_fe"),
        ("Thêm đặc trưng cho HGB", "hgb_raw", "hgb_fe"),
        ("CART → Random Forest", "cart", "rf"),
        ("Random Forest → HGB", "rf", "hgb_fe"),
        ("Tối ưu siêu tham số HGB", "hgb_fe", "hgb_tuned"),
        ("HGB tối ưu → Stacking", "hgb_tuned", "stacking"),
    ]
    rows = []
    for label, before, after in comparisons:
        old = cv_by_model[before]
        new = cv_by_model[after]
        rows.append({
            "So sánh": label, "Trước": before, "Sau": after,
            "AUC trước": old["CV ROC-AUC"], "AUC sau": new["CV ROC-AUC"],
            "Δ ROC-AUC": new["CV ROC-AUC"] - old["CV ROC-AUC"],
            "Δ Accuracy (điểm %)": 100 * (new["CV Accuracy"] - old["CV Accuracy"]),
            "Δ F1-Macro": new["CV F1-Macro"] - old["CV F1-Macro"],
        })
    return rows


def baseline_error_profile(baseline, validation):
    prediction = baseline.predict(validation.X)
    columns = ["credit_score", "debt_to_income_ratio", "annual_income"]
    rows = []
    for name, mask in [
        ("FP", (validation.y == 0) & (prediction == 1)),
        ("FN", (validation.y == 1) & (prediction == 0)),
        ("Toàn bộ Validation", np.ones(len(validation.y), dtype=bool)),
    ]:
        group = validation.X.loc[mask, columns]
        rows.append({"Nhóm": name, "Số hồ sơ": len(group), **group.median().to_dict()})
    return rows


def make_curves(baseline, selected, tuned_hgb, train, cv, jobs):
    learning = {}
    for key, model in [("baseline", baseline), ("selected", selected)]:
        sizes, scores_train, scores_cv = learning_curve(
            model, train.X, train.y, cv=cv, scoring="roc_auc",
            train_sizes=[0.1, 0.325, 0.55, 0.775, 1.0], shuffle=True,
            random_state=SEED, n_jobs=jobs, error_score="raise",
        )
        learning[key] = {
            "sizes": sizes.tolist(),
            "Train ROC-AUC": scores_train.mean(axis=1).tolist(),
            "CV ROC-AUC": scores_cv.mean(axis=1).tolist(),
            "CV std": scores_cv.std(axis=1).tolist(),
        }

    validation = {}
    for parameter, values in [
        ("clf__learning_rate", [0.03, 0.05, 0.08, 0.12, 0.16]),
        ("clf__max_depth", [3, 4, 6, 8, 10]),
    ]:
        scores_train, scores_cv = validation_curve(
            tuned_hgb, train.X, train.y, param_name=parameter, param_range=values,
            cv=cv, scoring="roc_auc", n_jobs=jobs, error_score="raise",
        )
        validation[parameter] = {
            "values": values, "picked": tuned_hgb.get_params()[parameter],
            "Train ROC-AUC": scores_train.mean(axis=1).tolist(),
            "CV ROC-AUC": scores_cv.mean(axis=1).tolist(),
            "CV std": scores_cv.std(axis=1).tolist(),
        }
    return learning, validation


def run(jobs):
    from plot_results_kt3 import write_figures

    train, validation, test = load_splits()
    cv = stratified_cv()
    candidates = make_candidates(train.X.columns)
    print(f"Dữ liệu: {DATA_PATH.name}. Train/Validation/Test: "
          f"{len(train.y)}/{len(validation.y)}/{len(test.y)}", flush=True)

    print("Tìm siêu tham số HGB trên 5-fold CV của Train...", flush=True)
    tuned, best_params, grid_rows = tune_hgb(candidates["hgb_fe"], train, cv, jobs)
    candidates["hgb_tuned"] = tuned
    candidates["stacking"] = make_stacking(candidates)

    cv_results = {}
    for key, estimator in candidates.items():
        row = score_cv(estimator, train, cv, jobs)
        cv_results[key] = {"id": key, "Mô hình": MODEL_NAMES[key], **row}
        print(f'{MODEL_NAMES[key]}: ROC-AUC {row["CV ROC-AUC"]:.5f} '
              f'± {row["CV ROC-AUC std"]:.5f}', flush=True)

    # Khóa lựa chọn trước khi dự đoán trên tập Test.
    # Khi bằng điểm, giữ ứng viên xuất hiện trước trong thứ tự đã khai báo.
    selected = max(cv_results, key=lambda key: cv_results[key]["CV ROC-AUC"])
    print(f"Đã chọn {MODEL_NAMES[selected]} theo trung bình ROC-AUC 5-fold.", flush=True)

    fitted = {key: clone(model).fit(train.X, train.y) for key, model in candidates.items()}
    val_results = []
    batch = validation.X.iloc[:1000]
    latency = {}
    for key, estimator in fitted.items():
        val_results.append({
            "id": key, "Mô hình": MODEL_NAMES[key],
            **evaluate(validation.y, estimator.predict(validation.X),
                       get_model_score(estimator, validation.X)),
        })
        latency[key] = measure_batch_ms(estimator, batch)

    print("Tính đường cong và độ quan trọng đặc trưng trên Train/Validation...", flush=True)
    learning, validation_curves = make_curves(
        candidates["baseline"], candidates[selected], tuned, train, cv, jobs
    )
    if selected in {"baseline", "hgb_raw", "stacking"}:
        # Stacking nhận hồ sơ gốc và có nhiều pipeline tầng 1.
        importance_X, importance_estimator = validation.X, fitted[selected]
    else:
        # Hoán vị từng cột trên đầu vào của bộ tiền xử lý để hiện cả 5 biến thiết kế.
        importance_X = engineer_features(validation.X)
        importance_estimator = fitted[selected][1:]
    importance = permutation_importance(
        importance_estimator, importance_X, validation.y, n_repeats=5,
        random_state=SEED, scoring="roc_auc", n_jobs=jobs,
    )
    order = np.argsort(importance.importances_mean)[::-1][:15]
    features = [
        {
            "feature": importance_X.columns[index],
            "mean_auc_decrease": float(importance.importances_mean[index]),
            "std": float(importance.importances_std[index]),
        }
        for index in order
    ]

    # Tập Test chỉ phục vụ đối chiếu baseline và mô hình đã chốt.
    test_results = []
    test_scores = {}
    for key in dict.fromkeys(("baseline", selected)):
        estimator = fitted[key]
        score = get_model_score(estimator, test.X)
        test_scores[key] = score
        test_results.append({
            "id": key, "Mô hình": MODEL_NAMES[key],
            **evaluate(test.y, estimator.predict(test.X), score),
        })

    files = dict(MODEL_FILES)
    files.setdefault(selected, "selected_pipeline.joblib")
    threshold_rows = threshold_table(
        validation.y, fitted["hgb_tuned"].predict_proba(validation.X)[:, 1]
    )
    summary = {
        "environment": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
        },
        "dataset": {
            "source_file": DATA_PATH.name, "total_samples": len(train.y) + len(validation.y) + len(test.y),
            "train_samples": len(train.y), "val_samples": len(validation.y), "test_samples": len(test.y),
            "features_raw": train.X.shape[1], "cat_features": len(CAT_COLUMNS),
            "features_engineered": train.X.shape[1] + len(NEW_COLUMNS),
        },
        "selection": {
            "model_id": selected, "model_name": MODEL_NAMES[selected],
            "criterion": "Mean ROC-AUC, StratifiedKFold 5-fold trên Train",
            "seed": SEED, "artifact": files[selected],
        },
        "cv_results": list(cv_results.values()),
        "val_results": val_results,
        "test_results": test_results,
        "ablation_study": paired_ablation(cv_results),
        "grid_search_hgb": {"space": HGB_GRID, "rows": grid_rows, "best_params": best_params},
        "latency_ms_per_1000_samples": latency,
        "top_features": features,
        "baseline_val_error_profile": baseline_error_profile(fitted["baseline"], validation),
        "threshold_validation": threshold_rows.to_dict(orient="records"),
        "learning_curves": learning,
        "validation_curves": validation_curves,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    for key, name in files.items():
        joblib.dump(fitted[key], SUMMARY_PATH.parent / name)
    write_figures(ROOT / "images", summary, test.y, test_scores)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    end = test_results[-1]
    print(f'Test {end["Mô hình"]}: AUC={end["ROC-AUC"]:.5f}, '
          f'Accuracy={end["Accuracy"]:.2%}, FP={end["FP"]}, FN={end["FN"]}', flush=True)
    print(f"Đã lưu JSON, các pipeline và 8 hình vào {ROOT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(4, joblib.cpu_count()),
                        help="Số tiến trình chạy CV và GridSearchCV")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs phải từ 1 trở lên")
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(1):
        run(args.jobs)


if __name__ == "__main__":
    main()
