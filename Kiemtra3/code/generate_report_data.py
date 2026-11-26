"""Cập nhật bảng và số liệu LaTeX từ JSON của lần thực nghiệm."""

import json

from workflow_kt3 import MODEL_SHORT_NAMES, NEW_COLUMNS, ROOT, SUMMARY_PATH

TEX_PATH = ROOT / "report" / "results.tex"
ESCAPE = str.maketrans({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"})


def text(value):
    return str(value).translate(ESCAPE)


def number(value, decimals=4):
    return f"{value:.{decimals}f}".replace(".", ",")


def signed(value, decimals=4):
    return f"{value:+.{decimals}f}".replace(".", ",")


def percent(value):
    return number(value * 100, 2) + r"\%"


def integer(value):
    return f"{value:,}".replace(",", ".")


def command(name, value):
    slash = chr(92)
    return f"{slash}newcommand{{{slash}{name}}}{{{value}}}"


def table(name, rows):
    def render(row):
        return " & ".join(map(str, row)) + r" \\"
    return command(name, "%\n" + "\n".join(render(row) for row in rows) + "\n")


def main():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    selected = summary["selection"]["model_id"]
    cv = {row["id"]: row for row in summary["cv_results"]}
    validation = {row["id"]: row for row in summary["val_results"]}
    test = {row["id"]: row for row in summary["test_results"]}
    latency = summary["latency_ms_per_1000_samples"]
    baseline_test = test["baseline"]
    final_cv, final_val, final_test = cv[selected], validation[selected], test[selected]
    base_errors = baseline_test["FP"] + baseline_test["FN"]
    final_errors = final_test["FP"] + final_test["FN"]
    grid = summary["grid_search_hgb"]
    best = grid["best_params"]
    leading_feature = summary["top_features"][0]
    baseline_curve = summary["learning_curves"]["baseline"]

    values = {
        "KTPython": summary["environment"]["python"],
        "KTScikit": summary["environment"]["scikit_learn"],
        "KTFinalModel": text(MODEL_SHORT_NAMES[selected]),
        "KTBaseCV": number(cv["baseline"]["CV ROC-AUC"]),
        "KTBaseCVStd": number(cv["baseline"]["CV ROC-AUC std"]),
        "KTBaseCurveStart": number(baseline_curve["CV ROC-AUC"][0]),
        "KTBaseCurveEnd": number(baseline_curve["CV ROC-AUC"][-1]),
        "KTFinalTrainAUC": number(final_cv["Train AUC"]),
        "KTFinalCV": number(final_cv["CV ROC-AUC"]),
        "KTFinalCVStd": number(final_cv["CV ROC-AUC std"]),
        "KTFinalVal": number(final_val["ROC-AUC"]),
        "KTBaseTest": number(baseline_test["ROC-AUC"], 5),
        "KTFinalTest": number(final_test["ROC-AUC"], 5),
        "KTBaseAcc": percent(baseline_test["Accuracy"]),
        "KTFinalAcc": percent(final_test["Accuracy"]),
        "KTFinalFscore": number(final_test["F1-Macro"]),
        "KTFinalPR": number(final_test["PR-AUC"]),
        "KTCVTestGap": number(abs(final_cv["CV ROC-AUC"] - final_test["ROC-AUC"]), 5),
        "KTTrainTestGap": number(abs(final_cv["Train AUC"] - final_test["ROC-AUC"]), 5),
        "KTBaseFP": integer(baseline_test["FP"]),
        "KTBaseFN": integer(baseline_test["FN"]),
        "KTFinalFP": integer(final_test["FP"]),
        "KTFinalFN": integer(final_test["FN"]),
        "KTBaseErrors": integer(base_errors),
        "KTFinalErrors": integer(final_errors),
        "KTErrorsReduced": integer(base_errors - final_errors),
        "KTErrorsReducedPct": percent((base_errors - final_errors) / base_errors),
        "KTFinalLatency": number(latency[selected], 2),
        "KTGridSize": len(grid["rows"]),
        "KTBestRate": number(best["clf__learning_rate"], 2),
        "KTBestDepth": best["clf__max_depth"],
        "KTBestIter": best["clf__max_iter"],
        "KTBestLTwo": number(best["clf__l2_regularization"], 2),
        "KTLeadingFeature": r"\texttt{" + text(leading_feature["feature"]) + "}",
        "KTLeadingFeatureImpact": number(leading_feature["mean_auc_decrease"]),
        "KTLogisticFEDelta": signed(cv["logistic_fe"]["CV ROC-AUC"] - cv["baseline"]["CV ROC-AUC"], 5),
        "KTHGBFEDelta": signed(cv["hgb_fe"]["CV ROC-AUC"] - cv["hgb_raw"]["CV ROC-AUC"], 5),
        "KTHGBTuningDelta": signed(cv["hgb_tuned"]["CV ROC-AUC"] - cv["hgb_fe"]["CV ROC-AUC"], 5),
    }
    body = ["% Tự động sinh từ models/kt3_experiment_summary.json."]
    body.extend(command(name, value) for name, value in values.items())
    body.append(table("KTCVRows", [
        (text(MODEL_SHORT_NAMES[key]), number(row["CV ROC-AUC"], 5),
         number(row["CV ROC-AUC std"], 5), percent(row["CV Accuracy"]), number(row["CV F1-Macro"]))
        for key, row in cv.items()
    ]))
    body.append(table("KTErrorRows", [
        (text(row["Nhóm"]), integer(row["Số hồ sơ"]), number(row["credit_score"], 1),
         number(row["debt_to_income_ratio"], 3), integer(round(row["annual_income"])))
        for row in summary["baseline_val_error_profile"]
    ]))
    body.append(table("KTGridRows", [
        (row["rank"], number(row["params"]["clf__learning_rate"], 2),
         row["params"]["clf__max_depth"], row["params"]["clf__max_iter"],
         number(row["params"]["clf__l2_regularization"], 2), number(row["mean_roc_auc"], 5))
        for row in grid["rows"][:5]
    ]))
    body.append(table("KTAblationRows", [
        (text(row["So sánh"]), number(row["AUC trước"], 5), number(row["AUC sau"], 5),
         signed(row["Δ ROC-AUC"], 5), signed(row["Δ Accuracy (điểm %)"], 2),
         signed(row["Δ F1-Macro"], 4))
        for row in summary["ablation_study"]
    ]))
    body.append(table("KTGeneralizationRows", [
        (text(MODEL_SHORT_NAMES[key]), number(cv[key]["CV ROC-AUC"]),
         number(cv[key]["CV ROC-AUC std"]), number(validation[key]["ROC-AUC"]),
         number(test[key]["ROC-AUC"], 5), percent(test[key]["Accuracy"]),
         number(test[key]["F1-Macro"]))
        for key in test
    ]))
    body.append(table("KTExtraTestRows", [
        (text(MODEL_SHORT_NAMES[key]), number(row["PR-AUC"]), number(row["Precision"]),
         number(row["Recall"]), number(row["Specificity"]), row["FP"], row["FN"])
        for key, row in test.items()
    ]))
    body.append(table("KTTradeoffRows", [
        (text(MODEL_SHORT_NAMES[key]), number(cv[key]["CV ROC-AUC"], 5),
         number(cv[key]["CV ROC-AUC std"], 5), number(validation[key]["ROC-AUC"]),
         number(latency[key], 2), interpretation)
        for key, interpretation in [
            ("baseline", "Hệ số tuyến tính"),
            ("rf", "Độ quan trọng tổng hợp"),
            ("hgb_raw", "Hoán vị đặc trưng"),
            ("hgb_tuned", "Hoán vị đặc trưng"),
            ("stacking", "Nhiều tầng mô hình"),
        ]
    ]))
    body.append(table("KTThresholdRows", [
        (number(row["Ngưỡng"], 2), row["FP"], row["FN"],
         number(row["Specificity"]), number(row["Recall"]))
        for row in summary["threshold_validation"]
    ]))
    TEX_PATH.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"Đã tạo {TEX_PATH}")


if __name__ == "__main__":
    main()
