"""Sinh tám hình đưa vào báo cáo KT3."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from workflow_kt3 import MODEL_SHORT_NAMES, NEW_COLUMNS

BLUE = "#1763A6"
ORANGE = "#AC4F14"
GREEN = "#25826D"
GRAY = "#748394"


def save(fig, path):
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def cv_comparison(folder, results, selected):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(11, 5.1), layout="constrained")
    names = [MODEL_SHORT_NAMES[row["id"]] for row in results]
    positions = np.arange(len(names))
    colors = [ORANGE if row["id"] == selected else BLUE for row in results]
    for ax, metric in zip(axes, ("ROC-AUC", "Accuracy", "F1-Macro")):
        means = [row[f"CV {metric}"] for row in results]
        stds = [row[f"CV {metric} std"] for row in results]
        ax.errorbar(means, positions, xerr=stds, fmt="none", ecolor=GRAY, capsize=3)
        ax.scatter(means, positions, c=colors, s=36, zorder=4)
        ax.set_title(metric)
        ax.set_xlabel("Trung bình ± độ lệch chuẩn")
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    fig.suptitle("So sánh mô hình bằng 5-fold CV trên tập Train")
    save(fig, folder / "so_sanh_mo_hinh_kt2_vs_kt3.png")


def test_curves(folder, y, scores, results):
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4.4), layout="constrained")
    fig_pr, ax_pr = plt.subplots(figsize=(6, 4.4), layout="constrained")
    for row, color in zip(results, (BLUE, ORANGE)):
        key = row["id"]
        label = MODEL_SHORT_NAMES[key]
        fpr, tpr, _ = roc_curve(y, scores[key])
        precision, recall, _ = precision_recall_curve(y, scores[key])
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{label} · AUC {row["ROC-AUC"]:.4f}')
        ax_pr.plot(recall, precision, color=color, linewidth=2,
                   label=f'{label} · AP {row["PR-AUC"]:.4f}')
    ax_roc.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
    ax_roc.set(xlabel="Tỷ lệ dương tính giả (FPR)", ylabel="Tỷ lệ dương tính đúng (TPR)",
               title="Đường ROC trên tập Test", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(loc="lower right")
    ax_pr.axhline(y.mean(), linestyle="--", color=GRAY, linewidth=1, label="Tỷ lệ nhãn duyệt")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall trên tập Test",
              xlim=(0, 1), ylim=(0.45, 1.02))
    ax_pr.legend(loc="lower left")
    save(fig_roc, folder / "duong_cong_roc_so_sanh_kt3.png")
    save(fig_pr, folder / "duong_cong_pr_so_sanh_kt3.png")


def confusion_matrices(folder, results):
    fig, axes = plt.subplots(1, len(results), figsize=(9, 3.65), squeeze=False, layout="constrained")
    for ax, row, cmap in zip(axes[0], results, ("Blues", "Greens")):
        matrix = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, cbar=False,
                    vmin=0, vmax=4200, ax=ax, xticklabels=["Từ chối", "Duyệt"],
                    yticklabels=["Từ chối", "Duyệt"], annot_kws={"size": 12})
        ax.tick_params(axis="y", labelrotation=0)
        ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn quan sát",
               title=f'{MODEL_SHORT_NAMES[row["id"]]} · {row["FP"] + row["FN"]:,} lỗi')
    fig.suptitle("Ma trận nhầm lẫn trên tập Test")
    save(fig, folder / "ma_tran_nham_lan_kt2_vs_kt3.png")


def feature_importance(folder, features):
    ordered = features[::-1]
    names = [item["feature"] for item in ordered]
    values = [item["mean_auc_decrease"] for item in ordered]
    stds = [item["std"] for item in ordered]
    colors = [ORANGE if name in NEW_COLUMNS else BLUE for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    ax.barh(names, values, xerr=stds, color=colors, capsize=2)
    ax.set(xlabel="Mức giảm ROC-AUC trung bình khi hoán vị (± độ lệch chuẩn)",
           title="Độ quan trọng trên Validation · Màu cam là đặc trưng mới")
    save(fig, folder / "do_quan_trong_dac_trung_kt3.png")


def plot_learning(folder, curves, selected):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, (key, label) in zip(axes, [("baseline", "Logistic KT2"),
                                      ("selected", MODEL_SHORT_NAMES[selected])]):
        curve = curves[key]
        cv_mean = np.asarray(curve["CV ROC-AUC"])
        std = np.asarray(curve["CV std"])
        ax.plot(curve["sizes"], curve["Train ROC-AUC"], "o-", color=BLUE, label="Train")
        ax.plot(curve["sizes"], cv_mean, "s-", color=ORANGE, label="5-fold CV")
        ax.fill_between(curve["sizes"], cv_mean - std, cv_mean + std, color=ORANGE, alpha=0.15)
        ax.set(xlabel="Số mẫu huấn luyện trong mỗi fold", title=label)
        ax.legend(loc="best")
    axes[0].set_ylabel("ROC-AUC")
    fig.suptitle("Đường cong học tập trên tập Train")
    save(fig, folder / "duong_cong_hoc_tap_kt3.png")


def plot_validation(folder, curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.05), layout="constrained")
    for ax, (parameter, curve) in zip(axes, curves.items()):
        name = parameter.removeprefix("clf__")
        cv_mean = np.asarray(curve["CV ROC-AUC"])
        std = np.asarray(curve["CV std"])
        values = curve["values"]
        ax.plot(values, curve["Train ROC-AUC"], "o-", color=BLUE, label="Train")
        ax.plot(values, cv_mean, "s-", color=ORANGE, label="5-fold CV")
        ax.fill_between(values, cv_mean - std, cv_mean + std, color=ORANGE, alpha=0.15)
        ax.axvline(curve["picked"], color=GREEN, linestyle="--",
                   label=f'GridSearchCV chọn {curve["picked"]}')
        ax.set(xlabel=name, ylabel="ROC-AUC")
        ax.legend(loc="best")
    fig.suptitle("Đường cong kiểm định HGB · Các tham số còn lại giữ cố định")
    save(fig, folder / "duong_cong_kiem_dinh_hyperparameter.png")


def plot_ablation(folder, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, layout="constrained")
    labels = [row["So sánh"] for row in rows]
    positions = np.arange(len(labels))
    for ax, metric in zip(axes, ("Δ ROC-AUC", "Δ Accuracy (điểm %)")):
        values = [row[metric] for row in rows]
        colors = [GREEN if value >= 0 else ORANGE for value in values]
        ax.barh(positions, values, color=colors, height=0.58)
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.set_xlabel(metric)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Ablation · Thay đổi theo từng cặp trên cùng 5-fold CV")
    save(fig, folder / "ablation_study_kt3.png")


def write_figures(folder, summary, test_labels, test_scores):
    folder.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    cv_comparison(folder, summary["cv_results"], summary["selection"]["model_id"])
    test_curves(folder, test_labels, test_scores, summary["test_results"])
    confusion_matrices(folder, summary["test_results"])
    feature_importance(folder, summary["top_features"])
    plot_learning(folder, summary["learning_curves"], summary["selection"]["model_id"])
    plot_validation(folder, summary["validation_curves"])
    plot_ablation(folder, summary["ablation_study"])
