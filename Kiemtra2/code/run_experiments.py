"""
Bài kiểm tra 2 - Tuần 6: Xây dựng, huấn luyện và đánh giá mô hình
Học phần: Học máy - Nhóm 13 (Kiều Việt Anh - DTC235200026 & Lương Chí Công - DTC235200092)
Đề tài: Dự đoán và giải thích quyết định phê duyệt khoản vay
Phân chia dữ liệu: 70% Train (35.000) - 15% Val (7.500) - 15% Test (7.500)

Chỉ sử dụng các thuật toán đã học trong Lesson 1 - Lesson 5 (machine-learning-main):
- DummyClassifier (Baseline lớp đa số - Lesson 04)
- Perceptron / PLA (Thuật toán học Perceptron - Lesson 04)
- SGDClassifier(loss='log_loss') (Hồi quy Logistic tối ưu bằng Stochastic Gradient Descent - Lesson 04 & 05)
- LogisticRegression mặc định và class_weight='balanced' (Tối ưu L-BFGS với L2 - Lesson 04)
"""

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    learning_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Times New Roman"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

RANDOM_SEED = 42


def get_model_score(pipe, X_data):
    """Lấy điểm số liên tục phục vụ ROC-AUC và ngưỡng (hỗ trợ cả predict_proba và decision_function)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X_data)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X_data)
    return pipe.predict(X_data)


def evaluate(y_true, y_pred, y_score):
    """Hàm đánh giá chuẩn hóa (mô phỏng lesson_04/10_lung_cancer_logistic_regression.ipynb)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity (Rec 0)": float(specificity),
        "F1-Macro": float(f1_score(y_true, y_pred, average="macro")),
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
    }


def main():
    # 1. Đường dẫn tương đối độc lập máy cá nhân
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    data_paths = [
        os.path.join(script_dir, "..", "data", "Loan_approval_data_2025.csv"),
        os.path.join(project_dir, "Kiemtra2", "data", "Loan_approval_data_2025.csv"),
        os.path.join(project_dir, "Nhom13", "Kiemtra2", "data", "Loan_approval_data_2025.csv"),
    ]

    data_file = next((p for p in data_paths if os.path.exists(p)), None)
    if not data_file:
        raise FileNotFoundError("Không tìm thấy file Loan_approval_data_2025.csv!")

    print(f"[1] Đọc dữ liệu từ: {data_file}")
    df = pd.read_csv(data_file)
    print(f"    - Kích thước tập dữ liệu: {df.shape[0]:,} mẫu, {df.shape[1]} thuộc tính")
    print("    - Tỷ lệ nhãn mục tiêu (loan_status):")
    print(df["loan_status"].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

    # 2. Chuẩn bị X, y (loại bỏ customer_id tránh rò rỉ định danh)
    X = df.drop(columns=["customer_id", "loan_status"]).copy()
    y = df["loan_status"].copy()

    # PHÂN CHIA TẬP DỮ LIỆU: 70% Train - 15% Validation - 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"\n[2] Phân chia tập dữ liệu (70% Train - 15% Val - 15% Test):")
    print(f"    - Tập Huấn luyện (Train): {X_train.shape[0]:,} mẫu ({X_train.shape[0]/len(df):.1%}) | Tỷ lệ duyệt: {y_train.mean():.2%}")
    print(f"    - Tập Kiểm định  (Val)  : {X_val.shape[0]:,} mẫu  ({X_val.shape[0]/len(df):.1%}) | Tỷ lệ duyệt: {y_val.mean():.2%}")
    print(f"    - Tập Kiểm tra   (Test) : {X_test.shape[0]:,} mẫu  ({X_test.shape[0]/len(df):.1%}) | Tỷ lệ duyệt: {y_test.mean():.2%}")

    cat_cols = ["occupation_status", "product_type", "loan_intent"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"    - Số đặc trưng số (numerical): {len(num_cols)} (chuẩn hóa Z-score đầy đủ)")
    print(f"    - Số đặc trưng phân loại (categorical): {len(cat_cols)}")

    # 3. ColumnTransformer tiền xử lý khép kín (chỉ fit bên trong Pipeline theo từng fold)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # 4. Danh mục 5 cấu hình mô hình trong phạm vi Lesson 04 & Lesson 05
    models = {
        "Dummy (Baseline đa số)": DummyClassifier(strategy="most_frequent"),
        "Perceptron (PLA)": Perceptron(
            max_iter=1000, tol=1e-3, random_state=RANDOM_SEED
        ),
        "SGDClassifier (Log-loss + GD)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=1e-4,
            random_state=RANDOM_SEED,
        ),
        "Logistic Regression (Mặc định)": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_SEED
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    # 5. Kiểm định chéo 5-Fold trên Tập Train (N=35.000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring_metrics = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "precision": "precision",
        "recall": "recall",
    }

    print(f"\n[3] Thực hiện Kiểm định chéo 5-Fold trên Tập huấn luyện (N={len(X_train):,})...")
    cv_results_summary = []
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        t0 = time.time()
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metrics,
            return_train_score=True,
            n_jobs=-1,
        )
        cv_time = time.time() - t0

        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        row = {
            "Mô hình": name,
            "Thời gian (s)": float(cv_time),
            "Train Acc": float(scores["train_accuracy"].mean()),
            "Val Acc (mean)": float(scores["test_accuracy"].mean()),
            "Val Acc (std)": float(scores["test_accuracy"].std()),
            "Train AUC": float(scores["train_roc_auc"].mean()),
            "Val AUC (mean)": float(scores["test_roc_auc"].mean()),
            "Val AUC (std)": float(scores["test_roc_auc"].std()),
            "Train F1-Macro": float(scores["train_f1_macro"].mean()),
            "Val F1-Macro (mean)": float(scores["test_f1_macro"].mean()),
            "Val F1-Macro (std)": float(scores["test_f1_macro"].std()),
            "Val Precision (mean)": float(scores["test_precision"].mean()),
            "Val Precision (std)": float(scores["test_precision"].std()),
            "Val Recall (mean)": float(scores["test_recall"].mean()),
            "Val Recall (std)": float(scores["test_recall"].std()),
        }
        cv_results_summary.append(row)
        print(
            f"    --> {name:34s} | CV AUC: {row['Val AUC (mean)']:.4f} +/- {row['Val AUC (std)']:.4f} "
            f"| F1-Macro: {row['Val F1-Macro (mean)']:.4f} +/- {row['Val F1-Macro (std)']:.4f} | Time: {cv_time:.2f}s"
        )

    # 6. Đánh giá trên Tập kiểm định (Validation Set - N=7.500)
    print(f"\n[4] Đánh giá trên Tập kiểm định (Validation Set - N={len(X_val):,})...")
    val_results = []
    for name, pipe in fitted_pipelines.items():
        score_val = get_model_score(pipe, X_val)
        pred_val = pipe.predict(X_val)
        res = evaluate(y_val, pred_val, score_val)
        res["Mô hình"] = name
        val_results.append(res)
        print(
            f"    --> {name:34s} | Val AUC: {res['ROC-AUC']:.4f} | F1-Macro: {res['F1-Macro']:.4f} "
            f"| Rec(1): {res['Recall']:.4f} | Spec(0): {res['Specificity (Rec 0)']:.4f}"
        )
    df_val = pd.DataFrame(val_results)

    # 7. Tinh chỉnh ngưỡng quyết định Out-of-Fold (theo lesson_04/10) trên Train + kiểm chứng trên Val
    print("\n[5] Khảo sát ngưỡng quyết định Out-of-Fold (Train OOF) cho Logistic Regression & SGDClassifier...")
    threshold_analysis = {}
    for target_model_name in ["Logistic Regression (Mặc định)", "SGDClassifier (Log-loss + GD)"]:
        pipe_target = fitted_pipelines[target_model_name]
        oof_prob = cross_val_predict(
            pipe_target, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        val_prob = pipe_target.predict_proba(X_val)[:, 1]
        thresh_rows = []
        for tau in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            oof_eval = evaluate(y_train, (oof_prob >= tau).astype(int), oof_prob)
            val_eval = evaluate(y_val, (val_prob >= tau).astype(int), val_prob)
            thresh_rows.append(
                {
                    "Ngưỡng (tau)": tau,
                    "OOF F1-Macro": oof_eval["F1-Macro"],
                    "OOF Precision": oof_eval["Precision"],
                    "OOF Recall": oof_eval["Recall"],
                    "OOF Spec(0)": oof_eval["Specificity (Rec 0)"],
                    "OOF FP": oof_eval["FP"],
                    "OOF FN": oof_eval["FN"],
                    "Val F1-Macro": val_eval["F1-Macro"],
                    "Val FP": val_eval["FP"],
                    "Val FN": val_eval["FN"],
                }
            )
        threshold_analysis[target_model_name] = thresh_rows
        print(f"  * {target_model_name}:")
        print(pd.DataFrame(thresh_rows).to_string(index=False))

    # 8. Thử nghiệm siêu tham số (C cho LogisticRegression & alpha/learning_rate cho SGDClassifier - Lesson 4 & 5)
    print("\n[6] Thử nghiệm siêu tham số trên Tập Validation (N=7.500)...")
    tuning_configs = [
        ("LogisticRegression (C=0.01 - Phạt L2 mạnh)", LogisticRegression(max_iter=1000, C=0.01, random_state=RANDOM_SEED)),
        ("LogisticRegression (C=0.1)", LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_SEED)),
        ("LogisticRegression (C=1.0 - Mặc định)", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)),
        ("LogisticRegression (C=10.0 - Phạt L2 nhẹ)", LogisticRegression(max_iter=1000, C=10.0, random_state=RANDOM_SEED)),
        ("SGDClassifier (alpha=1e-2, lr='optimal')", SGDClassifier(loss="log_loss", alpha=1e-2, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
        ("SGDClassifier (alpha=1e-4, lr='optimal')", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
        ("SGDClassifier (alpha=1e-4, lr='constant', eta0=0.01)", SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.01, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
    ]
    tuning_rows = []
    for cfg_name, cfg_clf in tuning_configs:
        p = Pipeline([("preprocessor", preprocessor), ("classifier", cfg_clf)])
        t0 = time.time()
        p.fit(X_train, y_train)
        fit_t = time.time() - t0
        tr_res = evaluate(y_train, p.predict(X_train), get_model_score(p, X_train))
        va_res = evaluate(y_val, p.predict(X_val), get_model_score(p, X_val))
        tuning_rows.append(
            {
                "Cấu hình": cfg_name,
                "Train AUC": tr_res["ROC-AUC"],
                "Val AUC": va_res["ROC-AUC"],
                "Val F1-Macro": va_res["F1-Macro"],
                "Gap AUC (Train-Val)": tr_res["ROC-AUC"] - va_res["ROC-AUC"],
                "Fit Time (s)": fit_t,
            }
        )
    print(pd.DataFrame(tuning_rows).to_string(index=False))

    # 9. Đánh giá độc lập cuối cùng trên Tập Kiểm tra (Test Set - N=7.500)
    print(f"\n[7] Đánh giá trên Tập kiểm tra độc lập cuối cùng (Test Set - N={len(X_test):,})...")
    test_results = []
    y_scores_test = {}
    y_preds_test = {}
    infer_times_ms = {}

    for name, pipe in fitted_pipelines.items():
        t0 = time.time()
        score_test = get_model_score(pipe, X_test)
        pred_test = pipe.predict(X_test)
        infer_times_ms[name] = (time.time() - t0) * 1000.0 / len(X_test)

        y_scores_test[name] = score_test
        y_preds_test[name] = pred_test

        res = evaluate(y_test, pred_test, score_test)
        res["Mô hình"] = name
        test_results.append(res)
        print(
            f"    --> {name:34s} | Test AUC: {res['ROC-AUC']:.4f} | F1-Macro: {res['F1-Macro']:.4f} "
            f"| Acc: {res['Accuracy']:.4f} | Prec: {res['Precision']:.4f} | Rec: {res['Recall']:.4f} | Spec(0): {res['Specificity (Rec 0)']:.4f}"
        )

    df_test = pd.DataFrame(test_results)
    df_cv = pd.DataFrame(cv_results_summary)

    # 10. Lưu toàn bộ 7 biểu đồ trực quan
    img_dir = os.path.join(script_dir, "..", "images")
    os.makedirs(img_dir, exist_ok=True)

    def save_fig(fig, filename):
        fig.savefig(os.path.join(img_dir, filename), dpi=90, bbox_inches="tight")
        print(f"    [Đã lưu đồ thị]: {filename}")

    model_colors = ["#8c8c8c", "#ff7f0e", "#2ca02c", "#1f77b4", "#17becf"]
    short_labels = [
        "Dummy",
        "Perceptron (PLA)",
        "SGD (Log-loss)",
        "LogReg (Mặc định)",
        "LogReg (Balanced)",
    ]

    # --- Đồ thị 1: So sánh hiệu năng CV giữa 5 mô hình kèm thanh sai số std (yerr) ---
    fig1, ax1 = plt.subplots(figsize=(12.5, 5.8))
    mean_cols = ["Val AUC (mean)", "Val F1-Macro (mean)", "Val Acc (mean)", "Val Precision (mean)", "Val Recall (mean)"]
    std_cols = ["Val AUC (std)", "Val F1-Macro (std)", "Val Acc (std)", "Val Precision (std)", "Val Recall (std)"]
    labels_plot = ["ROC-AUC", "F1-Macro", "Accuracy", "Precision (Duyệt)", "Recall (Duyệt)"]

    x = np.arange(len(labels_plot))
    width = 0.15
    offsets = (np.arange(len(df_cv)) - (len(df_cv) - 1) / 2.0) * width

    for m_name, s_label, color, offset in zip(df_cv["Mô hình"], short_labels, model_colors, offsets):
        row = df_cv[df_cv["Mô hình"] == m_name].iloc[0]
        vals = [row[m] for m in mean_cols]
        errs = [row[s] for s in std_cols]
        rects = ax1.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=2,
            label=s_label,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax1.set_ylabel("Điểm số (Score)", fontsize=12, fontweight="bold")
    ax1.set_title("So sánh Kiểm định chéo 5-Fold trên Tập Train (N=35,000) - Mean ± Std", fontsize=13, fontweight="bold", pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_plot, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.90, color="red", linestyle="--", linewidth=1.2, label="Mục tiêu ROC-AUC (>= 0.90)")
    ax1.axhline(0.85, color="orange", linestyle=":", linewidth=1.2, label="Mục tiêu F1/Prec/Rec (>= 0.85)")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.8, ncol=3)
    plt.tight_layout()
    save_fig(fig1, "bieu_do_so_sanh_mo_hinh.png")
    plt.close(fig1)

    # --- Đồ thị 2: Ma trận nhầm lẫn trên Test Set (N=7.500) cho cả 5 mô hình ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(15.0, 9.2))
    axes_flat = axes2.flatten()

    for idx, (name, s_label) in enumerate(zip(models.keys(), short_labels)):
        ax = axes_flat[idx]
        cm = confusion_matrix(y_test, y_preds_test[name], labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([[f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
            xticklabels=["Từ chối (0)", "Duyệt (1)"],
            yticklabels=["Từ chối (0)", "Duyệt (1)"],
        )
        ax.set_title(f"{s_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Nhãn Dự đoán", fontsize=10, fontweight="bold")
        ax.set_ylabel("Nhãn Thực tế", fontsize=10, fontweight="bold")

    axes_flat[-1].axis("off")
    plt.suptitle("Ma trận nhầm lẫn của 5 mô hình (Lesson 4–5) trên Tập kiểm tra (N=7,500)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_fig(fig2, "ma_tran_nham_lan_cac_mo_hinh.png")
    plt.close(fig2)

    # --- Đồ thị 3: Đường cong ROC trên Test Set (N=7.500) ---
    fig3, ax3 = plt.subplots(figsize=(8.8, 6.5))
    for name, s_label, color in zip(models.keys(), short_labels, model_colors):
        score = y_scores_test[name]
        fpr, tpr, _ = roc_curve(y_test, score)
        auc_score = roc_auc_score(y_test, score)
        ax3.plot(fpr, tpr, color=color, lw=2, label=f"{s_label} (AUC = {auc_score:.4f})")

    ax3.plot([0, 1], [0, 1], color="navy", lw=1.4, linestyle="--", label="Ngẫu nhiên (AUC = 0.5000)")
    ax3.set_xlim([-0.01, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel("Tỷ lệ dương tính giả (False Positive Rate)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Tỷ lệ dương tính thật (True Positive Rate / Recall)", fontsize=11, fontweight="bold")
    ax3.set_title("Đường cong ROC so sánh các mô hình trên Tập kiểm tra (N=7,500)", fontsize=13, fontweight="bold", pad=12)
    ax3.legend(loc="lower right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    save_fig(fig3, "duong_cong_roc_so_sanh.png")
    plt.close(fig3)

    # --- Đồ thị 4: Đường cong Precision-Recall trên Test Set (N=7.500) ---
    fig4, ax4 = plt.subplots(figsize=(8.8, 6.5))
    for name, s_label, color in zip(models.keys(), short_labels, model_colors):
        score = y_scores_test[name]
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, score)
        ap_score = average_precision_score(y_test, score)
        ax4.plot(rec_vals, prec_vals, color=color, lw=2, label=f"{s_label} (AP = {ap_score:.4f})")

    ax4.set_xlim([0.0, 1.02])
    ax4.set_ylim([0.4, 1.05])
    ax4.set_xlabel("Độ phủ (Recall)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Độ chính xác (Precision)", fontsize=11, fontweight="bold")
    ax4.set_title("Đường cong Precision-Recall trên Tập kiểm tra (N=7,500)", fontsize=13, fontweight="bold", pad=12)
    ax4.legend(loc="lower left", fontsize=9.5, frameon=True)
    plt.tight_layout()
    save_fig(fig4, "duong_cong_precision_recall.png")
    plt.close(fig4)

    # --- Đồ thị 5: Trọng số có dấu (Signed Coefficients) của Logistic Regression & SGDClassifier ---
    lr_pipe = fitted_pipelines["Logistic Regression (Mặc định)"]
    sgd_pipe = fitted_pipelines["SGDClassifier (Log-loss + GD)"]

    encoded_feat_names = lr_pipe.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in encoded_feat_names]

    lr_coef = pd.Series(lr_pipe.named_steps["classifier"].coef_[0], index=clean_names)
    sgd_coef = pd.Series(sgd_pipe.named_steps["classifier"].coef_[0], index=clean_names)

    top_idx_lr = lr_coef.abs().sort_values(ascending=False).head(12).index
    top_idx_sgd = sgd_coef.abs().sort_values(ascending=False).head(12).index

    fig5, (ax5_1, ax5_2) = plt.subplots(1, 2, figsize=(15.5, 6.8))

    vals_lr = lr_coef.loc[top_idx_lr][::-1]
    colors_lr = ["#2ca02c" if v > 0 else "#d62728" for v in vals_lr]
    ax5_1.barh(vals_lr.index, vals_lr.values, color=colors_lr, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax5_1.axvline(0, color="black", linewidth=0.9)
    ax5_1.set_title("Top 12 trọng số w (Logistic Regression - L-BFGS)", fontsize=11.5, fontweight="bold")
    ax5_1.set_xlabel("Trọng số w (Xanh: tăng duyệt, Đỏ: tăng từ chối)", fontsize=10, fontweight="bold")

    vals_sgd = sgd_coef.loc[top_idx_sgd][::-1]
    colors_sgd = ["#2ca02c" if v > 0 else "#d62728" for v in vals_sgd]
    ax5_2.barh(vals_sgd.index, vals_sgd.values, color=colors_sgd, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax5_2.axvline(0, color="black", linewidth=0.9)
    ax5_2.set_title("Top 12 trọng số w (SGDClassifier - Gradient Descent)", fontsize=11.5, fontweight="bold")
    ax5_2.set_xlabel("Trọng số w (Xanh: tăng duyệt, Đỏ: tăng từ chối)", fontsize=10, fontweight="bold")

    plt.suptitle("Chiều hướng và độ lớn trọng số các đặc trưng trong mô hình tuyến tính (Đã chuẩn hóa Z-Score)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_fig(fig5, "do_quan_trong_dac_trung.png")
    plt.close(fig5)

    # --- Đồ thị 6: Phân tích lỗi trên Test Set (N=7.500) của Logistic Regression (Mặc định) ---
    lr_preds = y_preds_test["Logistic Regression (Mặc định)"]
    test_df_analysis = X_test.copy()
    test_df_analysis["true_label"] = y_test.values
    test_df_analysis["pred_label"] = lr_preds

    def get_prediction_category(row):
        if row["true_label"] == 1 and row["pred_label"] == 1:
            return "TP (Duyệt đúng)"
        elif row["true_label"] == 0 and row["pred_label"] == 0:
            return "TN (Từ chối đúng)"
        elif row["true_label"] == 0 and row["pred_label"] == 1:
            return "FP (Duyệt nhầm nợ xấu - Nguy hiểm)"
        return "FN (Từ chối nhầm khách tốt - Tiếc nuối)"

    test_df_analysis["cat"] = test_df_analysis.apply(get_prediction_category, axis=1)

    print("\n[8] Thống kê phân tích lỗi trên Tập kiểm tra (Logistic Regression Mặc định, N=7,500):")
    cat_counts = test_df_analysis["cat"].value_counts()
    for c, cnt in cat_counts.items():
        print(f"    - {c:42s}: {cnt:,} hồ sơ ({cnt/len(test_df_analysis):.2%})")

    print("\n    [Thống kê trung vị đặc trưng theo nhóm TP/TN/FP/FN - Logistic Regression]:")
    median_stats = test_df_analysis.groupby("cat")[["debt_to_income_ratio", "credit_score", "interest_rate", "annual_income", "years_employed"]].median()
    print(median_stats.to_string())

    fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))
    key_analysis_features = [
        ("debt_to_income_ratio", "Tỷ lệ Nợ trên Thu nhập (DTI)", axes6[0, 0]),
        ("credit_score", "Điểm tín dụng (Credit Score)", axes6[0, 1]),
        ("interest_rate", "Lãi suất đề nghị (Interest Rate %)", axes6[1, 0]),
        ("annual_income", "Thu nhập hàng năm (USD)", axes6[1, 1]),
    ]

    cat_order = [
        "TP (Duyệt đúng)",
        "TN (Từ chối đúng)",
        "FP (Duyệt nhầm nợ xấu - Nguy hiểm)",
        "FN (Từ chối nhầm khách tốt - Tiếc nuối)",
    ]
    palette = {
        "TP (Duyệt đúng)": "#2ca02c",
        "TN (Từ chối đúng)": "#1f77b4",
        "FP (Duyệt nhầm nợ xấu - Nguy hiểm)": "#d62728",
        "FN (Từ chối nhầm khách tốt - Tiếc nuối)": "#ff7f0e",
    }

    for feat, title, ax in key_analysis_features:
        sns.boxplot(data=test_df_analysis, x="cat", y=feat, order=cat_order, palette=palette, ax=ax, fliersize=2)
        ax.set_title(f"Phân bố {title} theo phân loại dự đoán", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.set_xticklabels(["TP", "TN", "FP (Lỗi)", "FN (Lỗi)"], fontsize=10, fontweight="bold")

    plt.suptitle("Phân tích lỗi (Error Analysis): Ranh giới tuyến tính của Logistic Regression (N=7,500)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_fig(fig6, "phan_tich_loi_fp_fn.png")
    plt.close(fig6)

    # --- Đồ thị 7: Learning Curve cho Logistic Regression & SGDClassifier ---
    print("\n[9] Tính toán Đường cong học tập (Learning Curve) cho Logistic Regression & SGDClassifier...")
    fig7, (ax7_1, ax7_2) = plt.subplots(1, 2, figsize=(14, 5.6))

    for ax, m_key, m_title, col_val in [
        (ax7_1, "Logistic Regression (Mặc định)", "Learning Curve: Logistic Regression (L-BFGS)", "#1f77b4"),
        (ax7_2, "SGDClassifier (Log-loss + GD)", "Learning Curve: SGDClassifier (Gradient Descent)", "#2ca02c"),
    ]:
        train_sizes, train_scores, val_scores = learning_curve(
            fitted_pipelines[m_key],
            X_train,
            y_train,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 6),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        va_mean, va_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, "o-", color="#555555", label="Huấn luyện (Train Score)")
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#555555")
        ax.plot(train_sizes, va_mean, "o-", color=col_val, label="Kiểm định chéo (CV Score)")
        ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color=col_val)
        ax.set_title(m_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Số lượng mẫu huấn luyện (Training Examples)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Điểm ROC-AUC", fontsize=10.5, fontweight="bold")
        ax.set_ylim(0.92, 0.96)
        ax.legend(loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    save_fig(fig7, "duong_cong_hoc_tap_learning_curve.png")
    plt.close(fig7)

    # 11. Lưu mô hình & bản tóm tắt JSON (dọn file mô hình Lesson 9 cũ nếu có)
    models_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    for stale in ["best_xgboost_pipeline.joblib", "gradient_boosting_pipeline.joblib"]:
        stale_p = os.path.join(models_dir, stale)
        if os.path.exists(stale_p):
            os.remove(stale_p)

    lr_path = os.path.join(models_dir, "best_logistic_pipeline.joblib")
    sgd_path = os.path.join(models_dir, "sgd_logistic_pipeline.joblib")
    joblib.dump(fitted_pipelines["Logistic Regression (Mặc định)"], lr_path)
    joblib.dump(fitted_pipelines["SGDClassifier (Log-loss + GD)"], sgd_path)
    print(f"\n[10] Đã lưu pipeline Logistic Regression tại: {lr_path}")
    print(f"     Đã lưu pipeline SGDClassifier tại: {sgd_path}")

    summary_json_path = os.path.join(models_dir, "experiment_summary.json")
    payload = {
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "cv_results": cv_results_summary,
        "val_results": val_results,
        "test_results": test_results,
        "threshold_analysis": threshold_analysis,
        "tuning_results": tuning_rows,
        "lr_coefficients": lr_coef.loc[top_idx_lr].to_dict(),
        "sgd_coefficients": sgd_coef.loc[top_idx_sgd].to_dict(),
        "error_counts_lr_test": {str(k): int(v) for k, v in cat_counts.items()},
        "error_medians_lr_test": median_stats.to_dict(orient="index"),
        "infer_ms_per_sample": infer_times_ms,
        "roc_ap_test": {
            name: {
                "auc": float(roc_auc_score(y_test, y_scores_test[name])),
                "ap": float(average_precision_score(y_test, y_scores_test[name])),
            }
            for name in models.keys()
        },
        "perceptron_n_iter": int(fitted_pipelines["Perceptron (PLA)"].named_steps["classifier"].n_iter_),
        "sgd_n_iter": int(fitted_pipelines["SGDClassifier (Log-loss + GD)"].named_steps["classifier"].n_iter_),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"     Đã lưu tổng hợp số liệu JSON tại: {summary_json_path}")

    print("\n" + "=" * 110)
    print("BẢNG TỔNG HỢP KẾT QUẢ KIỂM ĐỊNH CHÉO 5-FOLD (TRAIN SET N=35,000):")
    print("=" * 110)
    summary_table = [
        {
            "Mô hình": r["Mô hình"],
            "Train AUC": f"{r['Train AUC']:.4f}",
            "CV ROC-AUC": f"{r['Val AUC (mean)']:.4f} +/- {r['Val AUC (std)']:.4f}",
            "CV F1-Macro": f"{r['Val F1-Macro (mean)']:.4f} +/- {r['Val F1-Macro (std)']:.4f}",
            "CV Acc": f"{r['Val Acc (mean)']:.4f} +/- {r['Val Acc (std)']:.4f}",
            "CV Prec": f"{r['Val Precision (mean)']:.4f} +/- {r['Val Precision (std)']:.4f}",
            "CV Rec": f"{r['Val Recall (mean)']:.4f} +/- {r['Val Recall (std)']:.4f}",
            "Time (s)": f"{r['Thời gian (s)']:.2f}",
        }
        for r in cv_results_summary
    ]
    print(pd.DataFrame(summary_table).to_string(index=False))

    print("\n" + "=" * 110)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM ĐỊNH (VAL SET N=7,500):")
    print("=" * 110)
    print(df_val[["Mô hình", "ROC-AUC", "F1-Macro", "Accuracy", "Precision", "Recall", "Specificity (Rec 0)", "FP", "FN"]].to_string(index=False))

    print("\n" + "=" * 110)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM TRA ĐỘC LẬP (TEST SET N=7,500):")
    print("=" * 110)
    print(df_test[["Mô hình", "ROC-AUC", "F1-Macro", "Accuracy", "Precision", "Recall", "Specificity (Rec 0)", "TP", "TN", "FP", "FN"]].to_string(index=False))
    print("=" * 110)
    print("HOÀN TẤT THỰC NGHIỆM LESSON 04-05 VỚI TỶ LỆ 70% TRAIN - 15% VAL - 15% TEST!")


if __name__ == "__main__":
    main()
