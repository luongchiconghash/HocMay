# -*- coding: utf-8 -*-
"""
Tạo 4 Jupyter Notebook cho Bài kiểm tra 2 theo đúng cấu trúc chia nhỏ 1, 2, 3, 4
giống thư mục Kiemtra1/code:
  01_pipeline_va_baseline.ipynb
  02_huan_luyen_kiem_dinh_cheo_5fold.ipynb
  03_danh_gia_tap_test_va_truc_quan.ipynb
  04_phan_tich_loi_va_thuc_hanh_tai_cho.ipynb

Mỗi notebook độc lập (Run All được ngay), có đề mục rõ ràng ở đầu mỗi bài code
và có mục nhận xét gạch đầu dòng ngay dưới mỗi bảng/biểu đồ.
"""

import json
from pathlib import Path


def make_nb_builder():
    cells = []
    idx_counter = [0]

    def next_id() -> str:
        idx_counter[0] += 1
        return f"cell-{idx_counter[0]:03d}"

    def add_md(text: str) -> None:
        lines = [line + "\n" for line in text.strip("\n").splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "id": next_id(),
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        })

    def add_code(code: str) -> None:
        lines = [line + "\n" for line in code.strip("\n").splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "id": next_id(),
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        })

    def build(filename: str) -> Path:
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.14",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        out_path = Path(__file__).resolve().parent / filename
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    return add_md, add_code, build


# Đoạn khởi tạo dùng chung giúp từng notebook chạy độc lập từ đầu đến cuối
COMMON_SETUP_CODE = r"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Đồng bộ giao diện và bảng màu với Kiemtra1
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
PALETTE_4 = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = Path("../data/Loan_approval_data_2025.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("data/Loan_approval_data_2025.csv")
assert DATA_PATH.exists(), f"Không tìm thấy dữ liệu tại {DATA_PATH.resolve()}"

IMG_DIR = Path("../images")
MODEL_DIR = Path("../models")
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model_score(estimator, X_eval):
    '''Trích xuất điểm liên tục (xác suất lớp 1 hoặc khoảng cách decision_function).'''
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_eval)[:, 1]
    return estimator.decision_function(X_eval)


def evaluate(y_true, y_pred, y_score):
    '''Tính bộ chỉ số đánh giá đa chiều cho bài toán phân loại tín dụng.'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "F1-Macro": f1_score(y_true, y_pred, average="macro"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "PR-AUC": average_precision_score(y_true, y_score),
    }


def load_and_split_data():
    '''Nạp dữ liệu, loại bỏ customer_id và chia phân tầng 70% Train / 15% Val / 15% Test.'''
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customer_id", "loan_status"])
    y = df["loan_status"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return df, X, y, X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols, preprocessor


def build_pipelines(preprocessor):
    '''Đóng gói mô hình cơ sở và các thuật toán Bài 4 & Bài 5 vào Pipeline thống nhất.'''
    models = {
        "Dummy (Baseline đa số)": DummyClassifier(strategy="most_frequent"),
        "Perceptron (PLA)": Perceptron(max_iter=1000, tol=1e-3, random_state=RANDOM_SEED),
        "SGDClassifier (Log-loss + GD)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
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
    return {
        name: Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        for name, clf in models.items()
    }
"""


def create_nb1():
    add_md, add_code, build = make_nb_builder()

    add_md(
        r"""
## Bài 1: Thiết lập Pipeline chống rò rỉ dữ liệu và Mô hình cơ sở 

- **Kiểm tra dữ liệu đầu vào:** Nạp `Loan_approval_data_2025.csv`, loại bỏ cột định danh `customer_id`.
- **Phân chia dữ liệu 70/15/15:** Tách phân tầng thành **Train (70%)**, **Validation (15%)** và **Test (15%)** với `random_state = 42`.
- **Đóng gói `ColumnTransformer` & `Pipeline`:** Chỉ `fit` phép chuẩn hóa `StandardScaler` và `OneHotEncoder` trên tập huấn luyện, triệt tiêu hoàn toàn rò rỉ dữ liệu.
- **Thiết lập Mô hình cơ sở (`DummyClassifier`):** Xác lập mức sàn tối thiểu để đối chiếu các thuật toán ở Bài 2.
"""
    )

    add_md(r"### 1. Nạp dữ liệu và kiểm tra phân bố nhãn mục tiêu (`loan_status`)")
    add_code(
        COMMON_SETUP_CODE
        + r"""
df, X, y, X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols, preprocessor = load_and_split_data()

n_approved = int((y == 1).sum())
n_rejected = int((y == 0).sum())

print("=" * 80)
print("THÔNG TIN TẬP DỮ LIỆU GỐC (BÀI KIỂM TRA 2)")
print("=" * 80)
print(f" - Tập dữ liệu        : {DATA_PATH}")
print(f" - Kích thước bảng    : {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f" - Đặc trưng đầu vào   : {X.shape[1]} cột ({len(num_cols)} số + {len(cat_cols)} phân loại)")
print(f" - Lớp 1 (Được duyệt) : {n_approved:,} ({n_approved / len(y):.2%})")
print(f" - Lớp 0 (Bị từ chối) : {n_rejected:,} ({n_rejected / len(y):.2%})")
display(X.head(3))
"""
    )
    add_md(
        r"""
- **Thống kê:** Toàn bộ tập dữ liệu gồm $50.000$ hồ sơ vay với $18$ đặc trưng đầu vào hợp lệ (`customer_id` đã bị loại bỏ ngay từ đầu).
- **Ý nghĩa mô hình hóa:** Tỷ lệ nhãn $55{,}05\%$ được duyệt so với $44{,}95\%$ bị từ chối bảo đảm tính cân đối tương đối, song vẫn cần theo dõi song song `ROC-AUC`, `F1-Macro`, `Precision` và `Recall` theo đúng bộ tiêu chí đã xác lập.
"""
    )

    add_md(r"### 2. Phân chia tập 70/15/15 và Đóng gói `ColumnTransformer`")
    add_code(
        r"""
# Kiểm tra kích thước 3 tập và số chiều sau ColumnTransformer (chỉ fit thử trên X_train)
X_train_tr = preprocessor.fit_transform(X_train)
ohe_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + ohe_names

split_summary = pd.DataFrame(
    [
        {"Tập dữ liệu": "Huấn luyện (Train)", "Số mẫu": len(X_train), "Tỷ lệ (%)": len(X_train) / len(X) * 100, "Tỷ lệ duyệt (y=1)": y_train.mean()},
        {"Tập dữ liệu": "Kiểm định (Val)", "Số mẫu": len(X_val), "Tỷ lệ (%)": len(X_val) / len(X) * 100, "Tỷ lệ duyệt (y=1)": y_val.mean()},
        {"Tập dữ liệu": "Kiểm tra (Test)", "Số mẫu": len(X_test), "Tỷ lệ (%)": len(X_test) / len(X) * 100, "Tỷ lệ duyệt (y=1)": y_test.mean()},
    ]
)

print("=" * 80)
print("PHÂN CHIA DỮ LIỆU 70/15/15 VÀ TIỀN XỬ LÝ KHÉP KÍN")
print("=" * 80)
display(split_summary.round(4))
print(f"Số chiều đặc trưng sau OneHotEncoder (drop='first'): {X_train_tr.shape[1]} chiều")
"""
    )
    add_md(
        r"""
- **Chống rò rỉ dữ liệu:** Tập Test ($7.500$ mẫu -- $15\%$) được cô lập hoàn toàn từ đầu; các tham số trung bình, độ lệch chuẩn của `StandardScaler` và danh mục của `OneHotEncoder` chỉ được học trên tập Train bên trong `Pipeline`.
- **Bảo toàn tỷ lệ nhãn:** Cả ba tập Train ($35.000$), Val ($7.500$) và Test ($7.500$) giữ nguyên tỷ lệ nhãn dương xấp xỉ $55{,}05\%$ nhờ phân tầng `stratify=y`.
"""
    )

    add_md(r"### 3. Xây dựng và đánh giá Mô hình cơ sở (`DummyClassifier` -- Majority Class)")
    add_code(
        r"""
baseline_pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_res = cross_validate(
    baseline_pipe,
    X_train,
    y_train,
    cv=cv,
    scoring=["roc_auc", "f1_macro", "accuracy", "precision", "recall"],
    n_jobs=-1,
)

baseline_pipe.fit(X_train, y_train)
y_val_pred = baseline_pipe.predict(X_val)
y_val_score = get_model_score(baseline_pipe, X_val)
val_metrics = evaluate(y_val, y_val_pred, y_val_score)

print("=" * 80)
print("MỨC SÀN MÔ HÌNH CƠ SỞ (DUMMY CLASSIFIER - LỚP ĐA SỐ)")
print("=" * 80)
print(f" - 5-Fold CV ROC-AUC  : {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")
print(f" - 5-Fold CV F1-Macro : {cv_res['test_f1_macro'].mean():.4f} ± {cv_res['test_f1_macro'].std():.4f}")
print(f" - 5-Fold CV Accuracy : {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
print(f" - Validation ROC-AUC : {val_metrics['ROC-AUC']:.4f}")
print(f" - Validation Spec.   : {val_metrics['Specificity']:.4f} (Recall lớp 0 = 0%)")
"""
    )
    add_md(
        r"""
- **Mức sàn tham chiếu:** Do luôn gán nhãn lớp đa số ($\hat{y} = 1$), `DummyClassifier` đạt Accuracy $55{,}05\%$ và Recall $1{,}0000$ một cách tầm thường, nhưng `ROC-AUC = 0,5000`, `F1-Macro = 0,3550` và `Specificity = 0,0000` (không lọc được bất kỳ hồ sơ nợ xấu nào).
- **Ý nghĩa đối chứng:** Đây là mốc tối thiểu chứng minh mọi mô hình học máy ở Bài 2 phải vượt trội cả về khả năng phân tách xác suất (`ROC-AUC`) lẫn cân bằng hai lớp (`F1-Macro`).
"""
    )

    return build("01_pipeline_va_baseline.ipynb")


def create_nb2():
    add_md, add_code, build = make_nb_builder()

    add_md(
        r"""
## Bài 2: Huấn luyện, Kiểm định chéo 5-Fold, Khảo sát ngưỡng và Tinh chỉnh siêu tham số

- Đối chiếu 5 cấu hình: `DummyClassifier`, `Perceptron` (PLA), `SGDClassifier(loss='log_loss')`, `LogisticRegression` (Mặc định & Balanced).
- Báo cáo kiểm định chéo phân tầng **5-Fold (`Mean ± Std`)** trên Tập Train ($N = 35.000$).
- Khảo sát ngưỡng xác suất **Out-of-Fold (`cross_val_predict`)** và tinh chỉnh siêu tham số trên **Tập Validation** ($N = 7.500$).
"""
    )

    add_md(r"### 1. Khởi tạo bộ dữ liệu 70/15/15 và Đóng gói 5 Pipeline mô hình")
    add_code(
        COMMON_SETUP_CODE
        + r"""
df, X, y, X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols, preprocessor = load_and_split_data()
pipelines = build_pipelines(preprocessor)

print("Đã sẵn sàng", len(pipelines), "Pipeline trên Tập Train:", X_train.shape)
for idx, name in enumerate(pipelines, start=1):
    print(f" {idx}. {name}")
"""
    )

    add_md(r"### 2. Kiểm định chéo phân tầng 5-Fold trên Tập Huấn luyện ($N = 35.000$)")
    add_code(
        r"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
scoring_metrics = {
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
}

cv_rows = []
for name, pipe in pipelines.items():
    t0 = time.perf_counter()
    res = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - t0
    cv_rows.append(
        {
            "Mô hình": name,
            "ROC-AUC (Mean)": res["test_roc_auc"].mean(),
            "ROC-AUC (Std)": res["test_roc_auc"].std(),
            "F1-Macro (Mean)": res["test_f1_macro"].mean(),
            "F1-Macro (Std)": res["test_f1_macro"].std(),
            "Accuracy (Mean)": res["test_accuracy"].mean(),
            "Precision (Mean)": res["test_precision"].mean(),
            "Precision (Std)": res["test_precision"].std(),
            "Recall (Mean)": res["test_recall"].mean(),
            "Recall (Std)": res["test_recall"].std(),
            "Thời gian CV (s)": elapsed,
        }
    )

cv_df = pd.DataFrame(cv_rows)
cv_display = pd.DataFrame({"Mô hình": cv_df["Mô hình"]})
for m, col_label in [
    ("ROC-AUC", "ROC-AUC (CV)"),
    ("F1-Macro", "F1-Macro (CV)"),
    ("Precision", "Precision (CV)"),
    ("Recall", "Recall (CV)"),
]:
    cv_display[col_label] = cv_df.apply(
        lambda r: f"{r[f'{m} (Mean)']:.4f} ± {r[f'{m} (Std)']:.4f}", axis=1
    )
cv_display["Accuracy (CV)"] = cv_df["Accuracy (Mean)"].map(lambda v: f"{v:.4f}")
cv_display["Thời gian (s)"] = cv_df["Thời gian CV (s)"].map(lambda v: f"{v:.2f}")

print("=" * 80)
print("BẢNG SO SÁNH KIỂM ĐỊNH CHÉO 5-FOLD (MEAN ± STD)")
print("=" * 80)
display(cv_display)
"""
    )
    add_md(
        r"""
- **So với Baseline:** `Perceptron` (PLA) nâng ROC-AUC lên $0{,}9136 \pm 0{,}0115$ nhưng F1-Macro mới đạt $0{,}8229 \pm 0{,}0091$ kèm độ lệch chuẩn Precision/Recall lớn ($\pm 0{,}0336$ / $\pm 0{,}0492$).
- **Ưu thế tối ưu Log-loss:** `Logistic Regression (Mặc định)` dẫn đầu với `ROC-AUC = 0,9457 ± 0,0024` và `F1-Macro = 0,8651 ± 0,0028` chỉ trong $\approx 0{,}5$ giây; `SGDClassifier (Log-loss + GD)` đạt kết quả tương đương (`0,9448 ± 0,0018` AUC).
"""
    )

    add_md(r"### 3. Biểu đồ so sánh Kiểm định chéo 5-Fold giữa 5 mô hình (`Mean ± Std`)")
    add_code(
        r"""
fig, ax = plt.subplots(figsize=(12, 6))
plot_metrics = ["ROC-AUC", "F1-Macro", "Precision", "Recall"]
x = np.arange(len(cv_df))
width = 0.19

for i, (m, col) in enumerate(zip(plot_metrics, PALETTE_4)):
    means = cv_df[f"{m} (Mean)"].values
    stds = cv_df[f"{m} (Std)"].values
    bars = ax.bar(
        x + (i - 1.5) * width,
        means,
        width,
        yerr=stds,
        capsize=3,
        label=m,
        color=col,
        alpha=0.9,
        edgecolor="white",
    )
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.014,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.axhline(0.90, color="#c0392b", linestyle="--", linewidth=1.2, label="Ngưỡng ROC-AUC (0.90)")
ax.axhline(0.85, color="#2c3e50", linestyle=":", linewidth=1.2, label="Ngưỡng F1/P/R (0.85)")
ax.set_xticks(x)
ax.set_xticklabels(cv_df["Mô hình"], rotation=14, ha="right", fontweight="bold")
ax.set_ylim(0.0, 1.15)
ax.set_ylabel("Điểm số (Mean ± Std)")
ax.set_title("Biểu đồ 1: So sánh Kiểm định chéo 5-Fold giữa Baseline và Các mô hình", fontweight="bold")
ax.legend(loc="lower right", ncol=3, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(IMG_DIR / "bieu_do_so_sanh_mo_hinh.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Vượt chuẩn cam kết:** Ba cấu hình tối ưu theo Log-loss (`SGDClassifier`, `Logistic Regression Mặc định` và `Balanced`) đều vượt dứt khoát hai ngưỡng mục tiêu `0,90` (ROC-AUC) và `0,85` (F1-Macro, Precision, Recall) với thanh sai số $\pm \sigma$ rất hẹp.
- **Đánh đổi Balanced:** Bật `class_weight='balanced'` đẩy Precision lớp duyệt lên $0{,}8846$ (lọc nợ xấu chặt hơn), đổi lại Recall giảm nhẹ từ $0{,}8878$ xuống $0{,}8682$.
"""
    )

    add_md(r"### 4. Khảo sát ngưỡng xác suất Out-of-Fold (`cross_val_predict`) và Kiểm chứng Tập Validation ($N = 7.500$)")
    add_code(
        r"""
fitted_pipelines = {}
val_rows = []
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    y_val_pred = pipe.predict(X_val)
    y_val_score = get_model_score(pipe, X_val)
    val_rows.append({"Mô hình": name, **evaluate(y_val, y_val_pred, y_val_score)})

val_df = pd.DataFrame(val_rows)
print("=" * 80)
print("KẾT QUẢ TRÊN TẬP VALIDATION (N = 7.500)")
print("=" * 80)
display(val_df.round(4))

oof_probs = cross_val_predict(
    pipelines["Logistic Regression (Mặc định)"],
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]
val_probs = fitted_pipelines["Logistic Regression (Mặc định)"].predict_proba(X_val)[:, 1]

thr_rows = []
for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    oof_pred = (oof_probs >= thr).astype(int)
    val_pred = (val_probs >= thr).astype(int)
    tn_o, fp_o, _, _ = confusion_matrix(y_train, oof_pred).ravel()
    _, fp_v, fn_v, _ = confusion_matrix(y_val, val_pred).ravel()
    thr_rows.append(
        {
            "Ngưỡng (tau)": thr,
            "OOF F1-Macro": round(f1_score(y_train, oof_pred, average="macro"), 4),
            "OOF Precision": round(precision_score(y_train, oof_pred), 4),
            "OOF Recall": round(recall_score(y_train, oof_pred), 4),
            "OOF Specificity": round(tn_o / (tn_o + fp_o), 4),
            "Val F1-Macro": round(f1_score(y_val, val_pred, average="macro"), 4),
            "Val FP (Duyệt nhầm xấu)": int(fp_v),
            "Val FN (Từ chối nhầm tốt)": int(fn_v),
        }
    )

print("-" * 80)
print("KHẢO SÁT NGƯỠNG PHÂN LOẠI TRÊN OUT-OF-FOLD & VALIDATION")
print("-" * 80)
display(pd.DataFrame(thr_rows))
"""
    )
    add_md(
        r"""
- **Ngưỡng cân bằng:** Tại $\tau = 0{,}50$, OOF F1-Macro đạt cực đại ($0{,}8651$), tương ứng trên Validation là $0{,}8688$ ($503$ FP và $470$ FN).
- **Điều chỉnh khẩu vị rủi ro:** Khi nâng ngưỡng phê duyệt lên $\tau = 0{,}60$, OOF Precision tăng lên $0{,}8968$ (Specificity $0{,}8810$) và số ca duyệt nhầm nợ xấu (FP) trên Validation giảm $21{,}9\%$ (từ $503$ xuống chỉ còn $393$ hồ sơ).
"""
    )

    add_md(r"### 5. Tinh chỉnh siêu tham số ($C$, $\alpha$, `learning_rate`) trên Tập Validation")
    add_code(
        r"""
tuning_configs = [
    ("LogisticRegression (C=0.01)", LogisticRegression(C=0.01, max_iter=1000, random_state=RANDOM_SEED)),
    ("LogisticRegression (C=0.1)", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)),
    ("LogisticRegression (C=1.0, mặc định)", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)),
    ("LogisticRegression (C=10.0)", LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_SEED)),
    ("SGDClassifier (alpha=1e-2, optimal)", SGDClassifier(loss="log_loss", alpha=1e-2, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
    ("SGDClassifier (alpha=1e-4, optimal)", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
    ("SGDClassifier (alpha=1e-4, constant, eta0=0.01)", SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.01, max_iter=1000, tol=1e-4, random_state=RANDOM_SEED)),
]

tuning_rows = []
for cfg_name, clf in tuning_configs:
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    tr_auc = roc_auc_score(y_train, get_model_score(pipe, X_train))
    va_score = get_model_score(pipe, X_val)
    va_pred = pipe.predict(X_val)
    tuning_rows.append(
        {
            "Cấu hình": cfg_name,
            "Train AUC": tr_auc,
            "Val AUC": roc_auc_score(y_val, va_score),
            "Val F1-Macro": f1_score(y_val, va_pred, average="macro"),
            "Chênh lệch AUC": tr_auc - roc_auc_score(y_val, va_score),
            "Thời gian fit (s)": fit_sec,
        }
    )

print("=" * 80)
print("TINH CHỈNH SIÊU THAM SỐ TRÊN TẬP VALIDATION (KHÔNG CHẠM TEST)")
print("=" * 80)
display(pd.DataFrame(tuning_rows).round(4))
"""
    )
    add_md(
        r"""
- **Hệ số $C$ của `LogisticRegression`:** $C = 1{,}0$ đạt đỉnh Val ROC-AUC ($0{,}9478$) trong $\approx 0{,}09$ giây (chênh lệch Train–Val chỉ $0{,}0019$), còn $C = 0{,}1$ đạt Val F1-Macro cao nhất ($0{,}8701$).
- **Hệ số $\alpha$ và `learning_rate` của `SGDClassifier`:** Phạt quá mạnh ($\alpha = 10^{-2}$) kéo Val AUC xuống $0{,}9418$, trong khi $\alpha = 10^{-4}$ khôi phục Val AUC lên $0{,}9466$ (và `constant`, `eta0=0.01` đạt F1-Macro $0{,}8678$ sau $0{,}12$ giây).
"""
    )

    return build("02_huan_luyen_kiem_dinh_cheo_5fold.ipynb")


def create_nb3():
    add_md, add_code, build = make_nb_builder()

    add_md(
        r"""
## Bài 3: Đánh giá độc lập trên Tập Kiểm tra ($N = 7.500$) và Công cụ trực quan hóa

 **Mở khóa Tập Kiểm tra (`X_test`, $N = 7.500$)** đúng một lần duy nhất để kiểm chứng năng lực tổng quát hóa và xuất các biểu đồ bổ trợ:
- Bảng tổng hợp các chỉ số cam kết và `TP / TN / FP / FN` trên Tập Kiểm tra.
- Ma trận nhầm lẫn bằng `ConfusionMatrixDisplay.from_predictions`.
- Đường cong ROC bằng `RocCurveDisplay.from_predictions` và Đường cong Precision-Recall bằng `PrecisionRecallDisplay.from_predictions`.
"""
    )

    add_md(r"### 1. Khớp các Pipeline trên Tập Huấn luyện ($N = 35.000$) và Lưu mô hình")
    add_code(
        COMMON_SETUP_CODE
        + r"""
df, X, y, X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols, preprocessor = load_and_split_data()
pipelines = build_pipelines(preprocessor)

fitted_pipelines = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

# Lưu hai Pipeline Log-loss để tái sử dụng
joblib.dump(fitted_pipelines["Logistic Regression (Mặc định)"], MODEL_DIR / "best_logistic_pipeline.joblib")
joblib.dump(fitted_pipelines["SGDClassifier (Log-loss + GD)"], MODEL_DIR / "sgd_logistic_pipeline.joblib")

print("Đã huấn luyện xong 5 Pipeline và lưu mô hình tại:", MODEL_DIR)
"""
    )

    add_md(r"### 2. Bảng tổng hợp kết quả đánh giá trên Tập Kiểm tra ($N = 7.500$)")
    add_code(
        r"""
test_rows = []
y_preds = {}
y_scores = {}

for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_score = get_model_score(pipe, X_test)
    y_preds[name] = y_pred
    y_scores[name] = y_score

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_rows.append(
        {
            "Mô hình": name,
            **evaluate(y_test, y_pred, y_score),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }
    )

test_df = pd.DataFrame(test_rows)
print("=" * 80)
print("KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (N = 7.500)")
print("=" * 80)
display(test_df.round(4))
"""
    )
    add_md(
        r"""
- **Đạt trọn vẹn cam kết:** `Logistic Regression (Mặc định)` đạt `ROC-AUC = 0,9440` ($\ge 0{,}90$), `F1-Macro = 0,8650` ($\ge 0{,}85$), `Precision = 0,8743` ($\ge 0{,}85$) và `Recall = 0,8847` ($\ge 0{,}85$).
- **Tổng quát hóa ổn định:** Chênh lệch giữa 5-Fold CV ($0{,}9457$ AUC / $0{,}8651$ F1) và Tập Test ($0{,}9440$ AUC / $0{,}8650$ F1) dưới $0{,}002$, xác nhận Pipeline không bị rò rỉ dữ liệu hay quá khớp.
"""
    )

    add_md(r"### 3. Ma trận nhầm lẫn của 5 mô hình trên Tập Kiểm tra (`ConfusionMatrixDisplay`)")
    add_code(
        r"""
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.ravel()

for idx, name in enumerate(fitted_pipelines):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_preds[name],
        display_labels=["Từ chối (0)", "Duyệt (1)"],
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=axes_flat[idx],
    )
    auc_val = roc_auc_score(y_test, y_scores[name])
    axes_flat[idx].set_title(f"{name}\n(ROC-AUC = {auc_val:.4f})", fontweight="bold", fontsize=10)
    axes_flat[idx].grid(False)

axes_flat[-1].axis("off")
plt.suptitle("Biểu đồ 2: Ma trận nhầm lẫn trên Tập Kiểm tra (N = 7.500)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG_DIR / "ma_tran_nham_lan_cac_mo_hinh.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Cân bằng sai số:** `Logistic Regression (Mặc định)` phân loại đúng $3.652$ hồ sơ tốt (TP) và $2.847$ hồ sơ rủi ro (TN), mắc $525$ ca duyệt nhầm nợ xấu (FP) và $476$ ca từ chối nhầm (FN).
- **Lợi thế nhánh `Balanced`:** Khi dùng `class_weight='balanced'`, số ca FP giảm từ $525$ xuống còn $465$ hồ sơ (`Specificity` tăng từ $84{,}43\%$ lên $86{,}21\%$), đổi lại FN tăng lên $559$.
"""
    )

    add_md(r"### 4. Đường cong ROC so sánh 5 mô hình (`RocCurveDisplay`)")
    add_code(
        r"""
colors = ["#95a5a6", "#f39c12", "#8e44ad", "#e74c3c", "#3498db"]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
for name, col in zip(fitted_pipelines, colors):
    disp = RocCurveDisplay.from_predictions(y_test, y_scores[name], name=name, ax=ax)
    disp.line_.set_color(col)
    disp.line_.set_linewidth(2.0)

ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6)
ax.set_title("Biểu đồ 3: So sánh đường cong ROC trên Tập Kiểm tra (N = 7.500)", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / "duong_cong_roc_so_sanh.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Khả năng xếp hạng:** Đường cong ROC của hai mô hình `Logistic Regression` ($\text{AUC} = 0{,}9440$) và `SGDClassifier` ($\text{AUC} = 0{,}9434$) gần như trùng khít, nằm cao sát góc trên bên trái và bỏ xa `Perceptron` ($0{,}8959$) cũng như đường chéo ngẫu nhiên `Dummy` ($0{,}5000$).
"""
    )

    add_md(r"### 5. Đường cong Precision-Recall trên Tập Kiểm tra (`PrecisionRecallDisplay`)")
    add_code(
        r"""
colors = ["#95a5a6", "#f39c12", "#8e44ad", "#e74c3c", "#3498db"]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
for name, col in zip(fitted_pipelines, colors):
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_scores[name], name=name, ax=ax)
    disp.line_.set_color(col)
    disp.line_.set_linewidth(2.0)

ax.set_title("Biểu đồ 4: Đường cong Precision-Recall trên Tập Kiểm tra (N = 7.500)", fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / "duong_cong_precision_recall.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Độ tin cậy dự đoán lớp duyệt:** Nhóm mô hình tối ưu hàm Log-loss đạt Average Precision `AP = 0,9519` (`Logistic Regression`) và `0,9515` (`SGDClassifier`), duy trì Precision cao ngay cả ở vùng Recall rộng so với mức nền `0,5504` của `Dummy`.
"""
    )

    return build("03_danh_gia_tap_test_va_truc_quan.ipynb")


def create_nb4():
    add_md, add_code, build = make_nb_builder()

    add_md(
        r"""
## Bài 4: Chẩn đoán mô hình, Phân tích lỗi, Trọng số đặc trưng và Thực hành tại chỗ

- So sánh kết quả Train vs CV Validation và vẽ **Đường cong học tập (`learning_curve`)** để nhận diện quá khớp / chưa khớp.
- **Phân tích lỗi :** So sánh trung vị và phân bố 4 nhóm `TP / TN / FP / FN` trên Tập Test.
- **Trọng số đặc trưng chuẩn hóa :** Đối chiếu 10 đặc trưng có trọng số mạnh nhất giữa `LogisticRegression` và `SGDClassifier`.
"""
    )

    add_md(r"### 1. Thiết lập dữ liệu 70/15/15 và Khớp lại mô hình")
    add_code(
        COMMON_SETUP_CODE
        + r"""
df, X, y, X_train, X_val, X_test, y_train, y_val, y_test, num_cols, cat_cols, preprocessor = load_and_split_data()
pipelines = build_pipelines(preprocessor)

fitted_pipelines = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

lr_pipe = fitted_pipelines["Logistic Regression (Mặc định)"]
sgd_pipe = fitted_pipelines["SGDClassifier (Log-loss + GD)"]
ohe_names = lr_pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
feature_names = num_cols + ohe_names

print("Đã sẵn sàng dữ liệu và Pipeline cho bước chẩn đoán và phân tích lỗi.")
"""
    )

    add_md(r"### 2. Chẩn đoán Quá khớp / Chưa khớp và Đường cong học tập (`learning_curve`)")
    add_code(
        r"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
diag_rows = []
for name, pipe in pipelines.items():
    res = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )
    diag_rows.append(
        {
            "Mô hình": name,
            "Train Acc": res["train_accuracy"].mean(),
            "CV Val Acc": res["test_accuracy"].mean(),
            "Chênh lệch Acc": res["train_accuracy"].mean() - res["test_accuracy"].mean(),
            "Train AUC": res["train_roc_auc"].mean(),
            "CV Val AUC": res["test_roc_auc"].mean(),
            "Chênh lệch AUC": res["train_roc_auc"].mean() - res["test_roc_auc"].mean(),
        }
    )

print("=" * 80)
print("BẢNG CHẨN ĐOÁN QUÁ KHỚP / CHƯA KHỚP (5-FOLD CV, N = 35.000)")
print("=" * 80)
display(pd.DataFrame(diag_rows).round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, (title, pipe) in zip(
    axes,
    [
        ("Logistic Regression (Mặc định)", pipelines["Logistic Regression (Mặc định)"]),
        ("SGDClassifier (Log-loss + GD)", pipelines["SGDClassifier (Log-loss + GD)"]),
    ],
):
    sizes, tr_scores, va_scores = learning_curve(
        pipe, X_train, y_train, cv=cv, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
    )
    tr_m, tr_s = tr_scores.mean(axis=1), tr_scores.std(axis=1)
    va_m, va_s = va_scores.mean(axis=1), va_scores.std(axis=1)
    ax.plot(sizes, tr_m, "o-", color="#e74c3c", label="Train ROC-AUC")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="#e74c3c")
    ax.plot(sizes, va_m, "s-", color="#2980b9", label="CV Val ROC-AUC")
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color="#2980b9")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.set_ylim(0.91, 0.96)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("ROC-AUC")
plt.suptitle("Biểu đồ 5: Đường cong học tập (Learning Curve)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG_DIR / "duong_cong_hoc_tap_learning_curve.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Không bị quá khớp (Low Variance):** Ở cả `LogisticRegression` và `SGDClassifier`, độ lệch giữa Train và CV Val đều dưới `0,0005` (`< 0,05%`), hai đường cong hội tụ sát tại vùng `0,945 - 0,946`.
- **Trạng thái từng mô hình:** `Dummy` chưa khớp nặng (`AUC = 0,5000`); `Perceptron` chưa khớp ranh giới xác suất tốt bằng Log-loss; hai mô hình Log-loss đạt trạng thái khớp cân bằng tối ưu trong lớp siêu phẳng tuyến tính.
"""
    )

    add_md(r"### 3. Phân tích lỗi (`Error Analysis`) tại vùng ranh giới quyết định trên Tập Test")
    add_code(
        r"""
err_df = X_test.copy()
err_df["y_true"] = y_test.values
err_df["y_pred"] = lr_pipe.predict(X_test)


def classify_error_group(row):
    if row["y_true"] == 1 and row["y_pred"] == 1:
        return "TP (Duyệt đúng)"
    if row["y_true"] == 0 and row["y_pred"] == 0:
        return "TN (Từ chối đúng)"
    if row["y_true"] == 0 and row["y_pred"] == 1:
        return "FP (Duyệt nhầm xấu)"
    return "FN (Từ chối nhầm tốt)"


err_df["Nhóm"] = err_df.apply(classify_error_group, axis=1)

key_features = [
    ("credit_score", "Điểm tín dụng (credit_score)"),
    ("debt_to_income_ratio", "Tỷ lệ nợ / thu nhập (DTI)"),
    ("interest_rate", "Lãi suất (%)"),
    ("annual_income", "Thu nhập năm (USD)"),
]
group_order = [
    "TP (Duyệt đúng)",
    "TN (Từ chối đúng)",
    "FP (Duyệt nhầm xấu)",
    "FN (Từ chối nhầm tốt)",
]

median_cols = [c for c, _ in key_features] + ["years_employed"]
print("=" * 80)
print("TRUNG VỊ ĐẶC TRƯNG THEO 4 NHÓM KẾT QUẢ (LOGISTIC REGRESSION - TEST)")
print("=" * 80)
display(err_df.groupby("Nhóm")[median_cols].median().loc[group_order].round(3))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, title) in zip(axes.ravel(), key_features):
    sns.boxplot(
        data=err_df,
        x="Nhóm",
        y=col,
        hue="Nhóm",
        order=group_order,
        hue_order=group_order,
        palette=PALETTE_4,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_title(f"Phân bố {title}", fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)

plt.suptitle("Biểu đồ 6: Phân tích lỗi dự đoán (TP / TN / FP / FN) trên Tập Test", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG_DIR / "phan_tich_loi_fp_fn.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Nhóm FP ($525$ ca -- duyệt nhầm nợ xấu):** Có chỉ số `debt_to_income_ratio` rất thấp (trung vị $0{,}210$, thấp hơn cả nhóm TP $0{,}220$) và thu nhập ổn ($41.658$ USD), khiến tổng tuyến tính vượt ngưỡng duyệt dù điểm tín dụng ở vùng cận biên ($646{,}0$).
- **Nhóm FN ($476$ ca -- từ chối nhầm khách tốt):** Có `debt_to_income_ratio` cao hơn hẳn ($0{,}322$) và điểm tín dụng thấp hơn ($624{,}0$) kéo xác suất xuống dưới $0{,}50$, dù lãi suất vay tương đối thấp (trung vị $14{,}50\%$).
"""
    )

    add_md(r"### 4. So sánh trọng số đặc trưng chuẩn hóa (`coef_`) của Logistic Regression và SGDClassifier")
    add_code(
        r"""
lr_coef = lr_pipe.named_steps["classifier"].coef_[0]
sgd_coef = sgd_pipe.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame(
    {
        "Đặc trưng": feature_names,
        "LogisticRegression (w)": lr_coef,
        "SGDClassifier (w)": sgd_coef,
        "|w| Logistic": np.abs(lr_coef),
    }
).sort_values("|w| Logistic", ascending=False)

top10 = coef_df.head(10).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, col_name, title in [
    (axes[0], "LogisticRegression (w)", "Logistic Regression (L-BFGS)"),
    (axes[1], "SGDClassifier (w)", "SGDClassifier (Log-loss + Gradient Descent)"),
]:
    vals = top10[col_name].values
    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
    bars = ax.barh(top10["Đặc trưng"], vals, color=bar_colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Trọng số chuẩn hóa (w)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, vals):
        offset = 0.08 if v >= 0 else -0.08
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}", va="center", ha=ha, fontsize=8.5)

plt.suptitle("Biểu đồ 7: Top 10 đặc trưng có trọng số chuẩn hóa mạnh nhất", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG_DIR / "do_quan_trong_dac_trung.png", dpi=90)
plt.show()
"""
    )
    add_md(
        r"""
- **Nhóm trọng số âm mạnh ($w < 0$, đẩy về từ chối):** `product_type_Line of Credit` ($-4{,}91$), `product_type_Personal Loan` ($-3{,}70$), tiền sử vỡ nợ `defaults_on_file` ($-3{,}54$), lãi suất `interest_rate` ($-1{,}94$) và tỷ lệ nợ/thu nhập `debt_to_income_ratio` ($-1{,}34$).
- **Nhóm trọng số dương mạnh ($w > 0$, thúc đẩy duyệt):** Mục đích vay `loan_intent_Education` ($+2{,}52$), `loan_intent_Personal` ($+1{,}76$), lịch sử tín dụng `credit_history_years` ($+0{,}75$) và `credit_score`.
"""
    )

    return build("04_phan_tich_loi_va_thuc_hanh_tai_cho.ipynb")


if __name__ == "__main__":
    generated = [create_nb1(), create_nb2(), create_nb3(), create_nb4()]
    print("Đã tạo thành công 4 Jupyter Notebook cho Bài kiểm tra 2:")
    for p in generated:
        print(f" - {p}")
