"""Tạo bốn notebook KT3 từ cùng mã chia dữ liệu và kết quả thực nghiệm."""

from pathlib import Path
from textwrap import dedent

import nbformat

CODE_DIR = Path(__file__).resolve().parent
SETUP = """
from pathlib import Path
import json
import sys

import pandas as pd
from IPython.display import Image, display

for directory in (Path.cwd(), Path.cwd() / "code"):
    if (directory / "workflow_kt3.py").is_file():
        sys.path.insert(0, str(directory))
        break

from workflow_kt3 import ROOT, SUMMARY_PATH, load_splits

train, validation, test = load_splits()
with SUMMARY_PATH.open(encoding="utf-8") as summary_file:
    summary = json.load(summary_file)
"""

NOTEBOOKS = {
    "01_phan_tich_han_che_va_thiet_ke_dac_trung.ipynb": [
        ("markdown", """
         # 01 · Hạn chế của mô hình cơ sở và thiết kế đặc trưng

         Nhóm 13 · Bài kiểm tra 3 · Tiêu chí 1–3

         Logistic Regression của KT2 là đối chứng. Notebook dùng Validation để xem
         lỗi, và dùng các fold của Train để đo tác động của đặc trưng.
         Chạy `code/run_experiments_kt3.py` nếu chưa có JSON và biểu đồ.
         """),
        ("code", SETUP),
        ("code", """
         from workflow_kt3 import NEW_COLUMNS, engineer_features, make_candidates

         pd.DataFrame(
             [(name, len(part.y), part.y.mean()) for name, part in
              [("Train", train), ("Validation", validation), ("Test", test)]],
             columns=["Tập", "Số hồ sơ", "Tỷ lệ nhãn duyệt"],
         )
         """),
        ("markdown", """
         ## Phân tích lỗi và giới hạn mô hình tuyến tính

         FP là dự đoán duyệt trên hồ sơ có nhãn từ chối. FN là dự đoán từ chối
         trên hồ sơ có nhãn duyệt. So sánh trung vị của các nhóm lỗi với Validation
         để tìm khác biệt về điểm tín dụng và thu nhập. Đường cong học tập cho
         biết hiệu quả của tăng kích thước huấn luyện với cấu hình hiện tại.
         """),
        ("code", """
         display(pd.DataFrame(summary["baseline_val_error_profile"]).round(4))
         display(Image(filename=str(ROOT / "images" / "duong_cong_hoc_tap_kt3.png"), width=900))
         """),
        ("markdown", """
         ## Giải pháp và tác động đo được

         | Hạn chế | Giải pháp |
         |---|---|
         | Hệ số tuyến tính chưa mô tả linh hoạt tương tác giữa các thuộc tính | So sánh với CART, Random Forest và HistGradientBoosting |
         | Nhiều thuộc tính có ý nghĩa khi biểu diễn theo tỷ số hoặc kết hợp | Tạo 5 đặc trưng từ các cột đầu vào và đo tác động bằng các cặp đối chứng |
         | Cấu hình cây ảnh hưởng đến độ phức tạp và chất lượng trên fold kiểm định | Tìm siêu tham số trên Train bằng GridSearchCV |

         Năm đặc trưng lần lượt là tỷ lệ dư nợ cộng khoản vay trên thu nhập, tỷ
         lệ tiết kiệm trên khoản vay, chỉ số lịch sử tín dụng tổng hợp, tích DTI
         với lãi suất và tỷ lệ thâm niên trên tuổi trưởng thành. Các phép tính
         chỉ sử dụng thông tin trong từng hồ sơ.
         """),
        ("code", """
         display(engineer_features(train.X).loc[:, list(NEW_COLUMNS)].head().round(4))

         rows = pd.DataFrame(summary["ablation_study"])
         display(rows.iloc[:2][["So sánh", "AUC trước", "AUC sau", "Δ ROC-AUC",
                                 "Δ Accuracy (điểm %)", "Δ F1-Macro"]].round(5))
         """),
        ("markdown", """
         Hai hàng sử dụng cùng mô hình, cùng fold và chỉ thay đổi tập đặc trưng.
         Vì vậy có thể đo tác động thêm đặc trưng cho từng họ mô hình. Dấu và
         độ lớn của thay đổi phụ thuộc thuật toán và chỉ số đang xét.
         """),
    ],
    "02_thu_nghiem_mo_hinh_va_ensemble.ipynb": [
        ("markdown", """
         # 02 · Thử nghiệm thuật toán và học tổ hợp

         Nhóm 13 · Bài kiểm tra 3 · Tiêu chí 4

         So sánh Logistic Regression, CART, SVM tuyến tính, Random Forest,
         HistGradientBoosting và Stacking trên cùng 5 fold. Chọn theo
         ROC-AUC, đồng thời xem Accuracy và F1-Macro.
         """),
        ("code", SETUP),
        ("markdown", """
         CART tạo các ngưỡng phân nhánh. SVM tuyến tính học biên phân lớp qua
         hinge loss. Random Forest lấy trung bình nhiều cây để giảm biến động.
         HistGradientBoosting thêm cây tuần tự theo gradient của hàm mất mát.
         Stacking huấn luyện Logistic Regression tầng hai trên các dự đoán OOF
         từ Random Forest, HGB và Logistic Regression. Các mô hình tầng một
         đều có tiền xử lý riêng trong từng fold OOF.
         """),
        ("code", """
         result = pd.DataFrame(summary["cv_results"])
         columns = ["Mô hình", "CV ROC-AUC", "CV ROC-AUC std", "CV Accuracy", "CV F1-Macro"]
         display(result[columns].sort_values("CV ROC-AUC", ascending=False).round(5))
         display(Image(filename=str(ROOT / "images" / "so_sanh_mo_hinh_kt2_vs_kt3.png"), width=960))
         """),
        ("markdown", """
         ## Chạy lại một đối chứng trong notebook

         Đo CART với đúng pipeline, Train và các fold từ script. Các mô hình
         tốn thời gian hơn được chạy trong script chính và hiện ở bảng trên.
         """),
        ("code", """
         import numpy as np
         from sklearn.model_selection import cross_val_score
         from threadpoolctl import threadpool_limits
         from workflow_kt3 import make_candidates, stratified_cv

         cart = make_candidates(train.X.columns)["cart"]
         with threadpool_limits(1):
             scores = cross_val_score(cart, train.X, train.y, cv=stratified_cv(), scoring="roc_auc")
         stored_scores = next(row for row in summary["cv_results"] if row["id"] == "cart")
         display(pd.DataFrame({"Fold": range(1, len(scores) + 1), "ROC-AUC": scores}).round(5))
         np.testing.assert_allclose(scores, stored_scores["Fold ROC-AUC"], atol=1e-8)
         """),
    ],
    "03_toi_uu_sieu_tham_so_va_ablation_study.ipynb": [
        ("markdown", """
         # 03 · Tối ưu siêu tham số và ablation

         Nhóm 13 · Bài kiểm tra 3 · Tiêu chí 5–6

         GridSearchCV chấm các cấu hình HGB trên 5 fold của 35.000 mẫu Train.
         Hàng đứng đầu tối đa hóa trung bình ROC-AUC. Tập Test được giữ cho
         đánh giá sau lựa chọn.
         """),
        ("code", SETUP),
        ("code", """
         grid = summary["grid_search_hgb"]
         display(pd.DataFrame(
             [(name.removeprefix("clf__"), str(values)) for name, values in grid["space"].items()],
             columns=["Siêu tham số", "Giá trị khảo sát"],
         ))
         rows = [{"Hạng": row["rank"], **{
                     name.removeprefix("clf__"): value for name, value in row["params"].items()
                 }, "ROC-AUC": row["mean_roc_auc"], "Độ lệch chuẩn": row["std_roc_auc"]}
                 for row in grid["rows"]]
         display(pd.DataFrame(rows).head(8).round(6))
         """),
        ("markdown", """
         Đường cong dưới thay đổi một siêu tham số và giữ các tham số còn lại
         theo cấu hình đã chọn. Tiền xử lý được fit trong từng fold vì
         validation_curve nhận cả pipeline. Đây là phân tích độ nhạy trên Train.
         """),
        ("code", """
         display(Image(filename=str(ROOT / "images" / "duong_cong_kiem_dinh_hyperparameter.png"), width=900))
         """),
        ("markdown", """
         ## Ablation trên cùng các fold

         Mỗi hàng lấy điểm cấu hình sau trừ điểm cấu hình trước. Hai hàng đầu
         giữ nguyên thuật toán và chỉ thêm đặc trưng. Các hàng đổi thuật toán
         giữ tập đặc trưng. Hàng tối ưu so cùng họ HGB trước và sau GridSearchCV.
         """),
        ("code", """
         columns = ["So sánh", "AUC trước", "AUC sau", "Δ ROC-AUC",
                    "Δ Accuracy (điểm %)", "Δ F1-Macro"]
         display(pd.DataFrame(summary["ablation_study"])[columns].round(5))
         display(Image(filename=str(ROOT / "images" / "ablation_study_kt3.png"), width=920))
         """),
        ("markdown", """
         Điểm CV của cấu hình tối ưu dùng chung các fold đã phục vụ tìm kiếm.
         Đánh giá mô hình đã chốt trên Test được trình bày trong notebook 04.
         """),
    ],
    "04_danh_gia_tong_quat_hoa_va_thuc_hanh_tai_cho.ipynb": [
        ("markdown", """
         # 04 · Đánh giá tổng quát hóa và thực hành tại chỗ

         Nhóm 13 · Bài kiểm tra 3 · Tiêu chí 7–10

         Báo cáo Test chỉ gồm Logistic Regression của KT2 và ứng viên đã chốt
         bằng ROC-AUC trung bình trên 5-fold CV của Train. PR-AUC trong bảng
         được tính bằng average precision.
         """),
        ("code", SETUP),
        ("code", """
         import joblib
         import numpy as np
         from sklearn.model_selection import cross_val_score, train_test_split
         from threadpoolctl import threadpool_limits
         from workflow_kt3 import SEED, evaluate, get_model_score, make_hgb, stratified_cv, threshold_table

         selection = summary["selection"]
         chosen_id = selection["model_id"]
         chosen = joblib.load(ROOT / "models" / selection["artifact"])
         ids = ["baseline", chosen_id]
         table = {}
         for key in ids:
             cv_row = next(row for row in summary["cv_results"] if row["id"] == key)
             val_row = next(row for row in summary["val_results"] if row["id"] == key)
             test_row = next(row for row in summary["test_results"] if row["id"] == key)
             table[cv_row["Mô hình"]] = {
                 "CV AUC": cv_row["CV ROC-AUC"], "Độ lệch chuẩn CV": cv_row["CV ROC-AUC std"],
                 "Validation AUC": val_row["ROC-AUC"], "Test AUC": test_row["ROC-AUC"],
                 "Test Accuracy": test_row["Accuracy"], "Test F1-Macro": test_row["F1-Macro"],
                 "Test PR-AUC": test_row["PR-AUC"], "FP": test_row["FP"], "FN": test_row["FN"],
                 "Độ trễ (ms/1.000 mẫu)": summary["latency_ms_per_1000_samples"][key],
             }
         display(pd.DataFrame.from_dict(table, orient="index").round(5))

         with threadpool_limits(1):
             checks = evaluate(test.y, chosen.predict(test.X), get_model_score(chosen, test.X))
         for metric in ("ROC-AUC", "Accuracy", "F1-Macro", "FP", "FN"):
             np.testing.assert_allclose(checks[metric], test_row[metric], atol=1e-10)
         """),
        ("code", """
         for figure in ("duong_cong_roc_so_sanh_kt3.png",
                        "duong_cong_pr_so_sanh_kt3.png",
                        "ma_tran_nham_lan_kt2_vs_kt3.png",
                        "do_quan_trong_dac_trung_kt3.png"):
             display(Image(filename=str(ROOT / "images" / figure), width=760))
         """),
        ("markdown", """
         ROC-AUC phản ánh khả năng xếp hạng hai lớp. Ma trận nhầm lẫn cho thấy
         sai số tại ngưỡng hiện hành. Độ trễ là trung vị của các lần gọi
         predict trên cùng lô 1.000 hồ sơ Validation, bao gồm tiền xử lý.
         Permutation Importance phản ánh mức phụ thuộc vào từng cột trên
         Validation và có thể bị chia sẻ giữa các biến tương quan.
         """),
        ("markdown", """
         ## Thực hành 1 · Quan sát ảnh hưởng của ngưỡng

         Kiều Việt Anh. Trên Validation, so sánh 0,50 với 0,60 và giải thích
         thay đổi của FP, FN, Specificity và Recall. Khi tăng ngưỡng, số dự đoán
         duyệt giảm. Việc chọn một ngưỡng vận hành cần chi phí sai số hoặc ràng
         buộc nghiệp vụ được xác định trước.
         """),
        ("code", """
         hgb = joblib.load(ROOT / "models" / "best_hist_gradient_boosting_pipeline.joblib")
         with threadpool_limits(1):
             probabilities = hgb.predict_proba(validation.X)[:, 1]
         threshold_table(validation.y, probabilities).round(4)
         """),
        ("markdown", """
         ## Thực hành 2 · Đổi tốc độ học và chạy CV nhanh

         Lương Chí Công. Lấy 15.000 hồ sơ có phân tầng từ Train. Giữ nguyên
         3 fold và các siêu tham số còn lại, so sánh tốc độ học 0,08 với 0,15.
         So sánh hai kết quả trên chính các fold này và đọc cả độ lệch chuẩn.
         """),
        ("code", """
         X_demo, _, y_demo, _ = train_test_split(
             train.X, train.y, train_size=15000, stratify=train.y, random_state=SEED
         )
         demo_rows = []
         with threadpool_limits(1):
             for learning_rate in (0.08, 0.15):
                 model = make_hgb(X_demo.columns, learning_rate=learning_rate)
                 scores = cross_val_score(model, X_demo, y_demo, cv=stratified_cv(3), scoring="roc_auc")
                 demo_rows.append({"learning_rate": learning_rate, "ROC-AUC": scores.mean(),
                                   "Độ lệch chuẩn": scores.std()})
         pd.DataFrame(demo_rows).round(5)
         """),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        notebook_cells = []
        for kind, source in cells:
            build = nbformat.v4.new_code_cell if kind == "code" else nbformat.v4.new_markdown_cell
            notebook_cells.append(build(dedent(source).strip()))
        notebook = nbformat.v4.new_notebook(
            cells=notebook_cells,
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        )
        nbformat.validate(notebook)
        nbformat.write(notebook, CODE_DIR / name)


if __name__ == "__main__":
    main()
