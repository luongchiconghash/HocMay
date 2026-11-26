# Dự đoán và giải thích quyết định phê duyệt khoản vay

Đề tài Học máy của Nhóm 13 xây dựng mô hình phân loại nhị phân cho hồ sơ vay. Bài toán gồm hai câu hỏi: hồ sơ có khả năng được phê duyệt hay không, và những đặc trưng nào ảnh hưởng đến dự đoán ở cấp toàn bộ dữ liệu hoặc từng hồ sơ?

## Dữ liệu và phạm vi

- **Đầu vào** gồm các đặc trưng như điểm tín dụng, thu nhập, khoản nợ hiện tại, số tiền vay, mục đích vay và tỷ lệ nợ trên thu nhập (DTI).
- **Đầu ra** là xác suất mô hình dự đoán hồ sơ được duyệt và nhãn phân loại theo biến đích `loan_status`.
- **Dữ liệu thực nghiệm** gồm 50.000 hồ sơ trong tệp `Loan_approval_data_2025.csv` từ bộ *Realistic Loan Approval Dataset – US & Canada* trên Kaggle, được lưu trong thư mục `data/` của từng bài kiểm tra.

## Nội dung trong repo

| Phần | Nội dung |
| --- | --- |
| [Kiểm tra 1](Kiemtra1/) | Khám phá và trực quan hóa dữ liệu, đánh giá chất lượng dữ liệu, xây dựng quy trình tiền xử lý. |
| [Kiểm tra 2](Kiemtra2/) | Xây dựng mô hình đối chứng Logistic Regression, kiểm định chéo 5 fold, đánh giá trên tập test và phân tích hai loại lỗi: dự đoán duyệt nhầm và dự đoán từ chối nhầm. |
| [Kiểm tra 3](Kiemtra3/) | Thiết kế đặc trưng, so sánh mô hình cây và mô hình tổ hợp, tối ưu HistGradientBoosting, đo đóng góp của đặc trưng bằng thí nghiệm loại bỏ và phân tích khả năng tổng quát hóa. |

Theo [báo cáo Kiểm tra 3](Kiemtra3/Nhom_13_bao_cao/Nhom13_bao_cao_kt3.pdf), trên 7.500 hồ sơ test, **HistGradientBoosting tối ưu** đạt ROC-AUC **0,98343**, Accuracy **92,87%** và F1-Macro **0,9279**. ROC-AUC của Logistic Regression đối chứng là **0,94404**.

## Diễn giải mô hình và tính công bằng

| Phương pháp | Câu hỏi phân tích |
| --- | --- |
| Permutation Importance | Chất lượng dự đoán giảm bao nhiêu khi xáo trộn một đặc trưng? Trong Kiểm tra 3, `credit_score` đứng đầu theo mức giảm ROC-AUC trên tập validation. |
| PDP và ICE | Khi thay đổi giá trị một đặc trưng, dự đoán thay đổi thế nào? PDP thể hiện xu hướng trung bình, còn ICE thể hiện đường dự đoán của từng hồ sơ. |
| SHAP | Với một hồ sơ cụ thể, đặc trưng nào đẩy điểm dự đoán lên hoặc xuống so với mức tham chiếu? Có thể tổng hợp độ lớn các đóng góp để xem xu hướng toàn cục. |

PDP, ICE và SHAP là các hướng diễn giải trong đề bài. Phần thực nghiệm trong repo có kết quả Permutation Importance. Khi đối chiếu các phương pháp, có thể kiểm tra xem đặc trưng được xếp hạng cao có xu hướng tác động rõ trên PDP, ICE và SHAP hay không, đồng thời lưu ý các đặc trưng tương quan có thể làm khác biệt kết quả.

Về tính công bằng, có thể so sánh tỷ lệ dự đoán được duyệt và tỷ lệ dự đoán đúng trong số hồ sơ có nhãn được duyệt giữa các nhóm tuổi, tại cùng ngưỡng phân loại. Đây là hướng phân tích của đề tài. Các chỉ số hiệu năng trên tập test chưa đủ để kết luận mô hình công bằng.
