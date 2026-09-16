# Customer Return Prediction - End-to-End Machine Learning Project

## 1. Mục tiêu
Dự đoán khách hàng có quay lại mua sắm trong một khoảng thời gian xác định (mặc định 90 ngày) dựa trên dữ liệu nhân khẩu học, giao dịch và hành vi website.

## 2. Điểm nổi bật
- Synthetic dataset 5.000 khách hàng để thực hành khi chưa có dữ liệu thật.
- Data validation cơ bản và loại bỏ duplicate.
- Train/test split có stratify.
- Pipeline chống data leakage: imputation + scaling + one-hot encoding nằm trong pipeline.
- So sánh Logistic Regression, KNN, Decision Tree, Random Forest và Gradient Boosting.
- Stratified K-Fold cross-validation.
- GridSearchCV để tinh chỉnh Random Forest.
- Đánh giá Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC và confusion matrix.
- Lưu model bằng joblib.
- CLI prediction và giao diện Streamlit.
- Unit tests cho các tình huống dự đoán tối thiểu 2 mẫu.

## 3. Cấu trúc
```text
customer_return_ml/
│
├── data/
│   └── customers.csv
│
├── models/
│   └── best_model.joblib
│
├── results/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_predict.py
│
├── app.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## 4. Cài đặt Windows / VS Code
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Tạo dữ liệu
```powershell
python -m src.generate_data
```

## 6. Huấn luyện và đánh giá
```powershell
python -m src.train
```
Script sẽ:
- tạo train/test split;
- cross-validation cho các model;
- tune Random Forest;
- chọn model theo F1 trên validation CV;
- đánh giá trên test set;
- lưu model vào `models/best_model.joblib`;
- ghi kết quả vào `results/`.

## 7. Dự đoán CLI
```powershell
python -m src.predict --age 28 --gender Male --purchase-count 8 --total-spent 4200 --avg-order-value 525 --days-since-last-purchase 12 --website-visits-30d 15 --cart-adds-30d 6 --discount-usage-rate 0.25 --support-tickets-90d 0 --customer-tenure-days 420 --preferred-channel Mobile --region North
```

## 8. Chạy giao diện
```powershell
streamlit run app.py
```

## 9. Chạy test
```powershell
pytest -q
```

## 10. Cách trình bày trong BTL
- Chương 1: AI, supervised learning, classification, các model sử dụng, thư viện.
- Chương 2: dataset, feature engineering, preprocessing, train/validation/test, K-Fold, GridSearch, đánh giá, cấu trúc source code.
- Thực nghiệm: ít nhất 2 mẫu dự đoán trực tiếp và bảng so sánh model.
- Kết luận: model tốt nhất, hạn chế của dữ liệu giả lập, hướng dùng dữ liệu thật.

> Lưu ý: dữ liệu trong project là dữ liệu mô phỏng để phục vụ học tập. Không được trình bày các con số mô phỏng như kết quả từ doanh nghiệp thật.
