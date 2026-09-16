import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.predict import predict_one

st.set_page_config(page_title="Customer Return Prediction", page_icon="📊", layout="centered")
st.title("📊 Dự đoán khả năng khách hàng quay lại mua sắm")
st.caption("Machine Learning demo - dữ liệu mô phỏng phục vụ học tập")

if not MODEL_PATH.exists():
    st.error("Chưa có model. Hãy chạy: python -m src.generate_data && python -m src.train")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi", 18, 80, 28)
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        purchase_count = st.number_input("Số lần mua", 1, 100, 8)
        total_spent = st.number_input("Tổng chi tiêu", 0.0, 1000000.0, 4200.0)
        avg_order_value = st.number_input("Giá trị đơn trung bình", 0.0, 100000.0, 525.0)
        days_since_last_purchase = st.number_input("Số ngày từ lần mua cuối", 1, 365, 12)
        website_visits_30d = st.number_input("Lượt truy cập website 30 ngày", 0, 500, 15)
    with col2:
        cart_adds_30d = st.number_input("Lần thêm giỏ 30 ngày", 0, 100, 6)
        discount_usage_rate = st.slider("Tỷ lệ dùng mã giảm giá", 0.0, 1.0, 0.25)
        support_tickets_90d = st.number_input("Ticket hỗ trợ 90 ngày", 0, 30, 0)
        customer_tenure_days = st.number_input("Thâm niên khách hàng (ngày)", 30, 3000, 420)
        preferred_channel = st.selectbox("Kênh chính", ["Web", "Mobile", "Store"])
        region = st.selectbox("Khu vực", ["North", "Central", "South"])

    submitted = st.form_submit_button("Dự đoán")

if submitted:
    customer = {
        "age": age,
        "gender": gender,
        "purchase_count": purchase_count,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "days_since_last_purchase": days_since_last_purchase,
        "website_visits_30d": website_visits_30d,
        "cart_adds_30d": cart_adds_30d,
        "discount_usage_rate": discount_usage_rate,
        "support_tickets_90d": support_tickets_90d,
        "customer_tenure_days": customer_tenure_days,
        "preferred_channel": preferred_channel,
        "region": region,
    }
    result = predict_one(MODEL_PATH, customer)
    st.subheader(result["label"])
    st.metric("Xác suất quay lại", f"{result['probability'] * 100:.2f}%")
    if result["prediction"]:
        st.success("Hệ thống xếp khách hàng vào nhóm có khả năng quay lại.")
    else:
        st.warning("Hệ thống xếp khách hàng vào nhóm khả năng quay lại thấp.")
