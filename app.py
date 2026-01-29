import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Dự báo giá nhà - CONQ016",
    page_icon="🏠",
    layout="wide"
)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model_objects():
    try:
        model = joblib.load('house_price_model.pkl')
        # Load features list để đảm bảo đúng thứ tự cột
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

model, model_features = load_model_objects()

# --- 3. GIAO DIỆN ---
st.title("🏠 Hệ thống Dự báo Giá nhà Thông minh")
st.markdown("**Phát triển bởi nhóm nghiên cứu: CONQ016**")
st.info("💡 Hướng dẫn: Nhập các thông số bên dưới để dự báo giá.")

st.sidebar.header("📝 Nhập liệu")

# --- PHẦN 1: CHẤT LƯỢNG (QUAN TRỌNG NHẤT) ---
with st.sidebar.expander("1. Đánh giá chất lượng (Total_Qua)", expanded=True):
    st.markdown("hãy đánh giá chất lượng từng hạng mục:")
    
    # 1.1 Overall Qual (Thang 1-10)
    overall_qual = st.slider("Chất lượng tổng thể (Overall Qual)", 1, 10, 5)
    
    # Hàm hỗ trợ tạo Selectbox chấm điểm
    def quality_select(label):
        return st.selectbox(
            label,
            options=[5, 4, 3, 2, 1, 0],
            format_func=lambda x: {
                5: "5 - Xuất sắc (Ex)",
                4: "4 - Tốt (Gd)",
                3: "3 - Trung bình (TA)",
                2: "2 - Khá (Fa)",
                1: "1 - Kém (Po)",
                0: "0 - Không có (None)"
            }[x]
        )

    # 1.2 Các chỉ số chất lượng khác
    exter_qual = quality_select("Chất lượng ngoại thất (Exter Qual)")
    kitchen_qual = quality_select("Chất lượng bếp (Kitchen Qual)")
    bsmt_qual = quality_select("Chất lượng tầng hầm (Bsmt Qual)")
    garage_qual = quality_select("Chất lượng nhà xe (Garage Qual)")

# --- PHẦN 2: DIỆN TÍCH ---
with st.sidebar.expander("2. Diện tích & Không gian", expanded=True):
    # Biến trực tiếp: 1st Flr SF, Garage Area
    flr1_sf = st.number_input("Diện tích tầng 1 (1st Flr SF)", value=1000.0)
    garage_area = st.number_input("Diện tích Gara (Garage Area)", value=500.0)

# --- PHẦN 3: THỜI GIAN ---
with st.sidebar.expander("3. Tuổi đời nhà (Age)", expanded=False):
    year_built = st.number_input("Năm xây dựng", min_value=1800, max_value=2025, value=2000)
    yr_sold = st.number_input("Năm bán", min_value=2000, max_value=2030, value=2024)

# --- 4. XỬ LÝ & DỰ BÁO ---
if st.button("🚀 Dự báo ngay", type="primary"):
    if model is None:
        st.error("⚠️ Không tìm thấy file 'house_price_model.pkl'. Hãy kiểm tra lại thư mục.")
    else:
        # --- FEATURE ENGINEERING (GIỐNG HỆT NOTEBOOK) ---
        
        # 1. Tính Age
        val_Age = yr_sold - year_built
        
        # 2. Tính Total_Qua
        # Công thức: Overall + Exter + Kitchen + Bsmt + Garage
        val_Total_Qua = overall_qual + exter_qual + kitchen_qual + bsmt_qual + garage_qual
        
        # Tạo DataFrame input
        # Lưu ý: Tên cột phải khớp chính xác với những gì model yêu cầu
        input_data = {
            'Age': val_Age,
            'Total_Qua': val_Total_Qua,
            '1st Flr SF': flr1_sf,
            'Garage Area': garage_area
        }
        
        input_df = pd.DataFrame([input_data])
        
# --- SẮP XẾP CỘT & DỰ BÁO ---
        try:
            # Tự động sắp xếp cột theo đúng thứ tự lúc train
            final_input = input_df[model_features]
            
            # 1. Dự báo (Kết quả này đang ở dạng Log)
            prediction_log = model.predict(final_input)[0]
            
            # 2. Chuyển đổi ngược lại giá tiền thật (Exponential)
            prediction = np.exp(prediction_log)
            
            st.success(f"💎 Giá nhà dự đoán: ${prediction:,.2f}")
            
            # Hiển thị thêm thông tin để check
            with st.expander("🔍 Xem chi tiết kỹ thuật"):
                st.write(f"Giá trị Logarit từ model: {prediction_log:.4f}")
                st.write(f"Giá trị thực (exp): ${prediction:,.2f}")
                st.write("Dữ liệu đầu vào:", final_input)
                
        except KeyError as e:
            st.error(f"Lỗi: Model yêu cầu cột {e} nhưng App chưa tính toán cột này.")