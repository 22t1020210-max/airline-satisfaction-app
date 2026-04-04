import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Airline Satisfaction App", page_icon="✈️", layout="wide")

# ==========================================
# HÀM TẢI DỮ LIỆU & MÔ HÌNH (SỬ DỤNG CACHE ĐỂ KHÔNG TRÀN RAM)
# ==========================================
@st.cache_data
def load_data(file_path, sample_size=5000):
    """Đọc dữ liệu và lấy mẫu ngẫu nhiên để vẽ biểu đồ cho nhẹ"""
    df = pd.read_csv(file_path)
    # Bỏ các cột không cần thiết
    cols_to_drop = ['Unnamed: 0', 'id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    # Điền khuyết
    df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
    # Lấy mẫu để tránh treo trình duyệt khi vẽ biểu đồ
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df

@st.cache_resource
def load_model():
    """Tải mô hình và bộ chuẩn hóa"""
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Tải trước dữ liệu và mô hình
train_df = load_data('data/train.csv')
model, scaler = load_model()

# ==========================================
# THANH ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Menu Điều Hướng")
page = st.sidebar.radio("Chọn trang:", 
                        ["1. Khám phá dữ liệu (EDA)", 
                         "2. Triển khai mô hình", 
                         "3. Đánh giá & Hiệu năng"])

st.sidebar.markdown("---")
st.sidebar.info("**Thông tin sinh viên:**\n\n- Họ tên: Hồ Gia Long\n- MSSV: 22T1020210\n- Môn học: Machine Learning with Python")

# ==========================================
# TRANG 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
if page == "1. Khám phá dữ liệu (EDA)":
    st.title("📊 Khám phá dữ liệu (EDA)")
    st.markdown("**Giá trị thực tiễn:** Hệ thống giúp các hãng hàng không định lượng chính xác yếu tố cốt lõi ảnh hưởng đến trải nghiệm bay, từ đó phân bổ ngân sách tối ưu để cải thiện dịch vụ.")
    
    st.subheader("1. Dữ liệu thô (Sample 5000 dòng)")
    st.dataframe(train_df.head(100)) # Hiển thị 100 dòng đầu để xem trước
    
    st.subheader("2. Phân tích trực quan")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tỷ lệ Hành khách Hài lòng vs Không hài lòng**")
        fig1, ax1 = plt.subplots(figsize=(5,4))
        # Đếm số lượng
        satisfaction_counts = train_df['satisfaction'].value_counts()
        ax1.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        st.pyplot(fig1)
        
    with col2:
        st.markdown("**Mức độ hài lòng theo Hạng ghế (Class)**")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.countplot(data=train_df, x='Class', hue='satisfaction', palette='Set2', ax=ax2)
        st.pyplot(fig2)
        
    st.info("**Nhận xét:** Dữ liệu có sự chênh lệch nhẹ nhưng không bị mất cân bằng nghiêm trọng. Khách hàng ở hạng Thương gia (Business) có tỷ lệ hài lòng cao hơn hẳn so với hạng Phổ thông (Eco).")

# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH (DỰ ĐOÁN)
# ==========================================
elif page == "2. Triển khai mô hình":
    st.title("✈️ Dự đoán mức độ hài lòng của hành khách")
    st.markdown("Vui lòng nhập thông tin chuyến bay và các đánh giá dịch vụ bên dưới:")
    
    # Tạo form nhập liệu để code gọn gàng và gom nhóm tốt hơn
    with st.form("prediction_form"):
        st.subheader("Thông tin hành khách & Chuyến bay")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            customer_type = st.selectbox("Loại khách hàng", ["Loyal Customer", "disloyal Customer"])
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=30)
        with col2:
            type_of_travel = st.selectbox("Mục đích bay", ["Personal Travel", "Business travel"])
            flight_class = st.selectbox("Hạng ghế", ["Eco", "Eco Plus", "Business"])
            flight_distance = st.number_input("Khoảng cách bay (dặm)", min_value=1, max_value=10000, value=500)
        with col3:
            dep_delay = st.number_input("Trễ giờ khởi hành (phút)", min_value=0, max_value=1500, value=0)
            arr_delay = st.number_input("Trễ giờ đến (phút)", min_value=0, max_value=1500, value=0)
            
        st.subheader("Đánh giá dịch vụ (1 - 5 sao)")
        # Lấy các cột đánh giá từ dữ liệu gốc
        service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
                        'Inflight entertainment', 'On-board service', 'Leg room service', 
                        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']
        
        # Tạo lưới slider
        s_col1, s_col2 = st.columns(2)
        user_ratings = {}
        for i, service in enumerate(service_cols):
            if i % 2 == 0:
                user_ratings[service] = s_col1.slider(service, 0, 5, 3) # 0 là không áp dụng, 1-5 là điểm
            else:
                user_ratings[service] = s_col2.slider(service, 0, 5, 3)
                
        submit_button = st.form_submit_button(label="🚀 Bấm để Dự đoán")

    if submit_button:
        # 1. Chuyển đổi Input thành DataFrame giống hệt lúc train
        input_data = {
            'Gender': 0 if gender == 'Male' else 1,
            'Customer Type': 0 if customer_type == 'disloyal Customer' else 1,
            'Age': age,
            'Type of Travel': 0 if type_of_travel == 'Personal Travel' else 1,
            'Class': 0 if flight_class == 'Eco' else (1 if flight_class == 'Eco Plus' else 2),
            'Flight Distance': flight_distance,
        }
        # Ghép thêm các cột đánh giá dịch vụ
        input_data.update(user_ratings)
        input_data['Departure Delay in Minutes'] = dep_delay
        input_data['Arrival Delay in Minutes'] = arr_delay
        
        input_df = pd.DataFrame([input_data])
        
        # Chắc chắn thứ tự cột khớp 100% với lúc train (rất quan trọng)
        feature_columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
                           'Flight Distance', 'Inflight wifi service',
                           'Departure/Arrival time convenient', 'Ease of Online booking',
                           'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                           'Inflight entertainment', 'On-board service', 'Leg room service',
                           'Baggage handling', 'Checkin service', 'Inflight service',
                           'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
        input_df = input_df[feature_columns]
        
        # 2. Chuẩn hóa dữ liệu bằng scaler đã lưu
        input_scaled = scaler.transform(input_df)
        
        # 3. Dự đoán
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        # 4. Hiển thị kết quả
        st.markdown("---")
        st.subheader("💡 Kết quả dự đoán:")
        
        if prediction[0] == 1:
            st.success(f"**Khách hàng này sẽ HÀI LÒNG (Satisfied)**")
            st.write(f"Độ tin cậy của mô hình: **{probability[0][1]*100:.2f}%**")
        else:
            st.error(f"**Khách hàng này KHÔNG HÀI LÒNG (Neutral or Dissatisfied)**")
            st.write(f"Độ tin cậy của mô hình: **{probability[0][0]*100:.2f}%**")

# ==========================================
# TRANG 3: ĐÁNH GIÁ MÔ HÌNH
# ==========================================
elif page == "3. Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng Mô hình (Evaluation)")
    st.markdown("Phần này đánh giá mô hình Logistic Regression trên tập dữ liệu kiểm thử (test.csv).")
    
    # Để tính toán chính xác, ta cần load lại tập test và xử lý
    with st.spinner("Đang tính toán các chỉ số..."):
        test_df = load_data('data/test.csv', sample_size=100000) # Lấy toàn bộ test để đánh giá chính xác
        
        # Mã hóa nhanh
        test_df['Gender'] = test_df['Gender'].map({'Male': 0, 'Female': 1})
        test_df['Customer Type'] = test_df['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
        test_df['Type of Travel'] = test_df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        test_df['Class'] = test_df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        y_test_true = test_df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        
        X_test = test_df.drop('satisfaction', axis=1)
        X_test_scaled = scaler.transform(X_test)
        
        # Dự đoán
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_true, y_pred)
        
        st.metric(label="Độ chính xác (Accuracy)", value=f"{acc*100:.2f}%")
        
        st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
        cm = confusion_matrix(y_test_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Không hài lòng', 'Hài lòng'], 
                    yticklabels=['Không hài lòng', 'Hài lòng'], ax=ax)
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        st.pyplot(fig)
        
        st.markdown("""
        **Phân tích sai số:**
        - Mô hình hoạt động khá tốt với độ chính xác ~87%.
        - Tuy nhiên, ma trận nhầm lẫn cho thấy vẫn có một lượng khách hàng bị dự đoán sai (Âm tính giả và Dương tính giả). Nguyên nhân có thể do Logistic Regression chỉ tìm được ranh giới tuyến tính.
        - **Hướng cải thiện:** Trong tương lai có thể thử nghiệm các mô hình phi tuyến tính phức tạp hơn như Random Forest hoặc XGBoost.
        """)