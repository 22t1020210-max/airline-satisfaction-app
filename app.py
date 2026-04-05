import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Airline Satisfaction App", page_icon="✈️", layout="wide")

# ==========================================
# 1. TÍNH NĂNG MỚI: ERROR HANDLING CHO LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    """Tải mô hình, bộ chuẩn hóa, giá trị trung bình và cấu trúc cột với Error Handling"""
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/mean_delay.pkl', 'rb') as f:
            mean_delay = pickle.load(f) 
        with open('models/model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f) 
        return model, scaler, mean_delay, model_columns
    except FileNotFoundError:
        st.error("❌ Không tìm thấy thư mục 'models/'. Vui lòng chạy lại file huấn luyện trước!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi hệ thống khi tải mô hình: {e}")
        st.stop()

# Tải Model và Tham số ra trước
model, scaler, mean_delay, model_columns = load_model()

# ==========================================
# 2. TÍNH NĂNG MỚI: TĂNG SAMPLE SIZE LÊN 25,000 & ERROR HANDLING DATA
# ==========================================
@st.cache_data
def load_data(file_path, mean_val, sample_size=25000):
    """Đọc dữ liệu, điền khuyết bằng trung bình chuẩn và lấy mẫu"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        cols_to_drop = ['Unnamed: 0', 'id']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(mean_val)
        
        # Tăng lượng mẫu lấy ngẫu nhiên lên 25000 để biểu đồ chính xác hơn
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file dữ liệu: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        st.stop()

train_df = load_data('data/train.csv', mean_val=mean_delay)

# ==========================================
# THANH ĐIỀU HƯỚNG & TÍNH NĂNG MỚI: DEMO PRESETS (TEST NHANH)
# ==========================================
st.sidebar.title("📌 Menu Điều Hướng")
page = st.sidebar.radio("Chọn trang:", 
                        ["1. Khám phá dữ liệu (EDA)", 
                         "2. Triển khai mô hình", 
                         "3. Đánh giá & Hiệu năng"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Thử nghiệm nhanh (Demo)")
st.sidebar.markdown("Bấm để tự động điền form dự đoán:")

# Khởi tạo trạng thái phiên (Session State) cho các nút Demo
if 'demo_preset' not in st.session_state:
    st.session_state.demo_preset = None

if st.sidebar.button("👨‍💼 Khách VIP (Trải nghiệm tốt)"):
    st.session_state.demo_preset = 'vip'
if st.sidebar.button("😤 Khách trễ chuyến (Dịch vụ tệ)"):
    st.session_state.demo_preset = 'bad'
if st.sidebar.button("🔄 Đặt lại Form"):
    st.session_state.demo_preset = 'reset'

st.sidebar.markdown("---")
st.sidebar.info("**Thông tin sinh viên:**\n\n- Họ tên: Hồ Gia Long\n- MSSV: 22T1020210\n- Môn học: Machine Learning with Python")

# ==========================================
# TRANG 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
if page == "1. Khám phá dữ liệu (EDA)":
    st.title("📊 Khám phá dữ liệu (EDA)")
    st.markdown("**Giá trị thực tiễn:** Giúp các hãng hàng không định lượng chính xác yếu tố cốt lõi ảnh hưởng đến trải nghiệm bay.")
    
    st.info(f"💡 Đang hiển thị biểu đồ phân tích dựa trên tập mẫu ngẫu nhiên gồm **{len(train_df):,}** hành khách.")
    
    st.subheader("1. Dữ liệu thô (Hiển thị 100 dòng đầu)")
    st.dataframe(train_df.head(100))
    
    st.subheader("2. Phân tích trực quan")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tỷ lệ Hành khách Hài lòng vs Không hài lòng**")
        fig1, ax1 = plt.subplots(figsize=(5,4))
        satisfaction_counts = train_df['satisfaction'].value_counts()
        ax1.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        st.pyplot(fig1)
        
    with col2:
        st.markdown("**Mức độ hài lòng theo Hạng ghế (Class)**")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.countplot(data=train_df, x='Class', hue='satisfaction', palette='Set2', ax=ax2)
        st.pyplot(fig2)
        
    st.success("**Nhận xét:** Khách hàng ở hạng Thương gia (Business) có tỷ lệ hài lòng cao hơn hẳn so với hạng Phổ thông (Eco).")

# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH (DỰ ĐOÁN)
# ==========================================
elif page == "2. Triển khai mô hình":
    st.title("✈️ Dự đoán mức độ hài lòng của hành khách")
    
    # --- Cài đặt giá trị mặc định dựa trên nút bấm ở Sidebar ---
    def_class_idx = 0  # 0: Eco, 1: Eco Plus, 2: Business
    def_delay = 0
    def_arr_delay = int(mean_delay)
    def_rating = 3
    
    if st.session_state.demo_preset == 'vip':
        def_class_idx = 2  # Business
        def_delay = 0
        def_arr_delay = 0
        def_rating = 5
    elif st.session_state.demo_preset == 'bad':
        def_class_idx = 0  # Eco
        def_delay = 120    # Trễ 2 tiếng
        def_arr_delay = 130
        def_rating = 1
    # -----------------------------------------------------------

    with st.form("prediction_form"):
        st.subheader("Thông tin hành khách & Chuyến bay")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            customer_type = st.selectbox("Loại khách hàng", ["Loyal Customer", "disloyal Customer"])
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=30)
        with col2:
            type_of_travel = st.selectbox("Mục đích bay", ["Personal Travel", "Business travel"])
            flight_class = st.selectbox("Hạng ghế", ["Eco", "Eco Plus", "Business"], index=def_class_idx)
            flight_distance = st.number_input("Khoảng cách bay (dặm)", min_value=1, max_value=10000, value=500)
        with col3:
            dep_delay = st.number_input("Trễ giờ khởi hành (phút)", min_value=0, max_value=1500, value=def_delay)
            arr_delay = st.number_input("Trễ giờ đến (phút)", min_value=0, max_value=1500, value=def_arr_delay)
            
        st.subheader("Đánh giá dịch vụ (1 - 5 sao)")
        service_map = {
            'Inflight wifi service': 'Dịch vụ Wifi',
            'Departure/Arrival time convenient': 'Giờ bay/đáp thuận tiện',
            'Ease of Online booking': 'Dễ dàng đặt vé Online',
            'Gate location': 'Vị trí cổng lên máy bay',
            'Food and drink': 'Đồ ăn và thức uống',
            'Online boarding': 'Làm thủ tục (Boarding) Online',
            'Seat comfort': 'Sự thoải mái của ghế ngồi',
            'Inflight entertainment': 'Giải trí trên chuyến bay',
            'On-board service': 'Dịch vụ trên máy bay',
            'Leg room service': 'Không gian để chân',
            'Baggage handling': 'Xử lý hành lý',
            'Checkin service': 'Dịch vụ Check-in',
            'Inflight service': 'Dịch vụ hỗ trợ chung',
            'Cleanliness': 'Độ sạch sẽ'
        }
        
        s_col1, s_col2 = st.columns(2)
        user_ratings = {}
        for i, (eng_key, vie_label) in enumerate(service_map.items()):
            if i % 2 == 0:
                user_ratings[eng_key] = s_col1.slider(vie_label, 0, 5, def_rating) 
            else:
                user_ratings[eng_key] = s_col2.slider(vie_label, 0, 5, def_rating)
                
        submit_button = st.form_submit_button(label="🚀 Bấm để Dự đoán")

    # 3. TÍNH NĂNG MỚI: ERROR HANDLING KHI BẤM DỰ ĐOÁN
    if submit_button:
        try:
            input_data = {
                'Gender': gender,
                'Customer Type': customer_type,
                'Age': age,
                'Type of Travel': type_of_travel,
                'Class': flight_class,
                'Flight Distance': flight_distance,
                'Departure Delay in Minutes': dep_delay,
                'Arrival Delay in Minutes': arr_delay
            }
            input_data.update(user_ratings)
            input_df = pd.DataFrame([input_data])
            
            categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
            
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
                    
            input_df = input_df[model_columns]
            
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            st.markdown("---")
            st.subheader("💡 Kết quả dự đoán:")
            
            if prediction[0] == 1:
                st.success(f"**Khách hàng này sẽ HÀI LÒNG (Satisfied)**")
                st.write(f"Độ tin cậy của mô hình: **{probability[0][1]*100:.2f}%**")
            else:
                st.error(f"**Khách hàng này KHÔNG HÀI LÒNG (Neutral or Dissatisfied)**")
                st.write(f"Độ tin cậy của mô hình: **{probability[0][0]*100:.2f}%**")
        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi trong quá trình dự toán: {e}")

# ==========================================
# TRANG 3: ĐÁNH GIÁ MÔ HÌNH
# ==========================================
# ==========================================
# TRANG 3: ĐÁNH GIÁ MÔ HÌNH
# ==========================================
elif page == "3. Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng & Trọng số Đặc trưng")
    st.markdown("Trang này giúp Ban quản trị đánh giá độ tin cậy của thuật toán và tìm ra các 'điểm chạm' dịch vụ quan trọng nhất.")
    
    with st.spinner("Đang tính toán các chỉ số trên tập kiểm thử..."):
        # Tải tập test
        test_df = load_data('data/test.csv', mean_val=mean_delay, sample_size=50000) 
        
        y_test_true = test_df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        X_test = test_df.drop('satisfaction', axis=1)
        
        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        for col in model_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[model_columns]
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            acc = accuracy_score(y_test_true, y_pred)
            st.metric(label="🎯 Độ chính xác (Accuracy)", value=f"{acc*100:.2f}%", help="Tỷ lệ dự đoán trúng tổng thể trên toàn bộ tập khách hàng kiểm thử.")
        with col2:
            roc_auc = roc_auc_score(y_test_true, y_pred_proba)
            st.metric(label="📏 Chỉ số ROC-AUC", value=f"{roc_auc:.4f}", help="Chỉ số đo lường khả năng phân biệt giữa khách Hài lòng và Không hài lòng. Càng gần 1 càng tốt.")
            
        st.markdown("---")
        
        # CHIA LÀM 2 CỘT: 1 BÊN ROC CURVE - 1 BÊN CONFUSION MATRIX
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            st.subheader("📈 Đường cong ROC")
            fpr, tpr, thresholds = roc_curve(y_test_true, y_pred_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('Tỷ lệ Dương tính giả (Đoán nhầm thành Hài lòng)')
            ax_roc.set_ylabel('Tỷ lệ Dương tính thật (Đoán trúng Hài lòng)')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            # THÊM GHI CHÚ CHO ROC:
            st.info("**💡 Cách đọc:** Đường cong màu cam càng ôm sát góc trên bên trái thì mô hình càng thông minh. Diện tích AUC = 1.0 là hoàn hảo.")

        with plot_col2:
            st.subheader("📊 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(y_test_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Không hài lòng', 'Hài lòng'],
                        yticklabels=['Không hài lòng', 'Hài lòng'])
            ax_cm.set_ylabel('Thực tế (True Label)')
            ax_cm.set_xlabel('Dự đoán (Predicted Label)')
            st.pyplot(fig_cm)
            # THÊM GHI CHÚ CHO MA TRẬN NHẦM LẪN:
            st.info("**💡 Cách đọc:** Đường chéo chính (đậm màu) là số lượng khách hàng mô hình **đoán đúng**. Hai ô nhạt màu còn lại là những ca bị **đoán sai**.")

        st.markdown("---")
        
        st.subheader("⚖️ Mức độ tác động của các dịch vụ (Feature Weights)")
        weights_df = pd.DataFrame({
            'Feature': model_columns,
            'Weight': model.coef_[0]
        }).sort_values(by='Weight', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        top_features = pd.concat([weights_df.head(5), weights_df.tail(5)])
        sns.barplot(data=top_features, x='Weight', y='Feature', palette='vlag', ax=ax)
        plt.title('Top 5 yếu tố tích cực nhất & Top 5 yếu tố tiêu cực nhất')
        plt.xlabel('Trọng số (Weight)')
        plt.ylabel('Các đặc trưng / Dịch vụ')
        st.pyplot(fig)
        
        # THÊM GHI CHÚ CHO TRỌNG SỐ:
        st.success("**💡 Phân tích dành cho Ban quản trị:** \n"
                   "- **Trọng số Dương (Hướng sang phải):** Đây là những dịch vụ 'ăn tiền', đóng vai trò cốt lõi làm khách hàng thỏa mãn (Ví dụ: Wifi, Dịch vụ On-board). Hãng cần tiếp tục duy trì và quảng bá.\n"
                   "- **Trọng số Âm (Hướng sang trái):** Đây là những điểm nghẽn gây phẫn nộ (Ví dụ: Thời gian trễ chuyến). Hãng cần ưu tiên ngân sách để khắc phục các lỗi này ngay lập tức.")