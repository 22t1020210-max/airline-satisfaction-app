import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

# Tạo thư mục models nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')

print("1. Đang đọc dữ liệu...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Bỏ các cột không mang ý nghĩa phân tích
cols_to_drop = ['Unnamed: 0', 'id']
train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], errors='ignore')
test_df = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')

print("2. Xử lý giá trị khuyết thiếu bằng MEAN (Trung bình)...")
train_mean_delay = train_df['Arrival Delay in Minutes'].mean()

# Dùng Mean để điền cho CẢ tập train và tập test
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(train_mean_delay)
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(train_mean_delay)

print("3. Mã hóa dữ liệu bằng ONE-HOT ENCODING...")
# Tách biến mục tiêu (target)
target_map = {'neutral or dissatisfied': 0, 'satisfied': 1}
y_train = train_df['satisfaction'].map(target_map)
y_test = test_df['satisfaction'].map(target_map)

X_train = train_df.drop('satisfaction', axis=1)
X_test = test_df.drop('satisfaction', axis=1)

# Các cột danh mục cần One-Hot Encoding
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Áp dụng get_dummies (One-Hot Encoding), dùng drop_first=True để tránh đa cộng tuyến trong Logistic Regression
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Đảm bảo tập Train và Test có cùng cấu trúc cột sau khi One-Hot Encoding
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print("4. Chuẩn hóa dữ liệu (Scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("5. Đang huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("6. Đánh giá mô hình (ROC-AUC & Feature Weights)...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Lấy xác suất của lớp 1 (Hài lòng)

# Tính và in ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"-> ROC-AUC Score: {roc_auc:.4f}")
print(f"-> Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

# Trích xuất và in Trọng số đặc trưng (Feature Weights)
weights_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Weight': model.coef_[0]
}).sort_values(by='Weight', ascending=False)

print("--- TRỌNG SỐ ĐẶC TRƯNG (Top yếu tố làm HÀI LÒNG) ---")
print(weights_df.head(5)) # In 5 yếu tố tích cực nhất
print("\n--- TRỌNG SỐ ĐẶC TRƯNG (Top yếu tố làm KHÔNG HÀI LÒNG) ---")
print(weights_df.tail(5)) # In 5 yếu tố tiêu cực nhất

print("\n7. Đang lưu mô hình và các bộ tham số...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Lưu lại columns sau khi One-Hot để dùng cho Web App (tránh lỗi mismatch cột)
with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)

# Lưu giá trị MEAN thay vì median
with open('models/mean_delay.pkl', 'wb') as f:
    pickle.dump(train_mean_delay, f)

print("✅ HOÀN TẤT! Các file model đã được lưu vào thư mục models/.")