import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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

print("2. Xử lý giá trị khuyết thiếu (Imputation)...")
# CẢI THIỆN: Tính median CHỈ TỪ TẬP TRAIN
train_median_delay = train_df['Arrival Delay in Minutes'].median()

# Dùng con số này để điền cho CẢ tập train và tập test
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(train_median_delay)
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(train_median_delay)

print("3. Mã hóa dữ liệu (Manual Encoding)...")
gender_map = {'Male': 0, 'Female': 1}
customer_map = {'disloyal Customer': 0, 'Loyal Customer': 1}
travel_map = {'Personal Travel': 0, 'Business travel': 1}
class_map = {'Eco': 0, 'Eco Plus': 1, 'Business': 2} 
target_map = {'neutral or dissatisfied': 0, 'satisfied': 1}

def encode_data(df):
    df_encoded = df.copy()
    df_encoded['Gender'] = df_encoded['Gender'].map(gender_map)
    df_encoded['Customer Type'] = df_encoded['Customer Type'].map(customer_map)
    df_encoded['Type of Travel'] = df_encoded['Type of Travel'].map(travel_map)
    df_encoded['Class'] = df_encoded['Class'].map(class_map)
    df_encoded['satisfaction'] = df_encoded['satisfaction'].map(target_map)
    return df_encoded

train_encoded = encode_data(train_df)
test_encoded = encode_data(test_df)

X_train = train_encoded.drop('satisfaction', axis=1)
y_train = train_encoded['satisfaction']
X_test = test_encoded.drop('satisfaction', axis=1)
y_test = test_encoded['satisfaction']

print("4. Chuẩn hóa dữ liệu (Scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Chỉ transform, tuyệt đối không fit lại trên test

print("5. Đang huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("6. Đánh giá mô hình...")
y_pred = model.predict(X_test_scaled)
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Báo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Không hài lòng', 'Hài lòng']))

print("7. Đang lưu mô hình và các bộ tham số...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# CẢI THIỆN: Lưu lại giá trị median để dùng trên Web App
with open('models/median_delay.pkl', 'wb') as f:
    pickle.dump(train_median_delay, f)

print("✅ HOÀN TẤT! Đã lưu model.pkl, scaler.pkl và median_delay.pkl vào thư mục models/.")