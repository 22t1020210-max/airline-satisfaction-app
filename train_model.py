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
# Thay đổi đường dẫn nếu bạn để file ở vị trí khác
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Bỏ các cột không mang ý nghĩa phân tích (id và cột index bị lỗi Unnamed: 0)
cols_to_drop = ['Unnamed: 0', 'id']
train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], errors='ignore')
test_df = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')

print("2. Xử lý giá trị khuyết thiếu...")
# Điền các giá trị NaN trong cột 'Arrival Delay in Minutes' bằng giá trị trung vị (median)
train_df['Arrival Delay in Minutes'] = train_df['Arrival Delay in Minutes'].fillna(train_df['Arrival Delay in Minutes'].median())
test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(test_df['Arrival Delay in Minutes'].median())

print("3. Mã hóa dữ liệu (Encoding)...")
# Định nghĩa các quy tắc chuyển đổi chữ thành số. 
# CHÚ Ý: Ta dùng cách này thay vì One-Hot Encoding để sau này đưa lên Web App dễ nhập liệu hơn.
gender_map = {'Male': 0, 'Female': 1}
customer_map = {'disloyal Customer': 0, 'Loyal Customer': 1}
travel_map = {'Personal Travel': 0, 'Business travel': 1}
class_map = {'Eco': 0, 'Eco Plus': 1, 'Business': 2} # Phân loại theo thứ bậc (Ordinal)
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

# Tách đặc trưng (X) và nhãn dự đoán (y)
X_train = train_encoded.drop('satisfaction', axis=1)
y_train = train_encoded['satisfaction']
X_test = test_encoded.drop('satisfaction', axis=1)
y_test = test_encoded['satisfaction']

print("4. Chuẩn hóa dữ liệu (Scaling)...")
# Logistic Regression hoạt động tính toán khoảng cách toán học, 
# nên cần chuẩn hóa (Scale) để các cột giá trị lớn (như Flight Distance) không làm mờ các cột 1-5 sao.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("5. Đang huấn luyện mô hình Logistic Regression...")
# Khởi tạo và huấn luyện mô hình
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("6. Đánh giá mô hình...")
y_pred = model.predict(X_test_scaled)
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Không hài lòng', 'Hài lòng']))

print("7. Đang lưu mô hình...")
# Lưu mô hình dự đoán
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# LƯU Ý QUAN TRỌNG: Phải lưu cả Scaler. 
# Khi lên Web, người dùng nhập liệu vào, ta phải dùng chính scaler này để chuẩn hóa thì mô hình mới hiểu được.
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ HOÀN TẤT! Đã lưu model.pkl và scaler.pkl vào thư mục models/.")