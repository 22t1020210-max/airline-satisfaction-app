# Airline Passenger Satisfaction Prediction App

## Giới thiệu dự án
Đây là ứng dụng Machine Learning dự đoán mức độ hài lòng của hành khách hàng không dựa trên các trải nghiệm bay. 
Dự án sử dụng thuật toán **Logistic Regression** để phân loại nhị phân (Hài lòng / Không hài lòng) dựa trên bộ dữ liệu Airline Passenger Satisfaction từ Kaggle. Giao diện Web App được xây dựng hoàn toàn bằng **Streamlit**.

## Nguồn dữ liệu
Bộ dữ liệu gồm các đặc trưng về thông tin hành khách (Tuổi, Giới tính, Hạng vé...) và điểm đánh giá từ 1-5 sao cho các dịch vụ (Wifi, Đồ ăn, Chỗ ngồi, Thái độ nhân viên...).
* Nguồn: [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

## Cấu trúc thư mục
```text
airline-satisfaction-app/
│── app.py                 # File code chính giao diện Streamlit
│── train_model.py         # Script tiền xử lý và huấn luyện mô hình
│── requirements.txt       # Danh sách thư viện môi trường
│── README.md              # Tài liệu giới thiệu
│── models/
│   ├── model.pkl          # Mô hình Logistic Regression đã huấn luyện
│   └── scaler.pkl         # Bộ chuẩn hóa dữ liệu
└── data/
    ├── train.csv          # Dữ liệu huấn luyện (dùng cho trang EDA)
    └── test.csv           # Dữ liệu kiểm thử