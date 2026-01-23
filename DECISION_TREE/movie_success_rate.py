import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# TỪ NHỮNG THÔNG TIN TRONG CSV, SỬ DỤNG DECISION TREE ĐỂ DỰ ĐOÁN XEM BỘ PHIM CÓ THÀNH CÔNG HAY KHÔNG
url = "D:\AI\Machine-Learning-Practice\DECISION_TREE\movie_success_rate.csv"
df = pd.read_csv(url)
features = [col for col in df.columns]
print(df.dtypes)
# ==============================================================================
# BƯỚC 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# ==============================================================================

# 1. Xử lý cột Target (Success) bị thiếu -> XÓA DÒNG ĐÓ
# Kiểm tra xem có bao nhiêu dòng bị thiếu Success
missing_success = df['Success'].isnull().sum()
if missing_success > 0:
    print(f"Phát hiện {missing_success} dòng thiếu nhãn 'Success'. Đang xóa...")
    df = df.dropna(subset=['Success'])

# 2. Xử lý các cột Features (Đặc trưng) bị thiếu (Revenue, Metascore...)
# Với Decision Tree, ta có thể điền 0 hoặc trung bình -> điền trung bình (median)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if col != 'Success': # Trừ cột target ra
        df[col] = df[col].fillna(df[col].median())

# Các cột chữ (Object) nếu thiếu thì điền chuỗi rỗng
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].fillna("Unknown")

print("Kích thước sau khi làm sạch:", df.shape)
# ==============================================================================
# BƯỚC 2: XỬ LÝ MÃ HÓA (Encoding)
# ==============================================================================
features_to_LabelEncode = ['Title', 'Director']
_le = {}
for feature in features_to_LabelEncode:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    _le[feature] = le
"""
Chú ý column 'Actors': list chứa tên các actors, các actor cách nhau bởi ", ", ví dụ như: "A, B, C, D"
Như vậy ta ko thể dùng LabelEncoder để xử lí như những cái khác
LabelEncoder biến những loại khác về các giá trị integer (Ví dụ: Title có: A, B và C -> LabelEncoder biến: A -> 1; B -> 2; C -> 3)
Vậy thì với cột Actors: ta cần tên của mỗi actor sẽ đc biểu thị bằng 1 giá trị (vì 1 actor có quyền đóng nhiều phim) -> MultiLabelBinarizer
"""

"""
MultiLabelBinarizer
Ví dụ: 1 phim có thể có nhiều thể loại: x = A, B, C,...
-> tổng hợp tất cả thể loại: ['A' 'B' 'C' 'D' 'E']
(nếu phim có thể loại nào thì là 1 ở vị trí đó, ko thì là 0)
-> sau khi MLB: x = [1 1 1 0 0]
"""
features_to_MultiLabel = ['Genre', 'Actors']
_mlb = {}
for feature in features_to_MultiLabel:
    # df[feature] = [[Action,Adventure,Sci-Fi], [Adventure,Mystery,Sci-Fi],...] (for example)
    df[feature] = df[feature].fillna("").apply(lambda x: [item.strip() for item in x.split(",")] if x else []) # strip() mặc định là space -> "  Hello" thành "Hello"
    mlb = MultiLabelBinarizer()
    df[feature] = mlb.fit_transform(df[feature])
    _mlb[feature] = mlb

target = 'Success'
# PHẦN EXCLUDE NÀY CÀNG NGÀY SẼ ĐƯỢC UPDATE THÊM SAU KHI CHẠY NHIỀU LẦN VÀ LOẠI BỎ NHỮNG THỨ ẢNH HƯỞNG LỚN TỚI SUCCESS QUÁ 
# LÚC ĐÓ NÓ KO CÒN DỰ ĐOÁN NỮA MÀ CÓ THỂ LÀ HỌC VẸT (VÌ 1 FEATURE ẢNH HƯỞNG QUÁ nHIỀU TỚI KẾT QUẢ TARGET)
exclude = [
    'Description', 'Revenue (Millions)', 'Votes', 'Rating', 'Metascore', 'Rank', 'Runtime (Minutes)', 'Title', 'Director'
]
X = df.drop(columns=[target] + exclude)
y = df[target]

# Train 70%, Val 15%, Test 15%
# BƯỚC ĐẦU CHIA TRAIN 70% -> 30% CÒN LẠI CHO TEMP
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 30% TEMP CÒN LẠI CHIA ĐÔI -> 15% CHO TEST VÀ 15% VAL
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = DecisionTreeClassifier(random_state=42) 
model.fit(X_train, y_train)

# Trên validation
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Trên test
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

scores = cross_val_score(model, X_train, y_train, cv=5)
print("CV mean accuracy:", scores.mean())

# Lấy độ quan trọng của các features
importances = model.feature_importances_
feature_names = X_train.columns

# Sắp xếp để dễ nhìn
indices = np.argsort(importances)[::-1]

# In ra Top 10 đặc trưng quan trọng nhất
print("--- Top 10 đặc trưng quyết định kết quả (SẼ ĐƯỢC LOẠI TRỪ NẾU % QUÁ CAO) ---")
for i in range(10):
    if i < len(indices):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
