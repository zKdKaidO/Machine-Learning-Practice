import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# TỪ NHỮNG THÔNG TIN TRONG CSV, SỬ DỤNG DECISION TREE ĐỂ DỰ ĐOÁN XEM BỘ PHIM CÓ THÀNH CÔNG HAY KHÔNG
url = "D:\\AI\\Machine-Learning-Practice\\DECISION_TREE\\movie_success_rate.csv"
df = pd.read_csv(url)
features = [col for col in df.columns]
print(df.dtypes)
features_to_labelencode = ['Title', 'Genre', 'Description', 'Director']
