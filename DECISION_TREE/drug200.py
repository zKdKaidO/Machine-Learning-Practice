import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


url = "D:\\BKU\\ML\\DECISION_TREE\\drug200.csv"
df = pd.read_csv(url)

feature = [col for col in df.columns]
print("Features:", feature)
print(df.dtypes)

# Encode some features needed
features_to_encode = ['Sex', 'BP', 'Cholesterol', 'Drug']
_le = {}
for feature in features_to_encode:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    _le[feature] = le

X_train = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)

scores = cross_val_score(clf, X_train, y, cv=kf, scoring='accuracy')

print("--- RESULT K-FOLD (k=5) ---")
print(f"Scores for each time: {scores}")
print(f"Mean score: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std():.4f}")






