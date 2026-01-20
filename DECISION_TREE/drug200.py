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
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)

scores_gini = cross_val_score(clf_gini, X_train, y, cv=kf, scoring='accuracy')
scores_entropy = cross_val_score(clf_entropy, X_train, y, cv=kf, scoring='accuracy')

print("--- RESULT K-FOLD (k=5) WITH GINI ---")
print(f"Scores for each time: {scores_gini}")
print(f"Mean score: {scores_gini.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores_gini.std():.4f}")

"""
--- RESULT K-FOLD (k=5) WITH GINI ---
Scores for each time: [1.    1.    0.975 0.975 1.   ]
Mean score: 99.00%
Standard Deviation: 0.0122
"""
print("--- RESULT K-FOLD (k=5) WITH ENTROPY ---")
print(f"Scores for each time: {scores_entropy}")
print(f"Mean score: {scores_entropy.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores_entropy.std():.4f}")
"""
--- RESULT K-FOLD (k=5) WITH ENTROPY ---
Scores for each time: [1.    1.    0.975 0.975 1.   ]
Mean score: 99.00%
Standard Deviation: 0.0122
"""



