import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path, target_column):
        self.data = pd.read_csv(file_path)

        # iloc[rows, cols]
        self.y__raw = self.data[target_column]
        X_raw = self.data.drop(columns=[target_column])
        self.y = self.y_raw.astype('category').cat.codes.values
        X_encoded = pd.get_dummies(X_raw)
        self.X = X_encoded.values.astype(np.float64)
        self.feature_names = X_encoded.columns.tolist()
        print(f"Đã xử lý xong! Kích thước ma trận X: {self.X.shape}")
    
    def get_data(self):
        return self.X, self.y
    
    

    
