import numpy as np
import pandas as pd
"""
This file is used to execute salmon_seabass.csv only
"""
class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep=";")
        print(self.data.head())

        # iloc[rows, cols]
        y_raw = self.data.iloc[:, -1]
        X_raw = self.data.drop(columns=self.data.columns[-1])
        self.y = y_raw.values.astype(np.int64)
        self.X = X_raw.values.astype(np.float64)
        print(f"Shape X: {self.X.shape}, Shape Y: {self.y.shape}")

    def get_data(self):
        return self.X, self.y
     # Data scaling 
    def scaling(self, type="min-max"):
        n_col = self.X.shape[1]
        if type == "min-max":
            for i in range(self.n_col):
                X_max = self.X[:, i].max()
                X_min = self.X[:, i].min()
                denominator = (X_max - X_min) if (X_max - X_min) != 0 else (X_max - X_min + 1e-6)
                self.X[:, i] = (self.X[:, i] - X_min) / denominator

        elif type == "z-score":
            for i in range(self.n_col):
                mean = np.mean(self.X[:, i], axis=0)
                var = np.var(self.X[:, i], axis=0)
                denominator = np.sqrt(var) if var != 0 else (np.sqrt(var) + 1e-6)
                self.X[:, i] = (self.X[:, i] - mean) / denominator 

        else:
            raise ValueError(f"Unknown normalization type: {type}, type again with 'min-max' or 'z-score' only!")
        
        def split_data(self, train_ratio=0.8, shuffle=True, seed=None):
            N = self.X.shape[0]
            N_train = int(N * train_ratio)
            np.arange(N)
            idx = np.random.permutation(N)
            train_idx, test_idx = idx[:N_train], idx[N_train:]
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_test,  y_test  = self.X[test_idx],  self.y[test_idx]
            return X_train, y_train, X_test, y_test
    
        def check_balance(self):
            

    

    
