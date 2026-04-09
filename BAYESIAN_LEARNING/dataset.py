import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        # iloc[rows, cols]
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
    
    def get_data(self):
        return self.X, self.y
    
    

    
