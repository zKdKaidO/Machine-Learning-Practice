import numpy as np

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.var = None
        self.mean = None
        self.prior = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        # Label Encoding
        label_to_idx = {}
        for idx, label in enumerate(self.classes):
            label_to_idx[label] = idx
        
        # Calculate variance & mean for feature
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)

        self.prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.mean(X_c, axis=0)
            self.prior[idx] = X_c.shape[0] / float(X.shape[0])    
        
        

        


