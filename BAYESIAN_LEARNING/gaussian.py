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
            self.var[idx, :] = np.var(X_c, axis=0)
            self.prior[idx] = X_c.shape[0] / float(X.shape[0])   

    def _pdf(self, X_new, class_idx):
        var = self.var[class_idx, :] + 1e-9
        mean = self.mean[class_idx, :]
        numerator = np.exp(-(X_new - mean)**2 / (2 * var))
        denominator = (np.sqrt(2 * np.pi * var)) 
        return numerator / denominator

    def predict(self, X_new):
        # Calculate y_pred for X_new
        y_pred = []
        
        for x in X_new:
            class_scores = []
            for idx, c in enumerate(self.classes):
                prior_log = np.log(self.prior[idx])
                pdf_vals = self._pdf(x, idx)
                likelihood_log = np.sum(np.log(pdf_vals + 1e-9))
                class_idx_point = prior_log + likelihood_log
                class_scores.append(class_idx_point)

            best_class_idx = np.argmax(class_scores)
            y_pred.append(self.classes[best_class_idx])

        return np.array(y_pred)

        
        

        


