import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components # K
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        N = X.shape[0]

        covariance_mat = np.dot(X_centered.T, X_centered) / (N-1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_mat)
        
        sorted_idx = np.argsort(eigenvalues)[::-1] # sort from high to low
        sorted_eigenvalues = eigenvalues[sorted_idx]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]

        filter_eigenvalues, filter_eigenvectors = sorted_eigenvalues[:self.n_components], sorted_eigenvectors[:, :self.n_components]
        self.components = filter_eigenvectors
    
    def transform(self, X):
        z = np.dot((X - self.mean), self.components) 
        return z
    
    def inverse_transform(self, z):
        X = np.dot(z, self.components.T) + self.mean
        return X

        