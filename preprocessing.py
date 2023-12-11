import numpy as np

class LDA():
    def __init__(self) -> None:
        pass

    def fit(self, X, y) -> None:
        pass

    def transform(self, X, k_features) -> np.ndarray:
        pass

    def fit_transform(self, X, y, k_features) -> np.ndarray:
        pass

class PCA():
    def __init__(self) -> None:
        pass

    def fit(self, X, y) -> None:
        pass

    def transform(self, X, k_features) -> np.ndarray:
        pass

    def fit_transform(self, X, y, k_features) -> np.ndarray:
        pass


class FactorAnalysis():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        Fit the model with X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        #create covariance matrix
        cov_matrix = np.cov(X_centered.T)

        #compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        #sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        #get the first n_components eigenvectors
        self.L = eigenvectors[:, :self.n_components]

        #get the first n_components eigenvalues
        self.eigenvalues_ = eigenvalues[:self.n_components]

        #compute the explained variance
        self.explained_variance_ = self.eigenvalues_ / np.sum(self.eigenvalues_)

        #compute the explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            New data to transform.
        """
        X_centered = X - self.mean_

        # X = L*F
        # L^t*X = L^t*L*F 
        # (L^t*L)^-1 * L^t*X = F

        #compute the factor matrix
        F = np.linalg.inv(self.L.T.dot(self.L)).dot(self.L.T).dot(X_centered.T).T
        return F
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        self.fit(X)
        return self.transform(X)