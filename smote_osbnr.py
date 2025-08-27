import time
import numpy as np
from imblearn.over_sampling import SMOTE
from PyOSBNR import OSBNR
from sklearn.base import BaseEstimator

class SMOTE_OSBNR(BaseEstimator):
    """
    SMOTE-OSBNR: A hybrid resampling method that applies OSBNR for noise reduction
    followed by SMOTE for minority oversampling.

    Parameters
    ----------
    k : int, default=5
        Number of clusters for the OSBNR algorithm.
    n_neighbors : int, default=5
        Number of neighbors to use for the SMOTE algorithm.
    """

    def __init__(self, k=5, n_neighbors=5):
        self.k = k
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        """
        Apply OSBNR noise reduction followed by SMOTE oversampling.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target vector.

        Returns
        -------
        X_resampled : ndarray
            The resampled feature matrix.
        y_resampled : ndarray
            The resampled target vector.
        """
        # Step 1: OSBNR noise reduction
        osbnr = OSBNR(k=self.k)
        X_osbnr, y_osbnr = osbnr.fit_resample(X, y)

        # Step 2: SMOTE oversampling
        smote = SMOTE(sampling_strategy='minority', k_neighbors=self.n_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_osbnr, y_osbnr)

        return X_resampled, y_resampled


if __name__ == "__main__":
    # Example usage (with timing)
    start = time.time()

    # Assuming X_train and y_train are defined
    sampler = SMOTE_OSBNR(k=28)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print("Resampled dataset shape:", X_resampled.shape)
    print("Processing time:", time.time() - start, "seconds")
