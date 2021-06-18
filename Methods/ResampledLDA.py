"""
This class contains implementations for ShrinkageLinearDiscriminantAnalysis and ResampledLinearDiscriminantAnalysis.
Both are based on the implementation by Jan Sosulski, but ResampledLinearDiscriminantAnalysis is modified with a
resampling procedure which is not from Jan Sosulski.
I have added his license in here:

Copyright 2020 Jan Sosulski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2021 GitHub, Inc.
"""
import sklearn
import numpy as np
import pyriemann
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def subtract_classwise_means(xTr, y):
    n_classes = 2
    n_features = xTr.shape[0]
    X = np.zeros((n_features, 0))
    cl_mean = np.zeros((n_features, n_classes))
    for cur_class in range(n_classes):
        class_idxs = y == cur_class
        cl_mean[:, cur_class] = np.mean(xTr[:, class_idxs], axis=1)

        X = np.concatenate([
            X,
            xTr[:, class_idxs] - np.dot(cl_mean[:, cur_class].reshape(-1, 1),
                                        np.ones((1, np.sum(class_idxs))))],
            axis=1)
    return X, cl_mean

def diag_indices_with_offset(p, offset):
    idxdiag = np.diag_indices(p)
    idxdiag_with_offset = list()
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[0]]))
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[1]]))
    return tuple(idxdiag_with_offset)


def _shrinkage(X: np.ndarray, gamma=None, T=None, S=None, block=False,
               N_channels=31, N_times=5, standardize=True) -> Tuple[np.ndarray, float]:

    p, n = X.shape

    if standardize:
        sc = StandardScaler()  # standardize_featurestd features
        X = sc.fit_transform(X.T).T
    Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    if S is None:
        S = np.matmul(Xn, Xn.T)
    Xn2 = np.square(Xn)
    idxdiag = np.diag_indices(p)

    # Target = B
    nu = np.mean(S[idxdiag])
    if T is None:
        if block:
            nu = list()
            for i in range(N_times):
                idxblock = diag_indices_with_offset(N_channels, i*N_channels)
                nu.append([np.mean(S[idxblock])] * N_channels)
            nu = [sl for l in nu for sl in l]
            T = np.diag(np.array(nu))
        else:
            T = nu * np.eye(p, p)

    # Ledoit Wolf
    V = 1. / (n - 1) * (np.matmul(Xn2, Xn2.T) - np.square(S) / n)
    if gamma is None:
        gamma = n * np.sum(V) / np.sum(np.square(S - T))
    if gamma > 1:
        print("logger.warning('forcing gamma to 1')")
        gamma = 1
    elif gamma < 0:
        print("logger.warning('forcing gamma to 0')")
        gamma = 0

    Cstar = (gamma * T + (1 - gamma) * S) / (n - 1)
    if standardize:  # scale back
        Cstar = sc.scale_[np.newaxis, :] * Cstar * sc.scale_[:, np.newaxis]
    return Cstar, gamma


class ShrinkageLinearDiscriminantAnalysis(
        sklearn.base.BaseEstimator,
        sklearn.linear_model._base.LinearClassifierMixin):

    def __init__(self, priors=None, only_block=False, N_times=5, N_channels=31, pool_cov=True, standardize_shrink=True):
        self.only_block = only_block
        self.priors = priors
        self.N_times = N_times
        self.N_channels = N_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink

    def fit(self, X_train, y):
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError('currently only binary class supported')
        assert len(X_train) == len(y)
        xTr = X_train.T

        n_classes = 2
        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
            # self.priors = np.array([1./n_classes] * n_classes)
        else:
            priors = self.priors

        X, cl_mean = subtract_classwise_means(xTr, y)
        if self.pool_cov:
            C_cov, C_gamma = _shrinkage(X, N_channels=self.N_channels, N_times=self.N_times,
                                        standardize=self.standardize_shrink)
        else:
            n_classes = 2
            C_cov = np.zeros((xTr.shape[0], xTr.shape[0]))
            for cur_class in range(n_classes):
                class_idxs = y == cur_class
                x_slice = X[:, class_idxs]
                C_cov += priors[cur_class] * _shrinkage(x_slice)[0]

        if self.only_block:
            C_cov_new = np.zeros_like(C_cov)
            for i in range(self.N_times):
                idx_start = i * self.N_channels
                idx_end = idx_start + self.N_channels
                C_cov_new[idx_start:idx_end, idx_start:idx_end] = C_cov[idx_start:idx_end, idx_start:idx_end]
            C_cov = C_cov_new

        C_invcov = np.linalg.pinv(C_cov)
        # w = np.matmul(C_invcov, cl_mean)
        w = np.linalg.lstsq(C_cov, cl_mean)[0]
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + np.log(priors)

        if n_classes == 2:
            w = w[:, 1] - w[:, 0]
            b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b

    def predict_proba(self, X):
        """Estimate probability.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        """Estimate log probability.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


class ResampledLinearDiscriminantAnalysis(sklearn.base.BaseEstimator, sklearn.linear_model._base.LinearClassifierMixin):
    def __init__(self, priors=None, only_block=False, N_times=5, N_channels=31, pool_cov=True, standardize_shrink=True):
        self.only_block = only_block
        self.priors = priors
        self.N_times = N_times
        self.N_channels = N_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink

    def fit(self, X_train, y):
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError('currently only binary class supported')
        assert len(X_train) == len(y)
        xTr = X_train.T

        n_classes = 2
        if self.priors is None:
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
        else:
            priors = self.priors

        X, cl_mean = subtract_classwise_means(xTr, y)
        cov = self.resampled_cov(X)
        w = np.linalg.lstsq(cov, cl_mean, rcond=None)[0]
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + np.log(priors)

        if n_classes == 2:
            w = w[:, 1] - w[:, 0]
            b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b

    def predict_proba(self, X):
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        return np.column_stack([1 - prob, prob])

    def resampled_cov(self, X, remove_outliers=False):
        iterate = 100
        cov_shape, gamma_shape = _shrinkage(X)
        covariance = np.zeros((iterate, cov_shape.shape[0], cov_shape.shape[0]))
        data = X.T
        for i in range(0, iterate):
            n_samples=int(data.shape[0])
            resampled = sklearn.utils.resample(data, n_samples=n_samples)
            resampled = resampled.T
            cov, gamma = _shrinkage(resampled)
            covariance[i, :, :] = cov
        if remove_outliers:
            final_est = self.matrix_outliers(covariance)
        else:
            print("No outlier removal and calculating mean")
            final_est = pyriemann.utils.mean.mean_covariance(covariance, metric='riemann')
        return final_est

    def matrix_outliers(self, covariance):
        mean = pyriemann.utils.mean.mean_covariance(covariance, metric='riemann')
        dists = np.zeros((100))
        for i in range(0, covariance.shape[0]):
            dists[i] = pyriemann.utils.distance.distance_riemann(mean, covariance[i, :, :])
        sorted_dists = np.sort(dists)
        outliers = sorted_dists[int(0.4 * len(sorted_dists)):len(sorted_dists)]
        print("got mean & removing 60p outlier BNCI1")
        new_covs = np.zeros((covariance.shape[0] - len(outliers), covariance.shape[1], covariance.shape[2]))
        c = 0
        for i in range(covariance.shape[0]):
            if dists[i] not in outliers:
                new_covs[c, :, :] = covariance[i, :, :]
                c = c + 1

        final_cov_removed_outliers = pyriemann.utils.mean.mean_covariance(new_covs, metric='riemann')
        return final_cov_removed_outliers
