import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        return np.concatenate([(x**i)[:, None]/np.sqrt(degree**i) for i in range(degree+1)], axis=1)
    return pbf


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.mu = theta_mean                        # prior mean
        self.cov = theta_cov                        # prior covariance
        self.prec = np.linalg.inv(theta_cov)        # prior precision (inverse covariance)

        self.fit_mu = None                          # posterior mean
        self.fit_prec = None                        # posterior precision
        self.fit_cov = None                         # posterior covariance

        self.sig = sig                              # sample noise used to fit model
        self.h = basis_functions                    # basis functions used by the model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.h(X)
        self.fit_mu = H.T@y[:, None]/self.sig + self.prec@self.mu[:, None]
        self.fit_prec = self.prec + H.T@H/self.sig
        self.fit_cov = np.linalg.inv(self.fit_prec)
        self.fit_mu = np.linalg.solve(self.fit_prec, self.fit_mu)[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # if the model hasn't been trained, return the prior prediction
        if self.fit_mu is None: return (self.h(X) @ self.mu[:, None])[:, 0]

        # otherwise, return the MMSE prediction
        return (self.h(X) @ self.fit_mu[:, None])[:, 0]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # if the model hasn't been trained, return the prior standard deviation
        if self.fit_mu is None: return np.sqrt(np.diagonal(self.h(X) @ self.cov @ self.h(X).T) + self.sig)

        # otherwise, return the variance of the MMSE prediction
        return np.sqrt(np.diagonal(self.h(X) @ np.linalg.solve(self.fit_prec, self.h(X).T)) + self.sig)


def BLR_fit_example():
    """
    An example of how to use the supplied Bayesian linear regression and polynomial basis functions code
    """
    # define function
    f = lambda x: np.sin(2*x)
    noise = 0.1
    x = np.linspace(-np.pi, np.pi, 200)
    y = f(x) + np.sqrt(noise)*np.random.randn(len(x))

    # define model parameters
    deg = 10
    pbf = polynomial_basis_functions(deg)
    n = np.arange(deg+1) + 1
    mean, cov = np.zeros(deg+1), np.diag(n)
    model = BayesianLinearRegression(mean, cov, noise, pbf).fit(x, y)

    # make prediction of MMSE and standard deviation
    pred, std = model.predict(x), model.predict_std(x)

    # plot points and prediction
    plt.figure()
    plt.plot(x, y, 'o')
    plt.fill_between(x, pred-std, pred+std, alpha=.5)
    plt.plot(x, pred, 'k', lw=2)
    plt.xlabel('$x$')
    plt.ylabel(r'$f_{\theta}(x)$')
    plt.show()


def load_prior():
    """
    An example of how to load the supplied prior for questions 5-7
    :return: the mean and covariance of the prior needed for questions 5-7
    """
    params = np.load('temp_prior.npy')

    # load the mean of the prior
    mean = params[:, 0]
    # load the covariance of the prior
    cov = params[:, 1:]

    print('Shape of the mean of the temperatures data set:', mean.shape)
    print('Shape of the covariance of the temperatures data set:', cov.shape)
    return mean, cov


if __name__ == '__main__':
    load_prior()
    BLR_fit_example()
