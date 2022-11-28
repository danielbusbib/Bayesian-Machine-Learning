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
        return np.vander(x, degree + 1, 1)

    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """

    def gbf(X):
        n = X.shape[0]
        b = centers.shape[0]
        values = []
        for x in X:
            values.append(1)
            for c in centers:
                values.append(np.exp(-np.sum((x - c) ** 2) / (2 * beta ** 2)))
        mat = np.reshape(values, (n, b + 1))  # reshape to design matrix size
        return mat

    return gbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """

    def csbf(X: np.ndarray):
        n = X.shape[0]
        k = knots.shape[0]
        values = []
        for x in X:
            values += [1, x, x ** 2, x ** 3]
            for knot in knots:
                m = x - knot
                if m >= 0:
                    values.append(m ** 3)
                else:
                    values.append(0)

        mat = np.reshape(values, (n, k + 4))  # reshape to design matrix size
        return mat

    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.w)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.theta_mean = theta_mean
        self.theta_cov = theta_cov
        self.sig = sig
        self.basis_functions = basis_functions
        self.mu, self.cov = None, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.basis_functions(X)
        M = self.sig * np.identity(X.shape[0]) + H @ self.theta_cov @ H.T
        self.cov = self.theta_cov - (self.theta_cov @ H.T @ np.linalg.inv(M) @ H @ self.theta_cov)
        self.mu = self.theta_mean + self.theta_cov @ H.T @ np.linalg.inv(M) @ (y - H @ self.theta_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_functions(X) @ self.mu

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
        h_x = self.basis_functions(X)
        c = h_x @ self.theta_cov @ h_x.T
        c = c + np.identity(c.shape[0]) * self.sig ** 2
        return np.sqrt(np.diag(c))

    def posterior_sample(self, X: np.ndarray, p) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        if p:  # posterior sample
            return self.basis_functions(X) @ np.random.multivariate_normal(self.theta_mean, self.theta_cov)
        else:  # prior sample
            return self.basis_functions(X) @ np.random.multivariate_normal(self.mu, self.cov)


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.w = None
        self.basis_functions = basis_functions

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        Hx = self.basis_functions(X)
        self.w = np.transpose(np.linalg.pinv(np.transpose(Hx))) @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        return self.basis_functions(X) @ self.w

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_prior(model, title=""):
    x = np.arange(0, 24, .1)
    # plot mean with confidence intervals
    plt.figure()
    mean = model.basis_functions(x) @ model.theta_mean
    std = np.sqrt(np.diagonal(model.basis_functions(x) @ model.theta_cov @ model.basis_functions(x).T))
    plt.fill_between(x, mean - std, mean + std, alpha=.5, label='confidence interval')
    for i in range(5):
        plt.plot(x, model.posterior_sample(x, p=True))
    plt.plot(x, mean, c='black', label='mean')
    plt.legend()
    plt.xlabel("hour")
    plt.ylabel("temperature")
    plt.title(f"prior - {title}")
    plt.xlim([0, 24])
    plt.show()


def main():
    # load the data for November 16 2020
    nov16 = np.load('nov162020.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    train_hours = nov16_hours[:len(nov16) // 2]
    test = nov16[len(nov16) // 2:]
    test_hours = nov16_hours[len(nov16) // 2:]
    # setup the model parameters
    degrees = [3, 7]
    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours)) ** 2):.2f}')

        # plot graphs for linear regression part
        plt.scatter(train_hours, train, label="train true")
        plt.scatter(test_hours, test, label="test true")
        plt.plot(nov16_hours, ln.predict(nov16_hours), label="preicted")
        plt.title(f"Polynomial regression with d={d}")
        plt.xlabel("hour"), plt.ylabel("temperature")
        plt.legend()
        plt.show()
    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees
    beta = 2.5  # lengthscale for Gaussian basis functions

    # sets of centers S_1, S_2, and S_3
    centers = [np.array([6, 12, 18]),
               np.array([4, 8, 12, 16, 20]),
               np.array([3, 6, 9, 12, 15, 18, 21])]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    def plot_posterior(blr, title):
        blr.fit(train_hours, train)
        std = blr.predict_std(nov16_hours)
        mmse = blr.predict(nov16_hours)
        plt.fill_between(nov16_hours, mmse - std, mmse + std, alpha=.5, label='confidence interval')
        for i in range(5):
            plt.plot(nov16_hours, blr.posterior_sample(nov16_hours, p=False))
        plt.plot(nov16_hours, mmse, c='black', label='MMSE')
        plt.scatter(train_hours, train, label="train true")
        plt.scatter(test_hours, test, label="test true")
        plt.legend()
        plt.xlabel("hour")
        plt.ylabel("temperature")
        plt.title(f"posterior - {title} - "
                  f"AVG ERR OF MMSE = {np.round(np.square(test - blr.predict(test_hours)).mean(), 2)}")
        plt.legend()
        plt.show()
        # print(f"MMSE AVG SQUARED ERROR: {np.square(test - blr.predict(test_hours)).mean()}")

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        # plot prior graphs
        plot_prior(blr, f"polynomial basis function, d={deg}")

        # plot posterior graphs
        plot_posterior(blr, f"polynomial basis function, d={deg}")

    # ---------------------- Gaussian basis functions
    for ind, c in enumerate(centers):
        rbf = gaussian_basis_functions(c, beta)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        plot_prior(blr, f"Gaussian basis function, Set {ind + 1}")

        # plot posterior graphs
        plot_posterior(blr, f"Gaussian basis function, Set {ind + 1}")

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)

        # plot prior graphs
        plot_prior(blr, f"Cubic regression spline, K{ind + 1}")

        # plot posterior graphs
        plot_posterior(blr, f"Cubic regression spline, K{ind + 1}")


if __name__ == '__main__':
    main()
