import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        return np.concatenate([(x**i)/np.sqrt(degree**i) for i in range(degree+1)], axis=1)
    return pbf


def poly_kernel(deg: float) -> Callable:
    """
    An implementation of the polynomial kernel
    :param deg: the degree of the polynomials used
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y): return (x@y.T + 1)**deg
    return kern


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray=None, theta_cov: np.ndarray=None, sample_noise: float=.1,
                 basis_functions: Callable=polynomial_basis_functions(1), kernel_function: Callable=None):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sample_noise:        the variance of the sample noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        :param kernel_function:     optional, a function that computes a kernel on two inputs; if given, then kernel
                                    regression is used, instead of regular BLR
        """
        self.mu = theta_mean                        # prior mean
        self.cov = theta_cov                        # prior covariance
        if theta_cov is not None:
            self.prec = np.linalg.inv(theta_cov)    # prior precision (inverse covariance)

        self.fit_mu = None                          # posterior mean
        self.fit_prec = None                        # posterior precision
        self.fit_cov = None                         # posterior covariance
        self.train = None                           # training data (changed only if using kernel regression)

        self.sig = sample_noise                     # sample noise used to fit model
        self.h = basis_functions                    # basis functions used by the model
        self.k = kernel_function                    # optional kernel functions used by the model
        self._samp = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample: bool=False) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data as a numpy array of shape [N, ...] where N is the number of training points
        :param y: the true regression values for the samples X as a numpy array of shape [N,]
        :param sample: a boolean indicating whether to sample theta from the posterior or to use the MMSE solution
        :return: the fitted model
        """
        # make sure the input data has the correct dimensions to use for regression
        if X.ndim == 1: X = X[:, None]
        elif X.ndim > 2: X = X.reshape(X.shape[0], -1)

        # if regular linear regression, find the full posterior (rec 4, eq 4.2)
        if self.k is None:
            H = self.h(X)
            self.fit_mu = H.T@y[:, None]/self.sig + self.prec@self.mu[:, None]
            self.fit_prec = self.prec + H.T@H/self.sig
            self.fit_cov = np.linalg.inv(self.fit_prec)
            self.fit_mu = np.linalg.solve(self.fit_prec, self.fit_mu)[:, 0]
            if sample: self.fit_mu += np.linalg.cholesky(self.fit_cov)@np.random.randn(*self.fit_mu.shape)

        # if kernel regression is used, fit model as if it is a GP (rec 8, eq 1.9)
        else:
            self.train = X.copy()
            K = self.k(X, X)
            gram = K + np.eye(X.shape[0]) * self.sig
            self.fit_prec = np.linalg.cholesky(gram)
            self.fit_mu = np.linalg.solve(self.fit_prec.T, np.linalg.solve(self.fit_prec, y))
            if sample: self._samp = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using the MMSE prediction
        :param X: the samples to predict as a numpy array of shape [N, ...]
        :return: the predictions for X
        """
        # make sure the input data has the correct dimensions to use for regression
        if X.ndim == 1: X = X[:, None]
        elif X.ndim > 2: X = X.reshape(X.shape[0], -1)

        # if regular linear regression, predict using rec 4, eq 4.5
        if self.k is None:
            # if the model hasn't been trained, return the prior prediction
            if self.fit_mu is None: return (self.h(X) @ self.mu[:, None])[:, 0]

            # otherwise, return the MMSE prediction
            return (self.h(X) @ self.fit_mu[:, None])[:, 0]

        # if kernel regression, predict using rec 8, eq 1.9
        else:
            if self.fit_mu is None: return np.zeros(X.shape[0])

            k = self.k(X, self.train)
            pred = k@self.fit_mu
            if not self._samp: return pred
            cov = self.k(X, X) - k @ np.linalg.solve(self.fit_prec.T, np.linalg.solve(self.fit_prec, k.T)) \
                  + np.eye(X.shape[0]) * 1e-8
            return pred + (np.linalg.cholesky(cov) @ np.random.randn(cov.shape[-1], 1))[:, 0]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data as a numpy array of shape [N, d]
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation as a numpy array of shape [N, d]
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # make sure the input data has the correct dimensions to use for regression
        if X.ndim == 1: X = X[:, None]
        elif X.ndim > 2: X = X.reshape(X.shape[0], -1)

        # if regular linear regression, predict using diagonal of rec 4, eq 4.4
        if self.k is None:
            # if the model hasn't been trained, return the prior standard deviation
            if self.fit_mu is None: return np.sqrt(np.diagonal(self.h(X) @ self.cov @ self.h(X).T))

            # otherwise, return the variance of the MMSE prediction
            return np.sqrt(np.diagonal(self.h(X) @ np.linalg.solve(self.fit_prec, self.h(X).T)))

        # if kernel regression, predict using diagonal of covariance of rec 8, eq 1.14
        else:
            if self.fit_mu is None: return np.sqrt(np.diagonal(self.k(X, X) + np.eye(X.shape[0])*self.sig))
            v = np.linalg.solve(self.fit_prec, self.k(X, self.train).T)
            return np.sqrt(np.clip(np.diagonal(self.k(X, X)) - np.sum(v * v, axis=0), 0, 1))

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the log-likelihood of each data point under the model
        :param X: the training data as a numpy array of shape [N, d]
        :param y: the true regression values for the samples X
        :return: a numpy array with shape [N,] containing the log-likelihood of each sample given the model
        """
        pr = self.predict(X)
        return -0.5*((pr-y)**2)/self.sig - 0.5*np.log(2*np.pi*self.sig)


def cluster_purity(cluster_labels: np.ndarray) -> float:
    """
    Calculates the purity of a cluster from a clustering algorithm using the true underlying labels
    :param cluster_labels: the ground truth labels for the elements in the cluster
    :return: the purity of the cluster
    """
    _, counts = np.unique(cluster_labels, return_counts=True)
    return np.max(counts)/len(cluster_labels)


def get_ellipse(mean, cov, color=None) -> Ellipse:
    """
    Create an ellipse of the contour line of a Gaussian
    :param mean: the mean of the Gaussian as a numpy array with shape (2,)
    :param cov: the covariance of the Gaussian as a numpy array with shape (2, 2)
    :param color: optional - the color of the ellipse as an RGB tuple ('edgecolor' in pyplot)
    :return: the ellipse as a matplotlib patch
    """
    v, w = np.linalg.eigh(cov)
    ang = 180. * np.arctan2(w[0, 1], w[0, 0]) / np.pi
    v = 2 * np.sqrt(2) * np.sqrt(v)
    return Ellipse(mean, v[0], v[1], 180 + ang, lw=2, facecolor='None',
                   edgecolor='k' if color is None else color)


def gmm_data(N: int=1000, k: int=5) -> np.ndarray:
    """
    Generate 2D GMM data
    :param N: the number of data points to sample
    :param k: the number of clusters in the GMM
    :return: an array of the sampled points as a numpy array of shape [N, 2]
    """
    np.random.seed(0)
    inds = np.random.choice(k, N, replace=True)

    # sample means
    rs = np.clip(np.random.randn(k), -1.5, 100) + 2
    thets = 2*np.pi*np.random.rand(k)
    means = np.concatenate([(rs*np.cos(thets))[:, None], (rs*np.sin(thets))[:, None]], axis=-1)

    # sample covariances
    covs = np.random.rand(k, 2, 2) - 0.5
    covs = covs@covs.transpose((0, 2, 1)) + \
           np.eye(covs.shape[-1])[None, ...]*np.sqrt(np.random.gamma(1, 1, size=k)[:, None, None])
    covs = np.linalg.cholesky(covs/10)

    # create data
    data = means[inds] + (covs[inds]@np.random.randn(N, 2, 1))[..., 0]
    np.random.seed(None)
    return data


def plot_2D_gmm(X: np.ndarray, means: np.ndarray, covs: np.ndarray, clusts: np.ndarray=None):
    """
    Plots the result of a 2D GMM
    :param X: the data the GMM was fitted to as a numpy array of shape [N, 2]
    :param means: the GMM means as a numpy array of shape [k, 2] where k is the number of clusters
    :param covs: the GMM covariances as a numpy array of shape [k, 2, 2]
    :param clusts: the clustering of the points to each of the Gaussians (optional)
    """
    # get standard matplotlib colors
    colors = plt.get_cmap('Set1').colors

    # if the clustering of the points was supplied, color them according to clusters
    if clusts is not None:
        inds = np.unique(clusts).astype(int)
        plt.figure()
        gca = plt.gca()
        for i, cl in enumerate(inds):
            plt.scatter(X[clusts == cl, 0], X[clusts == cl, 1], 20, color=colors[i%len(colors)], alpha=.5, marker='.')
            plt.plot(means[cl, 0], means[cl, 1], marker='+', markersize=10, color=colors[i%len(colors)])
            gca.add_patch(get_ellipse(means[cl], covs[cl], colors[i%len(colors)]))
        plt.axis('equal')
        plt.show()

    # if clustering wasn't supplied, just plot the data as black points
    else:
        plt.figure()
        gca = plt.gca()
        plt.scatter(X[:, 0], X[:, 1], 20, 'k', alpha=.5, marker='.')
        for i in range(means.shape[0]):
            plt.plot(means[i, 0], means[i, 1], marker='+', markersize=10, color=colors[i % len(colors)])
            gca.add_patch(get_ellipse(means[i], covs[i], colors[i % len(colors)]))
        plt.axis('equal')
        plt.show()


def plot_ims(image_list, title=''):
    """
    Plot images in a mosaic-style image
    :param image_list: a numpy array of images to plot
    :param title: the title of the plot
    """
    cols = np.where((len(image_list) % np.arange(1, np.floor(np.sqrt(len(image_list)) + 1))) == 0)[0][-1] + 1
    rows = len(image_list)//cols
    ims = np.concatenate([np.concatenate(image_list[i*rows:(i+1)*rows], axis=1) for i in range(cols)], axis=0)
    plt.figure(dpi=300)
    plt.imshow(ims.astype(float), cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()


def load_MNIST():
    """
    Loads a portion of the MNIST data set, along with the corresponding labels
    :return: two numpy arrays; the first is an array containing the images and the second containing the labels of the
             images
    """
    return np.load('mnist_partial.npy'), np.load('mnist_labs_partial.npy')


def load_dogs_vs_frogs(N: int=1000):
    """
    Load the image data for the dogs vs. frogs classification task
    :param N: number of images to return from each class
    :return: the tuple (X, y) where X is the concatenated array of shape [2N,32,32,3] of the images and y are the
             labels, as a numpy array of shape [2N,] where all dogs are labeled as 1 and all frogs as -1
    """
    dogs = np.load('dogs.npy')
    dogs = dogs[np.random.choice(dogs.shape[0], N, replace=False)]

    frogs = np.load('frogs.npy')
    frogs = frogs[np.random.choice(dogs.shape[0], N, replace=False)]

    X = np.concatenate([dogs, frogs], axis=0)
    y = np.ones(X.shape[0])
    y[-len(frogs):] = -1
    return X, y


def outlier_data(N: int):
    """
    Sample data with outliers
    :param N: number of points to sample
    :return: the sampled data with outliers as the tuple (x, y), both of with are numpy arrays of length N
    """
    np.random.seed(0)
    xx = np.linspace(0, 5, N)
    noise = .15
    y = 2*xx - 1 + np.sqrt(noise)*np.random.randn(len(xx))
    inds = [np.random.rand() <= .15 for i in range(len(xx))]
    y[inds] = -y[inds] + 9
    np.random.seed(None)
    return xx, y


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


def outlier_example():
    """
    An example of how to use the outlier_data function
    """
    N = 30
    x, y = outlier_data(N)
    plt.figure()
    plt.scatter(x, y, 15)
    plt.show()


def purity_example():
    """
    An example of how to use the cluster_purity function
    """
    # load the images and ground truth labels
    ims, labels = load_MNIST()
    ims = ims.reshape((ims.shape[0], 28, 28))

    # create a random clustering of the first 25 images and check the "cluster" purity of this cluster
    plot_ims(ims[:25], title=f'purity={cluster_purity(labels[:25])}')

    # calculate the purity of a clustering of only 1s (should be 1)
    inds = labels == 1
    plot_ims(ims[inds][:25], title=f'purity={cluster_purity(labels[inds][:25])}')


if __name__ == '__main__':
    BLR_fit_example()
    outlier_example()
    purity_example()
