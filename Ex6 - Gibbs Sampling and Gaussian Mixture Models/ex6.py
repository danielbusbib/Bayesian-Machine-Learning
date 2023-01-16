import numpy as np
from matplotlib import pyplot as plt
from ex6_utils import (plot_ims, load_MNIST, outlier_data, gmm_data, plot_2D_gmm, load_dogs_vs_frogs,
                       BayesianLinearRegression, poly_kernel, cluster_purity)
from scipy.special import logsumexp
from typing import Tuple


def outlier_regression(model: BayesianLinearRegression, X: np.ndarray, y: np.ndarray, p_out: float, T: int,
                       mu_o: float=0, sig_o: float=10) -> Tuple[BayesianLinearRegression, np.ndarray]:
    """
    Gibbs sampling algorithm for robust regression (i.e. regression assuming there are outliers in the data)
    :param model: the Bayesian linear regression that will be used to fit the data
    :param X: the training data, as a numpy array of shape [N, d] where N is the number of points and d is the dimension
    :param y: the regression targets, as a numpy array of shape [N,]
    :param p_out: the assumed probability for outliers in the data
    :param T: number of Gibbs sampling iterations to use in order to fit the model
    :param mu_o: the assumed mean of the outlier points
    :param sig_o: the assumed variance of the outlier points
    :return: the fitted model assuming outliers, as a BayesianLinearRegression model, as well as a numpy array of the
             indices of points which were considered as outliers
    """
    # todo <your code here>
    return None, None


class BayesianGMM:
    def __init__(self, k: int, alpha: float, mu_0: np.ndarray, sig_0: float, nu: float, beta: float,
                 learn_cov: bool=True):
        """
        Initialize a Bayesian GMM model
        :param k: the number of clusters to use
        :param alpha: the value of alpha to use for the Dirichlet prior over the mixture probabilities
        :param mu_0: the mean of the prior over the means of each Gaussian in the GMM
        :param sig_0: the variance of the prior over the means of each Gaussian in the GMM
        :param nu: the nu parameter of the inverse-Wishart distribution used as a prior for the Gaussian covariances
        :param beta: the variance of the inverse-Wishart distribution used as a prior for the covariances
        :param learn_cov: a boolean indicating whether the cluster covariances should be learned or not
        """
        # todo <your code here>

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the log-likelihood of each data point under each Gaussian in the GMM
        :param X: the data points whose log-likelihood should be calculated, as a numpy array of shape [N, d]
        :return: the log-likelihood of each point under each Gaussian in the GMM
        """
        if self.learn_cov:
            # todo <your code here>
            return None

        # todo <your code here>
        return None

    def cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Clusters the data according to the learned GMM
        :param X: the data points to be clustered, as a numpy array of shape [N, d]
        :return: a numpy array containing the indices of the Gaussians the data points are most likely to belong to,
                 as a numpy array with shape [N,]
        """
        # todo <your code here>
        return None

    def gibbs_fit(self, X:np.ndarray, T: int) -> 'BayesianGMM':
        """
        Fits the Bayesian GMM model using a Gibbs sampling algorithm
        :param X: the training data, as a numpy array of shape [N, d] where N is the number of points
        :param T: the number of sampling iterations to run the algorithm
        :return: the fitted model
        """
        # todo <your code here>
        return self


if __name__ == '__main__':
    # ------------------------------------------------------ section 2 - Robust Regression
    # ---------------------- question 2
    # load the outlier data
    x, y = outlier_data(50)
    # init BLR model that will be used to fit the data
    mdl = BayesianLinearRegression(theta_mean=np.zeros(2), theta_cov=np.eye(2), sample_noise=0.15)

    # sample using the Gibbs sampling algorithm and plot the results
    plt.figure()
    plt.scatter(x, y, 15, 'k', alpha=.75)
    xx = np.linspace(-0.2, 5.2, 100)
    for t in [0, 1, 5, 10, 25]:
        samp, outliers = outlier_regression(mdl, x, y, T=t, p_out=0.1, mu_o=4, sig_o=2)
        plt.plot(xx, samp.predict(xx), lw=2, label=f'T={t}')
    plt.xlim([np.min(xx), np.max(xx)])
    plt.legend()
    plt.show()

    # ---------------------- question 3
    # load the images to use for classification
    N = 1000
    ims, labs = load_dogs_vs_frogs(N)
    # define BLR model that should be used to fit the data
    mdl = BayesianLinearRegression(sample_noise=0.001, kernel_function=poly_kernel(2))
    # use Gibbs sampling to sample model and outliers
    samp, outliers = outlier_regression(mdl, ims, labs, p_out=0.01, T=50, mu_o=0, sig_o=.5)
    # plot the outliers
    plot_ims(ims[outliers], title='outliers')

    # ------------------------------------------------------ section 3 - Bayesian GMM
    # ---------------------- question 5
    # load 2D GMM data
    k, N = 5, 1000
    X = gmm_data(N, k)

    for i in range(5):
        gmm = BayesianGMM(k=50, alpha=.01, mu_0=np.zeros(2), sig_0=.5, nu=5, beta=.5)
        gmm.gibbs_fit(X, T=100)

        # plot a histogram of the mixture probabilities (in descending order)
        pi = None  # mixture probabilities from the fitted GMM
        plt.figure()
        plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
        plt.ylabel(r'$\pi_k$')
        plt.xlabel('cluster number')
        plt.show()

        # plot the fitted 2D GMM
        plot_2D_gmm(X, None, None, gmm.cluster(X))  # the second input are the means and the third are the covariances

    # ---------------------- questions 6-7
    # load image data
    MNIST, labs = load_MNIST()
    # flatten the images
    ims = MNIST.copy().reshape(MNIST.shape[0], -1)
    gmm = BayesianGMM(k=500, alpha=1, mu_0=0.5*np.ones(ims.shape[1]), sig_0=.1, nu=1, beta=.25, learn_cov=False)
    gmm.gibbs_fit(ims, 100)

    # plot a histogram of the mixture probabilities (in descending order)
    pi = None  # mixture probabilities from the fitted GMM
    plt.figure()
    plt.bar(np.arange(len(pi)), np.sort(pi)[::-1])
    plt.ylabel(r'$\pi_k$')
    plt.xlabel('cluster number')
    plt.show()

    # find the clustering of the images to different Gaussians
    cl = gmm.cluster(ims)
    clusters = np.unique(cl)
    print(f'{len(clusters)} clusters used')
    # calculate the purity of each of the clusters
    purities = np.array([cluster_purity(labs[cl == k]) for k in clusters])
    purity_inds = np.argsort(purities)

    # plot 25 images from each of the clusters with the top 5 purities
    for ind in purity_inds[-5:]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

    # plot 25 images from each of the clusters with the bottom 5 purities
    for ind in purity_inds[:5]:
        clust = clusters[ind]
        plot_ims(MNIST[cl == clust][:25].astype(float), f'cluster {clust}: purity={purities[ind]:.2f}')

