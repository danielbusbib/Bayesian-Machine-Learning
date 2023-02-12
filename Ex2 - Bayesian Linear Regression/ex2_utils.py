import numpy as np
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from matplotlib import pyplot as plt


def temps_example():
    """
    An example of how to load the temperature data. Note that there are 2 data sets here
    """
    X = np.load('jerus_daytemps.npy')
    hours = [2, 5, 8, 11, 14, 17, 20, 23]
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(hours, X[i, :], lw=2)
    plt.title('Daily Average Temperatures in November')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')

    y = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    plt.figure()
    plt.plot(hours, y, lw=2)
    plt.title('Temperatures on November 16 2020')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.show()


def confidence_interval_example():
    """
    An example of how random samples from the prior can be displayed, along with the mean function and confidence
    intervals
    """
    x = np.linspace(0, 2, 100)

    # create design matrix for 3rd order polynomial
    H = np.concatenate([x[:, None]**i for i in range(4)], axis=1)

    # create random prior
    mu = np.array([0, -3, 0, 1]) + .25*np.random.randn(4)
    S = .5*np.random.randn(H.shape[1], 100)
    S = S@S.T/100 + np.eye(mu.shape[0])*.001
    chol = np.linalg.cholesky(S)

    # find mean function
    mean = (H@mu[:, None])[:, 0]
    std = np.sqrt(np.diagonal(H@S@H.T))

    # plot mean with confidence intervals
    plt.figure()
    plt.fill_between(x, mean-std, mean+std, alpha=.5, label='confidence interval')
    for i in range(5):
        rand = (H@(mu[:, None] + chol@np.random.randn(chol.shape[-1], 1)))[:, 0]
        plt.plot(x, rand)
    plt.plot(x, mean, 'k', lw=2, label='mean')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.xlim([0, 2])
    plt.show()


if __name__ == '__main__':
    # temps_example()
    confidence_interval_example()
