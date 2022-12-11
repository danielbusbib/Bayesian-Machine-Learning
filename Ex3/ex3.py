import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    s = (y - model.h(X) @ map).T @ (y - model.h(X) @ map)
    N = X.shape[0]
    p = len(model.h(X)[1])
    a = 0.5 * np.log(np.linalg.det(map_cov) / np.linalg.det(sig))
    b = -0.5 * ((map - mu).T @ np.linalg.pinv(sig) @ (map - mu) + (1 / n) * s + N * np.log(n))
    c = -0.5 * p * np.log(2 * np.pi)
    return a + b + c


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: -x ** 4 + 3 * x ** 2 + 50 * np.sin(x / 6)
    f3 = lambda x: .5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: np.cos(x * 4) + 4 * np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    labels = ["worst evidence", "best evidence"]
    c = ["orange", "blue"]

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        evs = np.zeros(len(degrees))
        plt.subplot(1, 2, 1)

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            evs[j] = ev

        plt.plot(degrees, evs)
        plt.title(f"function {i + 1},"
                  f"best model : d = {np.argmax(evs) + 2}")
        plt.xlabel("degree")
        plt.ylabel("log evidence")

        # plot evidence versus degree and predicted fit
        max_deg, min_deg = np.argmax(evs) + 2, np.argmin(evs) + 2
        plt.subplot(1, 2, 2)
        plt.plot(x, y, 'o')

        for j, d in enumerate([min_deg, max_deg]):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            blr = BayesianLinearRegression(mean, cov, noise_var, pbf)
            blr.fit(x, y)

            pred, std = blr.predict(x), blr.predict_std(x)

            # plot points and prediction
            plt.fill_between(x, pred - std, pred + std, alpha=.5, color=c[j])
            plt.plot(x, pred, lw=2, label=labels[j], color=c[j])
            plt.legend()
            plt.xlabel('$x$')
            plt.ylabel(r'$f_{\theta}(x)$')

        plt.show()

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev

    # plot log-evidence versus amount of sample noise
    noise_max_evidence = np.round(noise_vars[np.argmax(evs)], 2)
    plt.plot(noise_vars, evs)
    plt.title(f"Log-evidence score for each of the models"
              f" as a function of the sample noise.\n"
              f"Noise with max evidence: {noise_max_evidence}")
    plt.xlabel("sample noise")
    plt.ylabel("log evidence")
    plt.show()


if __name__ == '__main__':
    main()
