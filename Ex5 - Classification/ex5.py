import numpy as np
from matplotlib import pyplot as plt
from ex5_utils import load_im_data, GaussianProcess, RBF_kernel, accuracy, Gaussian, plot_ims


def main():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    def mmse_mu(X, mu0):
        # calculate posterior mu
        a = (1 / sig) * X.sum(axis=0) + (1 / sig_0) * mu0
        b = (X.shape[0] * (1 / sig)) + (1 / sig_0)
        return a / b

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x = np.concatenate([x_p, x_m])

    plt.scatter(x_p[:, 0], x_p[:, 1], s=12, label=" + ")
    plt.scatter(x_m[:, 0], x_m[:, 1], s=12, label=" - ")
    plt.title(f'plot of points that were sampled from each class')
    plt.legend()
    plt.show()

    def decision_boundary(mu_p_post, mu_m_post):
        m = mu_p_post - mu_m_post
        y = ((np.linalg.norm(mu_p_post) * 2 - np.linalg.norm(mu_m_post) * 2) / (2 * m[1])) - (m[0] / m[1]) * x
        return y

    mu_p_post = mmse_mu(x_p, mu_p)
    mu_m_post = mmse_mu(x_m, mu_m)

    plt.figure()
    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label='x +')
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label='x -')
    plt.scatter(mu_p[0], mu_p[1], 10, label='mu +')
    plt.scatter(mu_m[0], mu_m[1], 10, label='mu -')
    plt.scatter(mu_p_post[0], mu_p_post[1], 10, label='mu|D +')
    plt.scatter(mu_m_post[0], mu_m_post[1], 10, label='mu|D -')
    plt.plot(x[:, 0], decision_boundary(mu_p_post, mu_m_post)[:, 0], color="black", label="MMSE - decision boundary")
    plt.title("MMSE of mu - decision boundary")
    plt.legend()
    plt.show()

    cov_posterior = (1 / (x_m.shape[0] * (1 / sig) + 1 / sig_0)) * np.eye(x_m.shape[1])

    plt.scatter(x_p[:, 0], x_p[:, 1], 10, label='x +')
    plt.scatter(x_m[:, 0], x_m[:, 1], 10, label='x -')
    plt.scatter(mu_p[0], mu_p[1], 10, label='mu +')
    plt.scatter(mu_m[0], mu_m[1], 10, label='mu -')
    plt.scatter(mu_p_post[0], mu_p_post[1], 10, label='mu|D +')
    plt.scatter(mu_m_post[0], mu_m_post[1], 10, label='mu|D -')
    plt.plot(x, decision_boundary(mu_p_post, mu_m_post), color="black", label="MMSE boundary")

    for i in range(10):
        s_mu_p_post = np.random.multivariate_normal(mu_p_post, cov_posterior)
        s_mu_m_post = np.random.multivariate_normal(mu_m_post, cov_posterior)
        plt.plot(x, decision_boundary(s_mu_p_post, s_mu_m_post), color="skyblue")

    plt.title("Decision boundaries samples from the posterior")
    plt.ylim([-2, 2])
    plt.show()

    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)

        gauss1 = Gaussian(beta, nu).fit(dogs)
        gauss2 = Gaussian(beta, nu).fit(frogs)

        # create function for fast predictions
        pred = lambda x: np.clip(gauss1.log_likelihood(x) - gauss2.log_likelihood(x), -25, 25)

        test_score[i] = accuracy(pred(test), labels_t)
        train_score[i] = accuracy(pred(train), labels)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.title(r'QDA with different values of $\nu$')
    plt.show()

    # ------------------------------------------------------ section 2.2
    # define question variables
    kern, sigma = RBF_kernel(.009), .1
    Ns = [250, 500, 1000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ns)), np.zeros(len(Ns))

    gp = None
    for i, N in enumerate(Ns):
        print(f'GP using {N} samples', end='', flush=True)

        # sample N from each class
        idx_dogs = np.random.choice(dogs.shape[0], N, replace=False)
        idx_frogs = np.random.choice(frogs.shape[0], N, replace=False) + dogs.shape[0]
        idx_train = np.concatenate([idx_dogs, idx_frogs])
        X, y = train[idx_train], labels[idx_train]

        # fit a GP regression model to the data
        gp = GaussianProcess(kern, sigma).fit(X, y)

        train_score[i] = accuracy(gp.predict(train), labels)
        test_score[i] = accuracy(gp.predict(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ns, train_score, lw=2, label='train')
    plt.plot(Ns, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.title('GP Regression')
    plt.show()

    # calculate how certain the model is about the predictions
    d = np.abs(gp.predict(dogs_t) / gp.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()
