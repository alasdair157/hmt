import numpy as np
from scipy.stats import wishart, invwishart


def generate_P(P_weights, daughter_order, size=1):
    if daughter_order:
        return np.random.dirichlet(alpha=P_weights, size=size)
    n_hidden = P_weights.shape[0]
    P = np.zeros((n_hidden, n_hidden, n_hidden), dtype=float)
    ## Else P matrices are symmetric
    for i in range(n_hidden):
        alpha = list(np.diag(P_weights[i])) + list(
            P_weights[i][np.triu_indices(n_hidden, 1)]
        )
        flat_P = np.random.dirichlet(alpha=alpha, size=1).squeeze()
        P[i] = np.diag(flat_P[:n_hidden])
        P[i][np.triu_indices(n_hidden, 1)] = flat_P[n_hidden:]
        P[i] = (P[i] + P[i].T) / 2
        assert np.isclose(P[i].sum(), 1)
    return P


def generate_params(model):
    pi = np.random.dirichlet(alpha=model.pi_weights, size=1).squeeze()
    P = generate_P(model.P_weights, model.daughter_order, size=1)

    # Sample precision and mean
    Ws = np.linalg.inv(model.emission_distr.W_invs)
    
    sigmainv = np.zeros_like(Ws)
    sigmas = np.zeros_like(model.emission_distr.W_invs)
    mus = np.zeros_like(model.emission_distr.ms)

    for i in range(model.n_hidden):
        sigmas[i] = invwishart.rvs(
            df=model.emission_distr.nus[i],
            scale=model.emission_distr.W_invs[i]
            )
        # sigmainv[i] = wishart.rvs(df=nu, scale=model.emission_distr.W_invs[i])

        # mu_var = np.linalg.inv(sigmainv[i]) / model.emission_distr.betas[i]
        mu_var = sigmas[i] / model.emission_distr.betas[i]
        mus[i] = np.random.multivariate_normal(model.emission_distr.ms[i], mu_var)

    return {
        'init_s_distr': pi,
        'P': P,
        'mus': mus,
        # 'sigmainv': sigmainv
        'sigmas': sigmas
    }

