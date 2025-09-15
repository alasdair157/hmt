import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.tri as tri

from sklearn.cluster import KMeans
from copy import deepcopy

import hmt
from hmt.utils import stationary_distribution

# Dirichlet
from scipy.special import gamma, loggamma
from scipy.stats import beta
from operator import mul


def random_P(n_hidden, sis_dep=True, d_order=False):
    if sis_dep:
        P = np.random.uniform(0, 1, (n_hidden, n_hidden, n_hidden))

        if not d_order:
            # Ensure symmetric
            P = (P + np.transpose(P, axes=(0, 2, 1))) / 2

        # Normalise for each value of i
        P = (P.T / np.sum(P, axis=(1, 2))).T
    else:
        P = np.random.uniform(0, 1, (n_hidden, n_hidden))
        P = (P.T / P.sum(axis=1)).T
        assert np.allclose(P.sum(axis=1), 1)
    return P


def random_params(n_hidden, n_obs, sis_dep=True, seed=None, d_order=False, mean_stretch=None):
    np.random.seed(seed)
    P = random_P(n_hidden, sis_dep, d_order)

    # Randomly initialise r means of size n
    np.random.seed(seed)
    mus = np.random.randn(n_hidden, n_obs)
    if mean_stretch is not None:
        mus = (mus.T * mean_stretch).T

    # Randomly initialise sigma
    np.random.seed(seed)
    sigmas = np.random.randn(n_hidden, n_obs, n_obs)

    # Ensure positive definite
    sigmas = sigmas @ np.transpose(sigmas, axes=(0, 2, 1))

    # Randomly initialise initial distribution
    np.random.seed(seed)
    init_s_distr = np.random.uniform(0, 1, n_hidden)
    init_s_distr /= init_s_distr.sum()
    return {
        'init_s_distr': init_s_distr,
        'P' : P,
        'mus': mus,
        'sigmas': sigmas,
        'sister_dep': sis_dep,
        'daughter_order': d_order
    }


def Pmatrix(p, q):
    r = round((1 - p - q) / 2, 3)
    if abs(r) < 1.0e-8:
        r = 0
    # print(p, q, r, p + q + 2 * r, end = '\r')
    P = np.array([
        [
            [p, r],
            [r, q]
        ],
        [
            [q, r],
            [r, p]
        ]
    ])
    return P


# def mask(arr):
#     b = np.ones(arr.shape+(4,)) # «white» matrix with alpha=1
#     for i in range(arr[0].shape[0]):
#         for j in range(i, arr[0].shape[0]):
#             b[j, i, 3] = 0 
#     return b 


def get_params(p, q, mu2, n_obs=1):
    return {
        'P': Pmatrix(p, q),
        'init_s_distr': np.array([0.5, 0.5]),
        'mus': np.array([
                [0 for _ in range(n_obs)],
                [mu2 for _ in range(n_obs)]
            ]),
        'sigmas': np.stack([np.eye(n_obs) for _ in range(2)])
    }


def sample_train_test_forest(n_hidden, n_obs, params, train_trees, train_nodes, test_trees, test_nodes):
    train_forest, test_forest = hmt.HMForest(n_hidden, n_obs), hmt.HMForest(n_hidden, n_obs)
    
    train_forest.set_params(**params)
    train_forest.sample(train_trees, train_nodes, 0)
    train_forest.clear_params()

    test_forest.set_params(**params)
    test_forest.sample(test_trees, test_nodes, 0)
    test_forest.clear_params()

    return train_forest, test_forest


def sample_train_test_tree(n_hidden, n_obs, params, train_nodes, test_nodes):
    train_tree, test_tree = hmt.HMTree(n_hidden, n_obs), hmt.HMTree(n_hidden, n_obs)
    
    train_tree.set_params(**params)
    train_tree.sample(train_nodes, 0)
    train_tree.clear_params()

    test_tree.set_params(**params)
    test_tree.sample(test_nodes, 0)
    test_tree.clear_params()

    return train_tree, test_tree


def test_tree(tree, Nr):
    max_loglk = -np.inf
    for i in range(Nr):
        tree.train(1.0e-6, 200, overwrite_params=True)
        if tree.loglikelihood > max_loglk:
            max_loglk = tree.loglikelihood
            best_params = tree.get_params()
    return best_params


def test_train_acc(train_tree, test_tree, true_params=None, get_max=True):
    train_acc = train_tree.predict_hidden_states()

    trained_params = train_tree.get_params()

    if get_max:
        train_tree.set_params(**true_params)
        max_train_acc = train_tree.predict_hidden_states()

        train_tree.set_params(**trained_params)

        test_tree.set_params(**true_params)
        max_test_acc = test_tree.predict_hidden_states()

    test_tree.set_params(**trained_params)
    test_acc = test_tree.predict_hidden_states()

    if get_max:
        return train_acc, max_train_acc, test_acc, max_test_acc
    return train_acc, test_acc


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.ndim == 2:
        U, s, Vt = np.linalg.svd(covariance[:2, :2])
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        el = Ellipse(
            xy=position[:2], width=nsig * width, height=nsig * height, angle=angle, **kwargs
            )
        ax.add_patch(el)


def plot_ellipses(mus, sigmas, ax=None, xlim=None, ylim=None):
    ax = ax or plt.gca()
    # ax.axis('equal')
    
    for i, (pos, covar) in enumerate(zip(mus, sigmas)):
        draw_ellipse(pos, covar, alpha=0.2, color=f"C{i}")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax


def draw_ellipsoid(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    U, s, Vt = np.linalg.svd(covariance)
    rx, ry, rz = 2 * np.sqrt(s)
    x = rx * np.outer(np.cos(theta), np.sin(phi))
    y = ry * np.outer(np.sin(theta), np.sin(phi))
    z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    for nsig in range(1, 4):
        x_, y_, z_ = nsig * x, nsig * y, nsig * z
        for i in range(len(x)):
            for j in range(len(x)):
                [x_[i,j],y_[i,j],z_[i,j]] = np.dot([x_[i,j],y_[i,j],z_[i,j]], U.T) + position

        ax.plot_surface(x_, y_, z_, linewidth=0.1, rstride=4, cstride=4, **kwargs)


def plot_ellipsoids(mus, sigmas, ax=None, c=None):
    ax = ax or plt.axes(projection="3d")
    
    for i, (pos, covar) in enumerate(zip(mus, sigmas)):
        if c is None:
            col = f"C{i}"
        else:
            col = c[i]
        ax.scatter(*pos, marker='x', s=50, color=col)
        draw_ellipsoid(pos, covar, ax, alpha=0.1, color=col)

    return ax


def var_sum(cov):
    var = np.sum(np.diag(cov))
    for i in range(cov.shape[0]):
        for j in range(i):
            var += 2 * cov[i, j]
    return var


def draw_normal(ax, mu, var, nsds=3, fill=True, scale=1, **kwargs):
    sd = np.sqrt(var)
    x = np.linspace(mu - nsds * sd, mu + nsds * sd, 100)
    y = np.exp(-(x - mu) ** 2 / (2 * var)) / (sd * np.sqrt(2 * np.pi))
    y *= scale
    ax.plot(x, y, **kwargs)
    if fill:
        ax.fill_between(x, y, alpha=0.1)
    

def draw_lognormal(ax, mu, var, nsds=5, fill=True, scale=1, **kwargs):
    sd = np.sqrt(var)
    x = np.linspace(0.01, 60, 1000)
    y = np.exp(-(np.log(x) - mu) ** 2 / (2 * var)) / (x * sd * np.sqrt(2 * np.pi))
    y *= scale
    ax.plot(x, y, **kwargs)
    if fill:
        ax.fill_between(x, y, alpha=0.1)


def draw_logmixture(ax, means, variances, stat_distr, nsds=3, **kwargs):
    # sd = np.sqrt(variances)

    x = np.linspace(0.01, 60, 1000)

    y = np.zeros_like(x)
    for mu, var, pi in zip(means, variances, stat_distr.flatten()):
        sd = np.sqrt(var)
        post = np.exp(-(np.log(x) - mu) ** 2 / (2 * var)) / (x * sd * np.sqrt(2 * np.pi))
        y += pi * post
    ax.plot(x, y, **kwargs)


def draw_mixture(ax, means, variances, stat_distr, nsds=3, **kwargs):
    sd = np.sqrt(variances)

    x = np.linspace(np.min(means - nsds * sd), np.max(means + nsds * sd), 100)

    y = np.zeros_like(x)
    for mu, var, pi in zip(means, variances, stat_distr.flatten()):
        post = np.exp(-(x - mu) ** 2 / (2 * var)) / (np.sqrt(2 * np.pi * var))
        y += pi * post
    ax.plot(x, y, **kwargs)


def starting_point(n_hidden, X, n_init=10):
    # Find k-means clusters as starting point
    X_ = X[~np.isnan(X).any(axis=1)]
    kmeans = KMeans(n_hidden, n_init=n_init)
    kmeans.fit(X_)
    start = kmeans.cluster_centers_
    var = np.stack(
        [np.diag(np.mean((X_ - start[i]) ** 2, axis=0)) for i in range(n_hidden)]
        )
    order = start[:, 0].argsort()
    return start[order], var[order]


def test_data(forest, Nr, init_pos=None, init_var=None, max_n_hiddenestart=10, permute=True, logging=False):
    max_loglk = -np.inf
    for i in range(Nr):
        forest.clear_params()
        # Add noise to k means start
        noise = np.random.normal(0, 1, size=forest.n_hidden*forest.n_obs).reshape(forest.n_hidden, forest.n_obs)
        mus = init_pos + noise
        forest.set_params(mus=mus, sigmas=init_var)
        n_restart = 0
        while n_restart < max_n_hiddenestart:
            try:
                forest.init_params()
                forest.train(1.0e-5, 1000, overwrite_params=False, permute=permute)
            except hmt.HMTError:
                print(f"Restart {n_restart + 1}", end="\r")
                n_restart += 1
            else:
                if logging:
                    print(i, n_restart, round(forest.loglikelihood))
                break
        if forest.loglikelihood > max_loglk:
            max_loglk = forest.loglikelihood
            best_params = forest.get_params()
    return best_params, max_loglk


def read_data(n_hidden, n_obs, filepath, func=None):
    forest = hmt.HMForest(n_hidden, n_obs)
    forest.has_null = True
    if func is None:
        forest.read_txt(filepath, agg_func=lambda x: np.where(x==0, np.nan, x))
    else:
        forest.read_txt(filepath, agg_func=lambda x: func(np.where(x==0, np.nan, x)))
    clean_forest(forest)

    X_df = pd.read_csv(filepath, sep="\t", header=None)

    # Remove id and mother columns
    X_df = X_df.drop(columns=[0, 1])
    X = X_df.to_numpy()

    # Filter out zero rows
    X = X[~(X == 0).all(axis=1)]

    # Convert remaining zeros to NaN
    X = np.where(X == 0, np.nan, X)
    X = func(X)
    return forest, X



def beat_loglk(forest, max_runs, init_pos, init_var, loglk_to_beat):
    max_loglk = loglk_to_beat
    curr_loglk = -np.inf
    tries = 0

    while curr_loglk < max_loglk and tries < max_runs:
        forest.clear_params()
        forest.set_params(mus=init_pos, sigmas=init_var)
        forest.init_params()
        init_params = forest.get_params()
        forest.train(1.0e-5, 1000, overwrite_params=False)
        curr_loglk = forest.loglikelihood
        tries += 1
        print(f"{tries}: {round(curr_loglk, 3)}           ", end="\r")
    
    if tries == max_runs:
        print("Did not find it")
    return forest.get_params(), init_params, curr_loglk



def aic_bic(model, r_range, n_starts=5, **init_emission_params):
    """Returns AIC and BIC scores for """
    # Initialise values
    # test_model = deepcopy(model)
    aic_vals = np.zeros(len(r_range))
    bic_vals = np.zeros(len(r_range))
    loglk_vals = np.full((len(r_range), 200), np.nan)
    logn = np.log(len(model))
    params = []

    for i, r in enumerate(r_range):
        model.n_hidden = r
        model.clear_params()
        _, loglks = model.train(n_starts, overwrite_params=True, store_log_lks=True, **init_emission_params)
        k = model.number_of_params()
        print(f"{r = }, {loglks[-1] = }, {k = }")#, end="\r")
        aic_vals[i] = 2 * k - 2 * loglks[-1]
        bic_vals[i] = logn * k - 2 * loglks[-1]
        loglk_vals[i, :len(loglks)] = loglks
        params.append(model.get_params())
    return aic_vals, bic_vals, loglk_vals, params


def akaike_weights(xIC):
    """Weights for AIC or BIC scores"""
    exp_d_xIC = np.exp(-(xIC - np.min(xIC)) / 2)
    return exp_d_xIC / np.sum(exp_d_xIC)


def clean_forest(forest):
    forest.remove_where(lambda tree: len(tree) < 2)
    
    # Remove root nodes that are all null
    trees_to_add = []
    trees_to_remove = []
    for tree in forest.trees:
        if np.isnan(tree.root.x).all():
            # Split into two trees
            # if tree.root.d0 is not None:
            #     tree0 = hmt.HMTree(forest.n_hidden, forest.n_obs)
            #     tree0.root = tree.root.d0
            #     tree0.root.set_tree(tree0)
            #     tree0.root.mother = None
            #     trees_to_add.append(tree0)

            # if tree.root.d1 is not None:
            #     tree1 = hmt.HMTree(forest.n_hidden, forest.n_obs)
            #     tree1.root = tree.root.d1
            #     tree1.root.set_tree(tree1)
            #     tree1.root.mother = None
            #     trees_to_add.append(tree1)
            trees_to_remove.append(tree)

    forest.trees += trees_to_add
    for tree in trees_to_remove:
        forest.trees.remove(tree)

    
    for tree in forest.trees:
        tree.remove_where('x', lambda x: np.isnan(x).all())


    forest.remove_where(lambda tree: len(tree) < 2)
    forest.update_trees()


" ================== Variational Bayes ================== "


    # cov = np.diag(norm_weights) - np.outer(norm_weights, norm_weights)
    # cov /= (weight_sum + 1)

    # return cov


""" ===== CELL DATA PLOTS ===== """
def combine_G2_and_M(x):
    if x.ndim == 1:
        x[2] = x[2] + x[3]
        x = x[:3]
        return x
    x[:, 2] = x[:, 2] + x[:, 3]
    return x[:, :3]



def plot_distributions(forest):
    fig, axes = plt.subplots(1, forest.n_obs, figsize=(4 * forest.n_obs, 4))#, sharex=True, sharey=True)
    if forest.n_obs == 3:
        titles = {0: 'G1', 1: 'S', 2: 'G2 + M'}
    if forest.n_obs == 4:
        titles = {0: 'G1', 1: 'S', 2: 'G2', 3: 'M'}
    for obs, ax in enumerate(axes.flat):
        ax.set_title(titles[obs], fontsize=15)
        for i in range(forest.n_hidden):
            draw_normal(ax, forest.mus[i, obs], forest.sigmas[i, obs, obs], label=f"Hidden State {i + 1}")
    axes[0].set_ylabel("Density", fontsize=15)

    axes[-1].legend(fontsize=12)
    fig.text(0.5, -0.05, "Time", fontsize=15)
    fig.suptitle("Cell Cycle Times", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_mixture(forest, X):
    fig = plt.figure(figsize=(4 * forest.n_obs, 4), dpi=100)
    gs = fig.add_gridspec(2, forest.n_obs, height_ratios=[1, 9])
    leg_ax = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, i]) for i in range(forest.n_obs)]
    if forest.n_obs == 3:
        titles = {0: 'G1', 1: 'S', 2: 'G2 + M'}
    if forest.n_obs == 4:
        titles = {0: 'G1', 1: 'S', 2: 'G2', 3: 'M'}

    stat_distr = stationary_distribution(forest.P)
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=15)
        for r in range(forest.n_hidden):
            draw_normal(ax, forest.mus[r, i], forest.sigmas[r, i, i], fill=False, scale=stat_distr[r, 0], alpha=0.6, label=f"Hidden State {r + 1}")
        draw_mixture(ax, forest.mus[:, i], forest.sigmas[:, i, i], stat_distr, color='C3', label="Mixture Probability")
        ax.hist(X[:, i], density=True, color='C4', alpha=0.5, bins=20, label="Data")

    # Lengend
    leg_ax.axis('off')
    leg_ax.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=5, fontsize=12)

    # Titles
    fig.text(-0.02, 0.4, "Density", rotation=90, verticalalignment='center', fontsize=15)
    fig.text(1.02, 0.4, "Density", rotation=90, verticalalignment='center', fontsize=15, color='white')
    fig.text(0.5, -0.05, "Time", horizontalalignment='center', fontsize=15)
    fig.suptitle("Cell Cycle Times", fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_ABIC(r_range, AIC, BIC, dpi=None):
    plt.figure(figsize=(7, 4), dpi=dpi)
    plt.plot(r_range, AIC, label="AIC score", marker='o', linestyle=(0, (5, 7)))
    plt.plot(r_range, BIC, label="BIC score", marker='o', linestyle=(0, (5, 7)))
    plt.title("AIC and BIC Scores", fontsize=20)
    plt.xlabel("Number of Hidden States", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.xticks(r_range)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def plot_mixture_and_distr(forest, X, title="Cell Cycle Times", xlim=None):
    # fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 4))#, sharex=True, sharey=True)
    fig = plt.figure(figsize=(4 * forest.n_obs, 8), dpi=200)
    gs = fig.add_gridspec(3, forest.n_obs, height_ratios=[1, 9, 9])
    leg_ax = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, i]) for i in range(forest.n_obs)]
    mix_axes = [fig.add_subplot(gs[2, i]) for i in range(forest.n_obs)]
    if forest.n_obs == 3:
        titles = {0: 'G1', 1: 'S', 2: 'G2 + M'}
    if forest.n_obs == 4:
        titles = {0: 'G1', 1: 'S', 2: 'G2', 3: 'M'}

    stat_distr = stationary_distribution(forest.P)
    for i, ax in enumerate(mix_axes):
        for r in range(forest.n_hidden):
            draw_lognormal(ax, forest.mus[r, i], forest.sigmas[r, i, i], fill=False, scale=stat_distr[r, 0], alpha=0.6, label=f"Hidden State {r + 1}")
        draw_logmixture(ax, forest.mus[:, i], forest.sigmas[:, i, i], stat_distr, color='C3', label="Mixture Probability")
        ax.hist(X[:, i], density=True, color='C4', alpha=0.5, bins=35, label="Data")
        ax.set_xlabel("Time in " + titles[i], fontsize=15)
        if xlim is not None:
            ax.set_xlim(xlim[i])
    mix_axes[0].set_ylabel("Density", fontsize=15)


    for i, ax in enumerate(axes):
        # ax.set_title(titles[i], fontsize=15)
        if xlim is not None:
            ax.set_xlim(xlim[i])
        for r in range(forest.n_hidden):
            draw_normal(ax, forest.mus[r, i], forest.sigmas[r, i, i], fill=True, label=f"Hidden State {r + 1}")
    
    axes[0].set_ylabel("Density", fontsize=15)

    

    # Lengend
    leg_ax.axis('off')
    leg_ax.legend(*mix_axes[0].get_legend_handles_labels(), loc='center', ncol=5, fontsize=12)

    # Titles
    fig.text(-0.075, 0.65, "Component\nDistributions", ha='center', va='center', fontsize=15)
    fig.text(-0.075, 0.22, "Mixture\nDistribution", ha='center', va='center', fontsize=15)
    fig.text(1.02, 0.4, "Density", rotation=90, verticalalignment='center', fontsize=15, color='white')
    fig.text(0.5, -0.04, "Time / hrs", horizontalalignment='center', fontsize=15)
    fig.suptitle(title, fontsize=20)
    # axes[1].legend(ncol=5, fancybox=True)

    plt.tight_layout()
    plt.show()


def plot_normal_lognormal(forest, X, title="Cell Cycle Times", xlim=None):
    # fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 4))#, sharex=True, sharey=True)

    # exp_forest = deepcopy(forest)
    # exp_forest.apply(np.exp, 'x')
    exp_X = np.exp(X)

    fig = plt.figure(figsize=(4 * forest.n_obs, 8), dpi=200)
    gs = fig.add_gridspec(3, forest.n_obs, height_ratios=[1, 9, 9])
    leg_ax = fig.add_subplot(gs[0, :])
    log_axes = [fig.add_subplot(gs[1, i]) for i in range(forest.n_obs)]
    time_axes = [fig.add_subplot(gs[2, i]) for i in range(forest.n_obs)]

    if forest.n_obs == 3:
        titles = {0: 'G1', 1: 'S', 2: 'G2 + M'}
    if forest.n_obs == 4:
        titles = {0: 'G1', 1: 'S', 2: 'G2', 3: 'M'}

    if forest.P.ndim == 1:
        stat_distr = forest.P
    else:
        stat_distr = stationary_distribution(forest.P).T[0]
    for i, (log_ax, time_ax) in enumerate(zip(log_axes, time_axes)):
        for r in range(forest.n_hidden):
            draw_lognormal(time_ax, forest.mus[r, i], forest.sigmas[r, i, i], fill=False, scale=stat_distr[r], alpha=0.6, label=f"Hidden State {r + 1}")
            draw_normal(log_ax, forest.mus[r, i], forest.sigmas[r, i, i], fill=False, scale=stat_distr[r], alpha=0.6, label=f"Hidden State {r + 1}")
        
        draw_logmixture(time_ax, forest.mus[:, i], forest.sigmas[:, i, i], stat_distr, color='C3', label="Mixture Probability")
        draw_mixture(log_ax, forest.mus[:, i], forest.sigmas[:, i, i], stat_distr, color='C3', label="Mixture Probability")

        time_ax.hist(exp_X[:, i], density=True, color='C4', alpha=0.5, bins=35, label="Data")
        log_ax.hist(X[:, i], density=True, color='C4', alpha=0.5, bins=35, label="Data")

        time_ax.set_xlabel("Time in " + titles[i], fontsize=15)
        if xlim is not None:
            time_ax.set_xlim(xlim[i])

    time_axes[0].set_ylabel("Density", fontsize=15)
    log_axes[0].set_ylabel("Density", fontsize=15)

    # Lengend
    leg_ax.axis('off')
    leg_ax.legend(*time_axes[0].get_legend_handles_labels(), loc='center', ncol=5, fontsize=12)

    # Titles
    fig.text(-0.075, 0.65, "Normal in\nLog time", ha='center', va='center', fontsize=15)
    fig.text(-0.075, 0.22, "LogNormal\nin time", ha='center', va='center', fontsize=15)
    fig.text(1.02, 0.4, "Density", rotation=90, verticalalignment='center', fontsize=15, color='white')
    fig.text(0.5, -0.04, "Time / hrs", horizontalalignment='center', fontsize=15)
    fig.suptitle(title, fontsize=20)
    # axes[1].legend(ncol=5, fancybox=True)

    plt.tight_layout()
    plt.show()


# Dirichlet stuff
# We define corners here as they don't change between functions

CORNERS = np.array([
    [1 / 2, np.sqrt(3) / 2],
    [0, 0],
    [1, 0],
])
PAIRS = [CORNERS[np.roll(range(3), -i)[1:]] for i in range(3)]
AREA = 0.5 * 1 * 0.75**0.5

def tri_area(xy, pair):
    return 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    coords = np.array([tri_area(xy, p) for p in PAIRS]) / AREA
    return np.clip(coords, tol, 1.0 - tol)


def bc2xy(bc):
    '''Converts barycentric coordinates to 2D Cartesian.'''
    bc = np.asarray(bc)
    # return CORNERS.T @ bc
    return bc @ CORNERS
    # return (
    #     bc[1] + bc[2] / 2,
    #     np.sqrt(3) * bc[2] / 2
    # )
    # return 


class Dirichlet():
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha)
        self.logbeta = np.sum(np.log(gamma(self.alpha))) - loggamma(np.sum(self.alpha))

    def pdf(self, p):
        '''Returns pdf value for `p`.'''
        p = np.asarray(p)
        log_pdf = np.log(p) @ (self.alpha - 1) - self.logbeta
        return np.exp(log_pdf)

# class Dirichlet():
#     def __init__(self, alpha):
#         self._alpha = np.array(alpha)
#         self._coef = gamma(np.sum(self._alpha)) / \
#                            np.multiply.reduce([gamma(a) for a in self._alpha])
#     def pdf(self, x):
#         '''Returns pdf value for `x`.'''
#         return self._coef * np.multiply.reduce([xx ** (aa - 1)
#                                                for (xx, aa)in zip(x, self._alpha)])


def plot_dirichlet(alpha, ax=None, nlevels=200, subdiv=8, dpi=100, **kwargs):
    assert len(alpha) == 3
    dist = Dirichlet(alpha)
    
    triangle = tri.Triangulation(CORNERS[:, 0], CORNERS[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    if ax is None:
        fig = plt.figure(dpi=dpi)
        ax = plt.gca()

    ax.triplot(triangle, color='k', linewidth=2)
    tricontour = ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
    ax.axis('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.75**0.5 + 0.05)
    ax.axis('off')
    return tricontour


def add_ticks(ax, step=0.2, fontsize=12):
    tick_values = np.arange(0, 1.05, step)
    sqrt3_div_2 = np.sqrt(3) / 2
    one_over_sqrt3 = 1 / np.sqrt(3)

    l_off = 0.03
    t_off = 0.005

    # Bottom side (from (0,0) to (1,0))
    for t in tick_values:
        ax.text(t, -0.04, f'{t:.1f}', ha='center', va='center', size=fontsize)
        ax.plot((t, t), (-t_off, -0.02 * one_over_sqrt3), color='k', linewidth=0.5)

    # Right side (from (1,0) to (1/2, sqrt(3)/2))
    for t in tick_values:
        x = 1 - t / 2
        y = sqrt3_div_2 * t
        ax.text(
            x + l_off, y + one_over_sqrt3 * l_off, f'{t:.1f}', ha='center', va='center',
            rotation=-60, size=fontsize
            )
        ax.plot(
            (x + t_off, x + 0.01 + t_off), (y + 0.005, y + one_over_sqrt3 * 0.01 + t_off),
            color='k', linewidth=0.5
            )

    # Left side (from (1/2, sqrt(3)/2) to (0,0))
    for t in tick_values:
        x = t / 2
        y = sqrt3_div_2 * t
        ax.text(
            x - l_off, y + one_over_sqrt3 * l_off, f'{1-t:.1f}', ha='center', va='center',
            rotation=60, size=fontsize
            )
        ax.plot(
            (x - t_off, x - 0.01 - t_off), (y + t_off, y + one_over_sqrt3 * 0.01 + t_off),
            color='k', linewidth=0.5
            )


def add_labels(ax, labels=(r'$p_1$', r'$p_2$', r'$p_3$'), fontsize=12):
    offset = 0.1

    # Right side (from (1,0) to (1/2, sqrt(3)/2))
    x, y = bc2xy((0.5, 0, 0.5))
    ax.text(
        x + offset, y + offset / np.sqrt(3), labels[0], fontsize=fontsize, ha='center', va='center'
    )

    # Left side (from (1/2, sqrt(3)/2) to (0,0))
    x, y = bc2xy((0.5, 0.5, 0))
    ax.text(
        x - offset, y + offset / np.sqrt(3), labels[1], fontsize=fontsize, ha='center', va='center'
    )

    # Bottom side (from (0,0) to (1,0))
    x, y = bc2xy((0, 0.5, 0.5))
    ax.text(
        x, y - offset, labels[2], fontsize=fontsize, ha='center', va='center'
    )

def P2bc(P, i=0):
    if np.isclose(P[i, 0, 1], P[i, 1, 0]):
        return np.array((P[i, 0, 0], P[i, 1, 1], 2 * P[i, 0, 1]))
    return np.array((P[i, 0, 0], P[i, 1, 1], P[i, 0, 1], P[i, 1, 0]))


def P_weights2bc(P_weights, i=0):#
    if np.isclose(P_weights[i, 0, 1], P_weights[i, 1, 0]):
        return np.array((P_weights[i, 0, 0], P_weights[i, 1, 1], P_weights[i, 0, 1]))
    return np.array((P_weights[i, 0, 0], P_weights[i, 1, 1], P_weights[i, 0, 1], P_weights[i, 1, 0]))


def plot_model(model, true_P, i=0, vmax=None, show_mean=False, s=2, extra_points=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, ax in enumerate(axes.flatten()):
        alpha = P_weights2bc(model.P_weights, i=i)
        true_Pbc = P2bc(true_P, i=i)
        pred_P = P2bc(model.P, i=i)

        tricontour = plot_dirichlet(
            alpha=alpha, ax=ax, cmap="Blues", vmin=0, vmax=vmax
            )
        add_ticks(ax, fontsize=10)
        add_labels(ax, [r"$p$", r"$q$", r"$2r$"])
        ax.scatter(*bc2xy(true_Pbc), color='C1', s=s, label="True P")
        ax.scatter(*bc2xy(pred_P), color='C2', s=s, label="VB Prediction")
        if show_mean:
            mean = alpha / alpha.sum()
            ax.scatter(*bc2xy(mean), color='C3', s=s, label="Mean of Dirichlet")
        if extra_points is not None:
            ax.scatter(*bc2xy(extra_points[i]), color='C4', s=s, label="ML Prediction")
        ax.set_title(rf"$P_{i}$", fontsize=20)
    # plt.colorbar(tricontour)

    plt.legend()


def plot_dir_marginals(
        alpha,
        true_ps=None,
        vb_preds=None,
        ml_preds=None,
        labels=(r'$p_1$', r'$p_2$', r'$p_3$')):
    fig, axes = plt.subplots(len(alpha), 1, figsize=(9, 5), sharex=True, sharey=True)

    x = np.linspace(0, 1, 500)
    alpha0 = alpha.sum()

    for i, ax in enumerate(axes.flatten()):
        ax.plot(x, beta.pdf(x, a=alpha[i], b=alpha0-alpha[i]), label="VB Posterior")
        if vb_preds is not None:
            ax.axvline(vb_preds[i], 0, 1, color='C2', label="VB Prediction")
        if ml_preds is not None:
            ax.axvline(ml_preds[i], 0, 1, color='C3', label="ML Prediction")
        if true_ps is not None:
            ax.axvline(true_ps[i], 0, 1, color='C1', label="True Value", linestyle='--')

        ax.set_ylabel(labels[i])
    axes[0].legend()
    plt.xlim(0, 1)
    plt.tight_layout()
