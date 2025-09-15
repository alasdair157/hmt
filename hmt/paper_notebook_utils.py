import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from hmt.notebook_utils import plot_ellipses


def get_gamma_pdf(x, hmmodel, hidden_state, obs):
    alpha = hmmodel.emission_distr.alphas[hidden_state, obs]
    beta = hmmodel.emission_distr.betas[hidden_state, obs]
    return stats.gamma.pdf(x, alpha, scale=1/beta)

def get_normal_pdf(x, hmmodel, hidden_state, obs):
    mu = hmmodel.emission_distr.mus[hidden_state, obs]
    sigma = np.sqrt(hmmodel.emission_distr.sigmas[hidden_state, obs, obs])
    return stats.norm.pdf(x, mu, sigma)

def get_lognormal_pdf(x, hmmodel, hidden_state, obs):
    mu = hmmodel.emission_distr.mus[hidden_state, obs]
    sigma = np.sqrt(hmmodel.emission_distr.sigmas[hidden_state, obs, obs])
    return stats.lognorm.pdf(x, scale=np.exp(mu), s=sigma)


def plot_model_fit(df, hmmodel, n_hidden, n_obs, labels_dict=None, distr="lognormal", axes=None, **kwargs):
    if axes is None:
        _, axes = plt.subplots(1, n_obs, figsize=(15, 5), sharey=True)
    labels_dict = labels_dict or {0: "G1", 1: "S", 2: "G2 + M"}
    # stat_distr = hmmodel.P.sum(axis=(0, 2))
    # stat_distr /= stat_distr.sum()
    stat_distr = df['ml_s'].value_counts().sort_index().to_numpy().astype(float)
    
    if len(df['ml_s'].value_counts()) != n_hidden:
        stat_distr = np.append(stat_distr, np.array([0] * (n_hidden - len(df['ml_s'].value_counts()))))
    stat_distr /= stat_distr.sum()
    
    for j in range(n_obs):
        x = np.linspace(0, df[f"x{j}"].max() + 10, 100)
        for i in range(n_hidden):
            if distr == "gamma":
                y = get_gamma_pdf(x, hmmodel, i, j)
            elif distr == "normal":
                y = get_normal_pdf(x, hmmodel, i, j)
            elif distr == "lognormal":
                y = get_lognormal_pdf(x, hmmodel, i, j)
            else:
                raise NotImplementedError("Invalid distribution")
            # y *= stat_distr[i]
            axes[j].plot(x, y, color=f"C{i}")
        sns.histplot(
            df, x=f"x{j}", hue="ml_s", stat="density", edgecolor=None,
            common_norm=True, ax=axes[j], legend=False, palette="tab10")
        axes[j].set_title(labels_dict[j]); axes[j].set_xlabel(""); axes[j].set_ylabel("")
    axes[0].set_ylabel("Density")
    return axes


def lognorm_plot_2d(df, hmmodel, n_hidden, obs1, obs2, labels_dict=None, ax=None):
    n_levels = 5
    labels_dict = labels_dict or {0: "G1", 1: "S", 2: "G2 + M"}
    ax = ax or plt.gca()
    x = np.linspace(0.01, df[labels_dict[obs1]].max() + 10, 100)
    y = np.linspace(0.01, df[labels_dict[obs2]].max() + 10, 100)
    X, Y = np.meshgrid(x, y)
    logX, logY = np.log(X), np.log(Y)
    pos = np.dstack((logX, logY))

    for i in range(n_hidden):
        normal_pdf = stats.multivariate_normal.pdf(
            pos,
            mean=hmmodel.emission_distr.mus[i, (obs1, obs2)],
            cov=hmmodel.emission_distr.sigmas[i, (obs1, obs2)][:, (obs1, obs2)]
        )
        lognorm_pdf = normal_pdf / (X * Y)
        ax.contour(X, Y, lognorm_pdf, levels=n_levels, colors=[f"C{i}"]*n_levels)
        sns.scatterplot(
            x=labels_dict[obs1], y=labels_dict[obs2], 
            hue="ml_s", data=df, palette="tab10", ax=ax,
            alpha=0.2, legend=False
            )
        ax.set_xlabel(""); ax.set_ylabel("")



def model_pair_plot(df, hmmodel, n_hidden, n_obs, labels_dict=None, distr="lognormal", axes=None):
    if axes is None:
        _, axes = plt.subplots(n_obs, n_obs, figsize=(10, 10))
    labels_dict = labels_dict or {0: "G1", 1: "S", 2: "G2 + M"}

    plot_model_fit(df, hmmodel, n_hidden, n_obs, labels_dict=labels_dict, distr=distr, axes=np.diag(axes))
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            axes[i, j].axis("off")
            if distr == "lognormal":
                lognorm_plot_2d(df, hmmodel, n_hidden, i, j, labels_dict={0: "x0", 1: "x1", 2: "x2"}, ax=axes[j, i])
    
    for i in range(n_obs):
        axes[i, 0].set_ylabel(labels_dict[i])
        axes[-1, i].set_xlabel(labels_dict[i])
    
    for ax in axes.flatten():
        ax.set_title("")
    plt.tight_layout()
    return axes
    

