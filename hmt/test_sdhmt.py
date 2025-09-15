import numpy as np
import numpy.random as random
import hmt
from sys import argv
from timeit import default_timer as timer


def split(tree, node, P, S_states, mu, sigma, level, max_level):
    c_distr = P[node.s]
    c = np.unravel_index(
        random.choice(np.arange(len(S_states) ** 2), 1, p=c_distr.flatten())[0],
        c_distr.shape
        )
    # print(c_distr)
    # print(c)
    s0, s1 = S_states[c[0]], S_states[c[1]]

    obs0 = random.multivariate_normal(mu[s0], sigma[s0], 1).flatten()
    obs1 = random.multivariate_normal(mu[s1], sigma[s1], 1).flatten()

    d0 = hmt.SDHMNode(node.id * 2, obs0, s=s0)
    d1 = hmt.SDHMNode(node.id * 2 + 1, obs1, s=s1)

    tree.add_node(d0, node.id)
    tree.add_node(d1, node.id)

    if level < max_level:
        split(tree, d0, P, S_states, mu, sigma, level+1, max_level)
        split(tree, d1, P, S_states, mu, sigma, level+1, max_level)
    return d0, d1


def sample_sdtree(S_states, X_states, max_level, params=None):
    curr_distr = params['init_s_distr'].copy()

    # Sample initial hidden state
    s = random.choice(S_states, 1, p=curr_distr)[0]

    if "emit_mat" in params.keys():
        obs = random.choice(X_states, 1, p=params['emit_mat'][s, :])[0]
    else:
        obs = random.multivariate_normal(params['mus'][s], params['sigmas'][s], 1).flatten()
    # Sample initial observed state
    

    # Initialise tree
    tree = hmt.HMTree(len(S_states), len(X_states), root=hmt.SDHMNode(1, obs, s=s))
    
    # Add nodes
    split(
        tree, tree.root, params["P"], S_states,
        mu=params['mus'], sigma=params['sigmas'],
        level=1, max_level=max_level
        )
    
    return tree

if __name__ == '__main__':
    S_states = (0, 1, 2)
    X_states = (0, 1)

    n_hidden, n_obs = 3, 2

    np.random.seed(42)
    P = np.random.uniform(0, 1, (n_hidden, n_hidden, n_hidden))
    # Normalise for each value of i
    P = (P.T / np.sum(P, axis=(1, 2))).T

    # Randomly initialise r means of size n
    np.random.seed(42)
    mus = np.random.randn(n_hidden, n_obs)
    # print(mus)

    # Randomly initialise sigma
    np.random.seed(42)
    sigmas = abs(np.random.randn(n_hidden, n_obs, n_obs))
    # Ensure positive definite
    sigmas = sigmas @ np.transpose(sigmas, axes=(0, 2, 1))
    init_s_distr = np.array([0.3, 0.4, 0.3])
    params = {
        'init_s_distr': init_s_distr,
        'P' : P,
        # 'emit_mat': np.array([
        #     [0.9, 0.1],
        #     [0.1, 0.9]
        # ])
        'mus': mus,
        'sigmas': sigmas
    }
    max_level = 8
    
    np.random.seed(42)
    tree = sample_sdtree(S_states, X_states, max_level, params)
    
    # tree.show('x')
    # tree.set_model_params(**params)
    np.random.seed(0)
    tree.init_params()
    # print(tree.mus)
    # print(mus)
    for i in range(1000):
        print(f"Iteration {i}", end="\r")
        tree.EMstep()
    # print(f"Difference in nu    : {np.linalg.norm(init_s_distr - tree.init_s_distr)}")
    # print(f"Difference in P     : {np.linalg.norm(P - tree.P)}")
    # print(f"Difference in mu    : {np.linalg.norm(mus - tree.mus)}")
    # print(f"Difference in sigma : {np.linalg.norm(sigmas - tree.sigmas)}")
    # print()
    # acc = tree.predict_hidden_states()
    # print(f"Accuracy = {acc}")
