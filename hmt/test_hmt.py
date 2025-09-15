import numpy as np
import numpy.random as random
import hmt.hidden_markov_tree as hmt
from sys import argv


def split(tree, node, P, S_states, mu, sigma, level, max_level, nodetype):
    curr_distr = P[node.s]
    s0 = random.choice(S_states, 1, p=curr_distr)[0]
    s1 = random.choice(S_states, 1, p=curr_distr)[0]

    obs0 = random.multivariate_normal(mu[s0], sigma[s0], 1).flatten()
    obs1 = random.multivariate_normal(mu[s1], sigma[s1], 1).flatten()

    d0 = nodetype(node.id * 2, obs0, s=s0)
    d1 = nodetype(node.id * 2 + 1, obs1, s=s1)

    tree.add_node(d0, node.id)
    tree.add_node(d1, node.id)

    if level < max_level:
        split(tree, d0, P, S_states, mu, sigma, level+1, max_level, nodetype)
        split(tree, d1, P, S_states, mu, sigma, level+1, max_level, nodetype)
    return d0, d1


def sample_tree(S_states, X_states, max_level, n_hidden, n_obs, params=None, nodetype=hmt.HMNode):
    curr_distr = params['init_s_distr'].copy()

    # Sample initial hidden state
    s = random.choice(S_states, 1, p=curr_distr)[0]

    if "emit_mat" in params.keys():
        obs = random.choice(X_states, 1, p=params['emit_mat'][s, :])[0]
    else:
        obs = random.multivariate_normal(params['mus'][s], params['sigmas'][s], 1).flatten()
    # Sample initial observed state
    

    # Initialise tree
    tree = hmt.HMTree(n_hidden, n_obs, False, root=nodetype(1, obs, s=s))
    
    # Add nodes
    split(
        tree, tree.root, params["P"], S_states,
        mu=params['mus'], sigma=params['sigmas'],
        level=1, max_level=max_level, nodetype=nodetype
        )
    
    return tree

if __name__ == '__main__':
    random.seed(1350)

    S_states = (0, 1)
    X_states = (0, 1, 2)

    n_hidden, n_obs = 2, 3
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
        'init_s_distr': np.array([0.5, 0.5]),
        'P': np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ]),
        'mus': mus,
        'sigmas': sigmas
    }

    max_level = 5
    # np.random.seed(42)
    tree = sample_tree(S_states, X_states, max_level, n_hidden, n_obs, params)
    # tree.set_model_params(**params)

    tree.init_params()
    tree.Estep()
    tree.Mstep()
    tree.Estep()
    tree.Mstep()
    # tree.show('x')

    # tree.s_distr()
    # tree.upward_pass()

    # tree.show('s')
    # acc = tree.predict_hidden_states()
    # tree.show('ml_s')
    # print(f'Accuracy = {acc}')

