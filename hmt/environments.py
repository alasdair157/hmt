import numpy as np
import pandas as pd
from copy import deepcopy
from timeit import default_timer
from itertools import permutations
from collections import defaultdict
import warnings

from hmt.hidden_markov_tree import HMNode, HMForest
from hmt.vb_hmt import VBHMForest
from hmt.exceptions import HMTError
from hmt.utils import div0, rowwise_outer, normal_round, digamma, log_addition, log_subtraction
import hmt.emissions as emissions


def add_passthrough_methods(method_names):
    def decorator(cls):
        for name in method_names:
            def make_method(name):
                def method(self, *args, **kwargs):
                    getattr(super(cls, self), name)(*args, **kwargs)
                    if self.next_env is not None:
                        getattr(self.next_env, name)(*args, **kwargs)
                return method
            setattr(cls, name, make_method(name))
        return cls
    return decorator


def censor1d(node, L):
    node.c = np.nan
    if L <= 0:
        node.x = np.nan
        if node.d0 is not None:
            censor1d(node.d0, L)
        if node.d1 is not None:
            censor1d(node.d1, L)
        return
    
    time = np.exp(node.x)
    if L < time:
        node.c = np.log(L)
        node.x = np.nan
        L = 0
    L -= time

    if node.d0 is not None:
        censor1d(node.d0, L)
    if node.d1 is not None:
        censor1d(node.d1, L)

def censor(node, L):
    if node.hmmodel.n_obs == 1:
        return censor1d(node, L)
    # print(L, node.x)
    if node.c is None:
        node.c = np.full_like(node.x, np.nan)
    if L <= 0:
        # L <= 0 <=> past censoring point, so set everything to nan
        node.c = np.full_like(node.x, np.nan)
        node.x = node.c.copy()
        if node.d0 is not None:
            censor(node.d0, L)
        if node.d1 is not None:
            censor(node.d1, L)
        return
    
    # if L > np.exp(node.x)

    for i, time in enumerate(np.exp(node.x)):
        if L < time:
            node.c[i] = np.log(L) # censoring time
            # print(node.c[i])
            node.x[i:] = np.nan
            L = 0
            break
        L -= time
    # if np.isnan(L):
    #     print("Here", node.id, node.x)
    if node.d0 is not None:
        censor(node.d0, L)
    if node.d1 is not None:
        censor(node.d1, L)

@add_passthrough_methods(
    ['check_params', 'clean', 'clear_params', 'find_missing_pattern', 'update_P', 'init_params']
    ) # Automatically define functions that cycle through each environment and call the function of the same name
class MultiEnvHMModel(HMForest):
    def __init__(
            self,
            n_hidden,
            n_obs,
            next_env=None,
            prev_env=None,
            life_P=None,
            idx=None,
            *hmargs,
            **hmkwargs
            ):
        """
        Args
        ----
        envs: tuple / list
            Tuple or list of the different environments, each environment is represented by a HMForest
        """
        super().__init__(
            n_hidden=n_hidden,
            n_obs=n_obs,
            *hmargs,
            **hmkwargs
        )
        self.trans_leaves = []
        self.trans_roots = []
        self._next_env = next_env
        self._prev_env = prev_env
        self.life_P = life_P
        self.idx = idx
        self.it = 0

    @property
    def next_env(self):
        return self._next_env
    
    @next_env.setter
    def next_env(self, env):
        if env is None:
            if self.next_env is not None:
                self.next_env._prev_env = None
                for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                    leaf.next = None
                    root.prev = None
            self._next_env = None
            for leaf in self.trans_leaves:
                leaf.next = None
            self.trans_leaves = []
            self.life_P = None
            return
        self._next_env = env
        env._prev_env = self
        if env.trans_roots:
            for leaf, root in zip(self.trans_leaves, env.trans_roots):
                assert leaf.id == root.id
                leaf.next = root
                root.prev = leaf
    
    @property
    def prev_env(self):
        return self._prev_env

    @prev_env.setter
    def prev_env(self, env):
        if env is None:
            if self.prev_env is not None:
                self.prev_env._next_env = None
                for leaf, root in zip(self.prev_env.trans_leaves, self.trans_roots):
                    leaf.next = None
                    root.prev = None
            self._prev_env = None
            for root in self.trans_roots:
                root.prev = None
            self.trans_roots = []
            return
        self._prev_env = env
        env._next_env = self
        for leaf, root in zip(env.trans_leaves, self.trans_roots):
            assert leaf.id == root.id
            leaf.next = root
            root.prev = leaf

    def __repr__(self):
        return f"MultiEnvHMModel({self.idx})"

    def __str__(self):
        return f"MultiEnvHMModel({self.idx})"

    def get_params(self):
        return {
            **super().get_params(),
            'life_P': self.life_P.copy() if self.life_P is not None else None,
            'next_env_params': None if self.next_env is None else self.next_env.get_params()
        }

    def set_params(
            self,
            init_s_distr=None,
            P=None,
            life_P=None,
            next_env_params=None,
            **emission_distr_params
            ):
        super().set_params(init_s_distr=init_s_distr, P=P, **emission_distr_params)
        if self.next_env is not None:
            if life_P is not None:
                self.life_P = life_P.copy()
            if next_env_params is not None:
                self.next_env.set_params(**next_env_params)

    # def emission(self, node):
    #     """
    #     Emission probability for nodes that are between two environments
    #     P(X, T | S) = P(X | T, S)P(T | S)
    #     P(T | S) = sum_{S_prev} P(T | S_prev)P(S_prev | S)
    #     """
    #     return self.emission_distr.emission(node)
        # if node.prev is None: # T is None, so just have P(X|S) as normal
        #     return self.emission_distr.emission(node)
        # x_given_t_s = self.emission_distr.emission(node)
        # t_given_s_prev = self.prev_env.emission_distr.survival_pdf(node.t)
        # life_P_reversed = div0(self.prev_env.life_P, node.s_distr)
        # life_P_reversed = (life_P_reversed.T * node.prev.s_distr).T
        # t_given_s_curr = t_given_s_prev @ life_P_reversed
        # return x_given_t_s * t_given_s_curr
        
    def number_of_params(self):
        n_params = super().number_of_params()
        if self.prev_env is not None:
            n_params -= (self.n_hidden - 1) # remove the starting probabilities for all envs but the first
        if self.next_env is not None:
            n_params += self.next_env.number_of_params()
        return n_params            
    
    def zero_loglikelihood(self):
        self.loglikelihood = 0
        if self.next_env is not None:
            self.next_env.zero_loglikelihood()
    
    def calculate_s_distr(self):
        pass

    def upward_pass(self):
        """Method depends on model used."""
        pass

    def downward_pass(self):
        """Method depends on model used."""
        pass

    def Estep(self):
        # Handle Errors
        if self.prev_env is None and self.init_s_distr is None:
            raise HMTError('Model has no initial S distribution.')
        if self.P is None:
            raise HMTError('Model has no transition matrix P.')
        if self.emission_distr is None:
            raise HMTError('Model has no emission distribution.')
        
        self.calculate_s_distr()
        self.upward_pass()
        self.downward_pass()
        return self.loglikelihood
    
    def update_init_s_distr(self):
        if self.prev_env is not None:
            raise HMTError("update_init_s_distr called from environment after first")
        super().update_init_s_distr()
    
    def update_emission_params(self):
        self.emission_distr.update_params()
        if self.next_env is not None:
            self.next_env.update_emission_params()
    
    def Mstep(self):
        # Update root nodes hidden state distribution
        self.update_init_s_distr()
        if not np.isclose(np.sum(self.init_s_distr), 1):
            raise HMTError("Updating initial distribution has gone wrong.")

        # Update P
        self.update_P()
        
        # Update emission distribution parameters
        self.update_emission_params()

    def train(self):
        if self.prev_env is not None:
            raise HMTError("Training is called from the first environment")

    def cutoff(self, cutoff_time):
        for root in self.roots:
            censor(root, cutoff_time) # Cut off times
            root.remove_where(lambda node: np.isnan(node.x).all() and np.isnan(node.c).all()) # Remove nodes after cutoff times
        self.get_leaves() # Find leaves after removing nodes
        self.trans_leaves = [leaf for leaf in self.leaves if not np.isnan(leaf.c).all()]

    def sample_next_s(self, root):
        pass


    def sample(
        self,
        n_nodes,
        cutoff_time,
        init_trees=1,
        dropout=0.0,
        sample_all_envs=True,
        nodes_per_root=None,
        has_death=False
    ):
        """
        Args
        ----
        change_times: list, tuple, ndarray
            Iterable of change times starting with start of experiment, then the
        TODO add random start times <- important!
        """
        if self.prev_env is None: # Sample normally
            super().sample(n_trees=init_trees, n_nodes=n_nodes, dropout=dropout, has_death=has_death)
        else: # Sample with truncation
            if self.prev_env.trans_leaves is None:
                raise HMTError("You must first sample from the previous environment")
            self.trans_roots = deepcopy(self.prev_env.trans_leaves)
            if nodes_per_root is None:
                nodes_per_root = n_nodes // len(self.trans_roots) + 1
            for leaf, root in zip(self.prev_env.trans_leaves, self.trans_roots):
                leaf.next, root.prev = root, leaf
                root.mother = None
                root.s = self.sample_next_s(root)
                
                # root.x = self.emission_distr.sample(root.s)
                root.t = np.where(~np.isnan(root.c), root.c, -np.inf)
                root.c = np.full_like(root.c, np.nan)
                
                if has_death:
                    d, env2_x = self.emission_distr.sample_truncated(t=root.t, hidden_state=root.s)
                else:
                    env2_x = self.emission_distr.sample_truncated(t=root.t, hidden_state=root.s)
                if self.n_obs == 1:
                    env2_x = env2_x.item()
                    root.c = root.c.item()
                env2_x = log_subtraction(env2_x, root.t)
                if self.n_obs > 1:
                    env2_x[~np.isnan(root.x)] = np.nan
                root.x = env2_x
                root.hmmodel = self # Change node's hmmodel to be current environment
                if has_death:
                    root.d = d
                    if d:
                        self.leaves.append(root)
                    else:
                        root.sample(nodes_per_root, p=dropout, has_death=has_death)
                else:
                    root.sample(nodes_per_root, p=dropout, has_death=has_death)

            self.roots = self.trans_roots.copy() # Can add other roots here if need be
        
        if self.next_env is not None:
            self.cutoff(cutoff_time)
            # self.trans_leaves = self.leaves
            if sample_all_envs:
                self.next_env.sample(n_nodes, cutoff_time, dropout=dropout, sample_all_envs=sample_all_envs, nodes_per_root=nodes_per_root)
    
    # TODO implement add_nodes_from_pandas

    @classmethod
    def from_pandas(cls, df, n_hidden, n_obs, obs_cols, n_envs, change_time, obs_func=None):
        cols = df.columns.tolist()
        assert set(obs_cols).issubset(set(cols)), "obs_cols must be a subset of the dataframe columns"

        # Create final env, then build back to start env, and read into start 
        forest = cls(n_hidden, n_obs, idx = n_envs)
        for i in range(1, n_envs):
            forest.prev_env = cls(n_hidden, n_obs, idx=n_envs - i)
            forest = forest.prev_env
        
        for tree_id in df["Tree ID"].unique():
            tree_df = df[df["Tree ID"] == tree_id].reset_index(drop=True)
            ## NOTE assumes the root is the first row in the dataframe

            # Initialise root and read in tree
            root = HMNode.from_pandas_row(tree_df.iloc[0], obs_cols=obs_cols, hmmodel=forest, obs_func=obs_func)
            next_time = np.sum(np.where(~np.isnan(root.x), np.exp(root.x), 0)) + np.sum(np.where(~np.isnan(root.c), np.exp(root.c), 0))
            if np.sum(np.exp(np.where(np.isnan(root.x), root.c, root.x))) > change_time:
                root.split(start_time=0, change_time=change_time)
            root.from_pandas(df=tree_df, obs_cols=obs_cols, obs_func=obs_func, curr_time=next_time, change_time=change_time)


            forest.roots.append(root)
            forest.leaves += root.get_leaves()

        temp = forest
        while temp.next_env is not None:
            temp = temp.next_env
            for root in temp.roots:
                temp.leaves += root.get_leaves()
            
        return forest


class MultiEnvSwitchHMModel(MultiEnvHMModel):
    def __init__(
            self,
            n_hidden,
            n_obs,
            next_env=None,
            prev_env=None,
            life_P=None,
            idx=None,
            *hmargs,
            **hmkwargs
            ):
        super().__init__(
            n_hidden,
            n_obs,
            next_env,
            prev_env,
            life_P,
            idx,
            *hmargs,
            **hmkwargs
            )
    
    def permute(self, perm):
        super().permute(perm)
        if self.next_env is not None:
            self.life_P = self.life_P[perm, :]
        if self.prev_env is not None:
            self.prev_env.life_P = self.life_P[:, perm]

    def init_life_P(self):
        if self.next_env is None:
            raise HMTError("Cannot initialise environment transition matrix when next environment is None")
        self.life_P = np.full((self.n_hidden, self.next_env.n_hidden), 1 / self.next_env.n_hidden)
        assert np.allclose(self.life_P.sum(axis=1), 1)
    
    def init_params(self, init_method='kmeans'):
        super().init_params(init_method)
        if self.next_env is not None:
            self.init_life_P()
            self.next_env.init_params(init_method)
    
    def extract_ml_s(self):
        for root in self.roots:
            if self.sister_dep:
                root.sd_extract_ml_s(d_drec=True, n_drec=True)
            else:
                root.extract_ml_s(d_drec=True, n_drec=True)
    
    def Mstep(self):
        # Update next_env P
        if self.next_env is not None:
            self.update_life_P()
        super().Mstep()

    def update_life_P(self):
        xi_ij = sum(leaf.xi_n for leaf in self.trans_leaves)
        xi_i = sum(leaf.xi for leaf in self.trans_leaves)
        self.life_P = (xi_ij.T / xi_i).T
        assert self.life_P.shape == (self.n_hidden, self.next_env.n_hidden)
        assert np.allclose(self.life_P.sum(axis=1), 1)
    
    def calculate_s_distr(self):
        for root in self.roots:
            if self.prev_env is None:
                root.s_distr = self.init_s_distr * 1.0
            if self.sister_dep:
                root.calculate_sd_s_distr(d_drec=True, n_drec=True)
            else:
                root.calculate_s_distr(drec=True)
    
    def upward_pass(self):
        self.zero_loglikelihood()
        if self.sister_dep:
            for root in self.roots:
                root.sd_upward_pass(d_urec=True, n_urec=True)
        else:
            for root in self.roots:
                root.upward_pass(urec=True)
    
    def downward_pass(self):
        if self.sister_dep:
            for root in self.roots:
                root.sd_downward_pass(d_drec=True, n_drec=True)
        else:
            for root in self.roots:
                root.downward_pass(drec=True)
    
    def urec_next(self, node):
        """
        Only called when node.next is not None
        """
        # if hasattr(self.next_env.emission_distr, "distrs"):
        trunc_pdf = self.next_env.emission_distr.emission(node.next)
        node.xi_n = self.life_P * node.next.beta_c_u * trunc_pdf
            # self.next_env.emission_distr.distrs[node.d].trunc_pdf(
            #     x=log_addition(node.next.x, node.next.t), t=node.next.t
            # )
        # else:
        #     node.xi_n = self.life_P * node.next.beta_c_u * self.next_env.emission_distr.trunc_pdf(
        #         x=log_addition(node.next.x, node.next.t), t=node.next.t
        #         )
            
        # NOTE Really it's beta_{n(u), u}, but in the code this works the same as
        # beta_{c(u), u}, so we assign it to the same variable for simplicity
        node.beta_c_u = node.xi_n.sum(axis=1)
        # node.xi_n = (node.xi_n.T / node.beta_c_u).T
        node.xi_n = div0(node.xi_n.T, node.beta_c_u).T
        assert node.beta_c_u.shape == (self.n_hidden, )
    
    def drec_next(self, node):
        """
        Only called when node.next is not None
        """
        # node.xi_n = self.life_P * node.next.beta_c_u * self.next_env.emission_distr.trunc_pdf(
        #     x=log_addition(node.next.x, node.next.t), t=node.next.t
        #     )
        # NOTE same calculation in urec, so not repeating code
        node.xi_n = (node.xi_n.T * node.xi).T
        node.next.xi = node.xi_n.sum(axis=0)
        assert node.next.xi.shape == (self.next_env.n_hidden, )
        assert np.isclose(node.next.xi.sum(), 1)
        assert np.allclose(node.xi_n.sum(axis=1), node.xi)
    
    def sd_viterbi_next(self, node):
        """
        Only called when self.next is not None
        """
        node.delta = self.emission(node)
        _delta = self.life_P * node.next.delta
        # print(_delta)
        assert _delta.shape == self.life_P.shape
        node.optimal_next_state = np.argmax(_delta, axis=1)
        assert node.optimal_next_state.shape == (self.n_hidden, )
        node.delta = (_delta[node.optimal_next_state].T * node.delta).T
        # print(np.log(node.delta))

    def sd_viterbi_extract_ml_s_next(self, node):
        ...


    
    def extract_next_ml_s(self, node):
        node.next.ml_s = np.argmax(node.xi_n[node.ml_s])

    def sample_next_s(self, root):
        return np.random.choice(np.arange(self.n_hidden), p=self.prev_env.life_P[root.s])



class VBMultiEnvHMModel(MultiEnvHMModel, VBHMForest):
    def __init__(self, n_hidden, n_obs, next_env=None, prev_env=None, life_P=None, idx=None, *hmargs, **hmkwargs):
        super().__init__(n_hidden, n_obs, next_env, prev_env, life_P, idx, *hmargs, **hmkwargs)


@add_passthrough_methods(
    ['clear_hyperparams']
)
class VBMultiEnvSwitchHMModel(VBMultiEnvHMModel, MultiEnvSwitchHMModel):
    def __init__(self, n_hidden, n_obs, next_env=None, prev_env=None, life_P=None, idx=None, *hmargs, **hmkwargs):
        super().__init__(n_hidden, n_obs, next_env, prev_env, life_P, idx, *hmargs, **hmkwargs)
        self.prior_life_P_weights = None
        self.life_P_weights = None
    
    def set_priors(self, life_P_weights=None, pi_weights=None, P_weights=None, emission_distr_priors=None):
        if life_P_weights is not None:
            assert self.next_env is not None
        self.prior_life_P_weights = life_P_weights
        super().set_priors(pi_weights, P_weights, emission_distr_priors)
    
    def clear_priors(self):
        self.prior_life_P_weights = None
        super().clear_priors()

    
    def clear_hyperparams(self):
        self.life_P_weights = None
        super().clear_hyperparams()
        if self.next_env is not None:
            self.next_env.clear_hyperparams()
    
    def get_hyperparams(self):
        return {
            **super().get_hyperparams(),
            "life_P_weights": self.life_P_weights.copy() if self.life_P_weights is not None else None,
            "next_env_hyperparams": None if self.next_env is None else self.next_env.get_hyperparams()
        }

    def get_params_and_hyperparams(self):
        return {
            **super().get_params_and_hyperparams(),
            'next_env_params': None if self.next_env is None else self.next_env.get_params(),
            'next_env_hyperparams': None if self.next_env is None else self.next_env.get_hyperparams()
        }
    
    def set_params_and_hyperparams(
            self,
            life_P_weights=None,
            life_P=None,
            pi_weights=None,
            init_s_distr=None,
            P_weights=None,
            P=None,
            next_env_params=None,
            next_env_hyperparams=None,
            **emission_params_and_hyperparams
    ):
        self.set_params(
            init_s_distr=init_s_distr, P=P, life_P=life_P, next_env_params=next_env_params
        )
        self.set_hyperparams(
            life_P_weights=life_P_weights, pi_weights=pi_weights, P_weights=P_weights,
            next_env_hyperparams=next_env_hyperparams
            )
       
        self.emission_distr.set_params_and_hyperparams(**emission_params_and_hyperparams)

    def set_hyperparams(
            self,
            life_P_weights=None,
            pi_weights=None,
            P_weights=None,
            next_env_hyperparams=None,
            **emission_hyperparams
            ):
        if life_P_weights is not None:
            self.life_P_weights = life_P_weights.copy()
        if next_env_hyperparams is not None and self.next_env is not None:
            self.next_env.set_hyperparams(**next_env_hyperparams)
        super().set_hyperparams(pi_weights, P_weights, **emission_hyperparams)

    def init_hyperparams(self):
        """
        Emission_hyperparams is no longer kwarg, as can't set multiple environments hyperparams, do this manually
        """
        super().init_hyperparams()
        if self.next_env is not None:
            if self.prior_life_P_weights is None:
                self.prior_life_P_weights = np.ones((self.n_hidden, self.next_env.n_hidden))
            if self.life_P_weights is None:
                self.life_P_weights = np.ones((self.n_hidden, self.next_env.n_hidden))
            self.next_env.init_hyperparams()
    
    def update_life_P_weights(self):
        self.life_P_weights = self.prior_life_P_weights + sum(leaf.xi_n for leaf in self.trans_leaves)
    
    def update_params(self):
        super().update_params()
        if self.next_env is not None:
            self.life_P = np.exp(
                    (digamma(self.life_P_weights).T - digamma(self.life_P_weights.sum(axis=1))).T
                )
            # Normalise life_P
            self.life_P = (self.life_P.T / self.life_P.sum(axis=1)).T
            assert np.allclose(self.life_P.sum(axis=1), 1)
            self.next_env.update_params()
        
    def Mstep(self):
        # Hyperparams
        self.update_P_weights()
        self.emission_distr.update_hyperparams()
        if self.prev_env is None:
            self.update_pi_weights()
        if self.next_env is not None:
            self.update_life_P_weights()

        # Params
        self.update_params()
        
        if self.next_env is not None:
            self.next_env.Mstep()
    
    def number_of_params(self):
        n_params = super().number_of_params()
        if self.next_env is not None:
            n_params += self.next_env.number_of_params()
        return n_params
    
    def prior_KL_div(self):
        q_KL = super().prior_KL_div()
        if self.next_env is not None:
            q_KL += self.next_env.prior_KL_div()
        return q_KL

'''
class MultiEnvNoSwitchHMModel(MultiEnvHMModel):
    def __init__(
            self,
            n_hidden,
            n_obs,
            next_env=None,
            prev_env=None,
            life_P=None,
            idx=None,
            *hmargs,
            **hmkwargs
            ):
        super().__init__(
            n_hidden,
            n_obs,
            next_env,
            prev_env,
            life_P,
            idx,
            *hmargs,
            **hmkwargs
            )

    def check_params(self):
        if self.next_env is not None and self.next_env.n_hidden != self.n_hidden:
            raise HMTError("No switch model needs the same number of hidden states")
        super().check_params
        if self.next_env is not None:
            self.next_env.check_params()

    def calculate_s_distr(self):
        for root in self.roots:
            if self.prev_env is None:
                root.s_distr = self.init_s_distr * 1.0
            if self.sister_dep:
                root.calculate_sd_s_distr(d_drec=True, n_drec=False)
            else:
                root.calculate_s_distr(drec=True)

        if self.next_env is not None:
            # Copy the relevant values over then continue downward recursion
            for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                root.s_distr = leaf.s_distr.copy()
                root.s_distr = leaf.s_distr @ self.life_P
            self.next_env.calculate_s_distr()

    def upward_pass(self):
        self.zero_loglikelihood()
        if self.next_env is not None:
            # Start upward pass from roots of last environment in chain 
            self.next_env.upward_pass()
            # Copy the relevant values over then continue upward recursion
            self.loglikelihood = self.next_env.loglikelihood
            if self.sister_dep:
                for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                    leaf.beta_c = root.beta_c.copy()
                    leaf.beta_c_u = root.beta_c_u.copy()
            else:
                for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                    leaf.beta = root.beta.copy()
                    leaf.m_d_beta = root.m_d_beta.copy()
        # Continue recursion from this environment
        if self.sister_dep:
            for root in self.roots:
                root.sd_upward_pass(d_urec=True, n_urec=False)
        else:
            for root in self.roots:
                root.upward_pass(urec=True)
    
    def downward_pass(self):
        if self.sister_dep:
            for root in self.roots:
                root.sd_downward_pass(d_drec=True, n_drec=False)
        else:
            for root in self.roots:
                root.downward_pass(drec=True)
    
        if self.next_env is not None:
            # Copy the relevant values over then continue downward recursion
            if self.sister_dep:
                for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                    root.xi = leaf.xi.copy()
                    # xi_c, xi_f are calculated at the start of the next environment as
                    # they use that environment's transition matrix rather than this one's
            else:
                for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                    root.xi = leaf.xi.copy()
                    root.m_d_xi = leaf.m_d_xi.copy() # Test this

            self.next_env.downward_pass()
    
    def extract_ml_s(self, mode='greedy'):
        """
        Extract the most likely hidden state of nodes in the forest

        Args
        ----
        mode: str
            'greedy' - uses the most likely state given the mother state
            'viterbi' - uses the most likely state given the entire tree
        """
        super().extract_ml_s()
        if self.next_env is not None:
            for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                root.ml_s = leaf.ml_s
            self.next_env.extract_ml_s(mode=mode)
    
    def viterbi(self):
        """
        Calculate P(S_u | X), the hidden state distribution of each node given all observed data.
        """
        if self.next_env is not None:
            self.next_env.viterbi()
            for leaf, root in zip(self.trans_leaves, self.next_env.trans_roots):
                leaf.delta = root.delta
                leaf.optimal_daughter_states = root.optimal_daughter_states
        super().viterbi()
    
    def sample_next_s(self, root):
        return root.s
'''