import numpy as np
import pandas as pd
import logging
import warnings
import hmt.emissions as emissions
from hmt.exceptions import HMTError
from hmt.hidden_markov_tree import HMModel, HMNode, HMTree, HMForest
from hmt.utils import digamma, rowwise_outer
from collections import defaultdict

logging.basicConfig(level=logging.ERROR, format="hmt %(levelname)s: %(message)s")

class VBHMModel(HMModel):
    def __init__(self, *hmm_args, **hmm_kwargs):
        super().__init__(*hmm_args, **hmm_kwargs)
        # Variational Bayesian parameters
        self.prior_pi_weights = None
        self.prior_P_weights = None

        self.pi_weights = None
        self.P_weights = None
    
    
    @property
    def n_hidden(self):
        """Unfortunately due to Python's issue with inheriting properties, we have to redefine this property"""
        return self._n_hidden
    
    @n_hidden.setter
    def n_hidden(self, value):
        # print("VB: Setting n_hidden")
        if not isinstance(value, int):
            raise HMTError("Number of hidden states must be an integer.")
        if value < 1:
            raise HMTError("Number of hidden states must be greater than 0.")
        # print(f"Setting n_hidden to {value}")
        self._n_hidden = value
        self.emission_distr.n_hidden = value
        self.clear_params()
        self.clear_hyperparams()
        self.clear_priors()
    

    @property
    def n_obs(self):
        return self._n_obs
    
    @n_obs.setter
    def n_obs(self, value):
        if not isinstance(value, int):
            raise HMTError("Number of observations must be an integer.")
        if value < 1:
            raise HMTError("Number of observations must be greater than 0.")
        # print(f"Setting n_obs to {value}")
        self._n_obs = value
        self.emission_distr.n_obs = value
        self.clear_params()
        self.clear_hyperparams()
        self.clear_priors()

    
    def set_emissions(self, emission_distr="MVN", emission_kwargs={}):
        if isinstance(emission_distr, str):
            if emission_distr == "MVN":
                self.emission_distr = emissions.VBMVN(self.n_hidden, self.n_obs, **emission_kwargs)
            elif emission_distr == "Gamma":
                self.emission_distr = emissions.VBMVGamma(self.n_hidden, self.n_obs, **emission_kwargs)
            else:
                raise NotImplementedError("Only MVN and Gamma distributions are implemented by default")
        else:
            self.emission_distr = emission_distr
            if emission_kwargs:
                self.emission_distr.set_params(**emission_kwargs)
        
        if self.emission_distr.type != "VB":
            raise HMTError("For variational Bayesian (VB) model, emission distribution must also be VB")
    

    def set_priors(self, pi_weights=None, P_weights=None, emission_distr_priors=None):
        self.prior_pi_weights = pi_weights
        self.prior_P_weights = P_weights
        self.emission_distr.set_priors(**emission_distr_priors)
    

    def clear_priors(self):
        self.prior_pi_weights = None
        self.prior_P_weights = None
        self.emission_distr.clear_priors()
    
    def init_hyperparams(self, **emission_hyperparams):
        """
        Initialise priors to least informative and initialise random initial value for each hyperparameter
        """
        # Initial S distribution
        if self.prior_pi_weights is None:
            self.prior_pi_weights = np.ones(self.n_hidden)
        if self.pi_weights is None:
            self.pi_weights = np.abs(np.random.randn(self.n_hidden))

        if self.sister_dep:
            # P tensor
            if self.prior_P_weights is None:
                self.prior_P_weights = np.ones((self.n_hidden, self.n_hidden, self.n_hidden))
            if self.P_weights is None:
                self.P_weights = np.abs(np.random.randn(self.n_hidden, self.n_hidden, self.n_hidden))

            if not self.daughter_order:
                self.P_weights = (self.P_weights + np.transpose(self.P_weights, axes=(0, 2, 1))) / 2
        else:
            # P matrix
            if self.prior_P_weights is None:
                self.prior_P_weights = np.ones((self.n_hidden, self.n_hidden))
            if self.P_weights is None:
                self.P_weights = np.abs(np.random.randn(self.n_hidden, self.n_hidden))

        self.emission_distr.init_hyperparams(self, **emission_hyperparams)
        

    def update_params(self):
        self.init_s_distr = np.exp(
            digamma(self.pi_weights) - digamma(np.sum(self.pi_weights))
        )
        # Normalise pi
        self.init_s_distr /= self.init_s_distr.sum()

        if self.sister_dep:
            self.P = np.exp(
                (digamma(self.P_weights).T - digamma(self.P_weights.sum(axis=(1, 2)))).T
            )
            if not self.daughter_order:
                # Halve off diagonal elements
                self.P /= 2
                diag_indices = np.arange(self.n_hidden)
                self.P[:, diag_indices, diag_indices] *= 2
            # Normalise P
            self.P = (self.P.T / self.P.sum(axis=(1, 2))).T
        else:
            self.P = np.exp(
                (digamma(self.P_weights).T - digamma(self.P_weights.sum(axis=1))).T
            )
            # Normalise P
            self.P = (self.P.T / self.P.sum(axis=1)).T
        
        self.emission_distr.update_params()


    def update_pi_weights(self):
        """Updates the pi weights, different for tree and forest"""
        pass

    
    def update_hyperparams(self):
        self.update_pi_weights()

        if self.sister_dep:
            self.P_weights = self.prior_P_weights + self.sum('xi_f')
            if not self.daughter_order:
                self.P_weights += np.transpose(self.P_weights, axes=(0, 2, 1))
                diag_indices = np.arange(self.n_hidden)
                self.P_weights[:, diag_indices, diag_indices] /= 2
        else:
            self.P_weights = self.prior_P_weights + self.sum('m_d_xi')
        
        self.emission_distr.update_hyperparams(self)
        

    def clear_hyperparams(self):
        self.pi_weights = None
        self.P_weights = None
        self.emission_distr.clear_hyperparams()
    

    def get_hyperparams(self):
        return {
            "pi_weights": self.pi_weights,
            "P_weights": self.P_weights,
            **self.emission_distr.get_hyperparams()
        }
    

    def set_hyperparams(self, pi_weights=None, P_weights=None, **emission_hyperparams):
        if pi_weights is not None:
            self.pi_weights = pi_weights
        if P_weights is not None:
            self.P_weights = P_weights
        self.emission_distr.set_hyperparams(**emission_hyperparams)
    

    def get_params_and_hyperparams(self):
        return {
            **self.get_params(),
            **self.get_hyperparams()
        }
    

    def set_params_and_hyperparams(
            self,
            init_s_distr=None,
            pi_weights=None,
            P=None,
            P_weights=None,
            sister_dep=None,
            daughter_order=None,
            check_params=True,
            **all_emission_params
            ):
        self.set_params(init_s_distr=init_s_distr, P=P, sister_dep=sister_dep, daughter_order=daughter_order, check_params=check_params)
        self.set_hyperparams(pi_weights=pi_weights, P_weights=P_weights, check_params=check_params)
        self.emission_distr.set_params_and_hyperparams(**all_emission_params)
    
    def Mstep(self):
        self.update_hyperparams()
        self.update_params()

    def train(
            self,
            n_starts=1,
            tol=1e-6,
            maxits=200,
            store_log_lks=False,
            permute=False,
            overwrite_params=True,
            surpress_warning=False,
            store_params=None,
            **init_emission_params
            ):

        if not overwrite_params:
            start_params = self.get_params() 
            # NOTE ensure only the parameters you want are stored in the model - all others should be None
        
        # Find patterns of missingness:
        if self.find_null():
            self.has_null = True
            self.emission_distr.missing_pattern = {idxs: defaultdict() for idxs in self.null_indices()}

        best_loglk = -np.inf
        best_it = 0
        best_params = {}
        n_failed = 0

        for _ in range(n_starts):
            try:
                ## Start of training loop
                self.it = 0
                self.clear_params()
                self.clear_hyperparams()

                self.init_hyperparams(**init_emission_params)
                
                self.update_params()
                # print(self.emission_distr.mus)
                # print(self.emission_distr.sigmas)
                
                if not overwrite_params:
                    self.set_params(**start_params)
                
                if store_params is not None:
                    if isinstance(store_params, str):
                        stored_params = [self.get_params_and_hyperparams()[store_params]]
                    elif isinstance(store_params, list):
                        stored_params = {key: [self.get_params_and_hyperparams()[key]] for key in store_params}

                curr_log_lk = self.Estep()
                if store_log_lks:
                    log_lks = np.zeros(maxits)
                    log_lks[0] = curr_log_lk


                for self.it in range(1, maxits):
                    # print(self.it)
                    self.update_hyperparams()
                    self.update_params()
                    if store_params is not None:
                        curr_params = self.get_params_and_hyperparams()
                        if isinstance(store_params, str):
                            stored_params.append(curr_params[store_params])
                        elif isinstance(store_params, list):
                            for key in store_params:
                                stored_params[key].append(curr_params[key])
                    prev_log_lk = curr_log_lk
                    curr_log_lk = self.Estep()
                    if store_log_lks:
                        log_lks[self.it] = curr_log_lk
                    if abs((prev_log_lk - curr_log_lk) / prev_log_lk) < tol:
                        break
                
                if curr_log_lk > best_loglk:
                    best_loglk = curr_log_lk
                    best_params = self.get_params_and_hyperparams()
                    best_it = self.it # Used to ensure best run actually converged
                    if store_log_lks:
                        best_loglks = log_lks[:self.it]
                    if store_params is not None:
                        best_stored_params = stored_params
            except Exception as e:
                # print(f"It: {self.it}")
                # Data under some initial parameters will be so unlikely that it leads to underflow in the
                # probabilities, so we just have to restart. If this becomes and issue you can fix some
                # of the initial parameters, especially the parameters of the emission distribution.
                n_failed += 1
                last_e = e
        
        if best_it == maxits - 1:
            warnings.warn("Loglikelihood did not converge, try changing the tol and maxits arguments")
        
        if n_failed > 0 and not surpress_warning:
            warnings.warn(f"{n_failed} / {n_starts} runs failed. This is likely due to initial parameters that do not match the data. If this is an issue, try changing init_emission_params")
            # print(f"The above warning was caused by {last_e}")
            raise last_e
            # logging.info(f"{n_failed} / {n_starts} runs failed. This is likely due to initial parameters that do not match the data. If this is an issue, try changing init_emission_params")
        self.set_params_and_hyperparams(**best_params)

        if permute:
            self.permute()

        if store_params is not None and store_log_lks:
            return best_it, best_loglks, best_stored_params
        if store_log_lks:
            return best_it, best_loglks
        if store_params is not None:
            return best_it, best_stored_params
        return best_it, best_loglk


class VBHMTree(VBHMModel, HMTree):
    """
    Using the model
    Q(theta) = Q(pi)Q(P)Q(theta_distr)
    pi ~ Dir(pi_weights)
    P ~ Dir(P_weights)
    """
    def __init__(self, *hmm_args, **hmm_kwargs):
        super().__init__(*hmm_args, **hmm_kwargs)
    

    def update_pi_weights(self):
        self.pi_weights = self.prior_pi_weights + self.root.xi
    

    @staticmethod
    def from_numpy(X, n_hidden, n_obs, has_true_s):
        tree = VBHMTree(n_hidden, n_obs)

        root_data = X[0]

        # Initialise root
        tree.root = HMNode(
            node_id=root_data[0],
            observed=root_data[2 : 2 + tree.n_obs].astype(float),
            tree=tree,
            s=root_data[-1] if has_true_s else None
        )
        # Add daughters recursively
        tree.root.from_numpy(X=X, has_true_s=has_true_s)

        # Find leaf nodes
        tree.leaves = tree.root.get_leaves()
        return tree


    @staticmethod
    def from_pandas(df, n_hidden, n_obs, obs_func=None):
        has_true_s = "s" in df.columns.tolist()
        ndarr = df.to_numpy(na_value=np.nan)
        return VBHMTree.from_numpy(ndarr, n_hidden, n_obs, has_true_s, obs_func=obs_func)


    @staticmethod
    def from_csv(path, n_hidden, n_obs, obs_func=None, **csv_kwargs):
        df = pd.read_csv(path, **csv_kwargs)
        return VBHMTree.from_pandas(df, n_hidden, n_obs, obs_func=obs_func)


class VBHMForest(VBHMModel, HMForest):
    def __init__(self, *hmm_args, **hmm_kwargs):
        super().__init__(*hmm_args, **hmm_kwargs)
    

    def update_pi_weights(self):
        self.pi_weights = self.prior_pi_weights + np.sum([root.xi for root in self.roots], axis=0)
    
    @staticmethod
    def from_numpy(X, n_hidden, n_obs, has_true_s, has_tree_id, obs_func=None):
        """
        Assumes the array is in the format:

        [root1, nan, obs, hidden (if known)]    (tree 1)
        [cell, mother, obs, hidden (if known)]
        [cell, mother, obs, hidden (if known)]
        ...
        [root2, nan, obs, hidden (if known)]    (tree 2)
        [cell, mother, obs, hidden (if known)]
        ...
        [root3, nan, obs, hidden (if known)]    (tree 3)
        [cell, mother, obs, hidden (if known)]
        ...
        NOTE: The tree ID is not necesarily required
        """
        print("VBForest from numpy")
        forest = VBHMForest(n_hidden, n_obs)

        if has_tree_id:
            print("Has tree ID")
            for tree_id in np.unique(X[:, 0]):
                tree_X = X[X[:, 0] == tree_id, 1:]
                root_data = tree_X[0]
                obs = root_data[2 : 2 + forest.n_obs].astype(float)
                if obs_func is not None:
                    obs = obs_func(obs)
                # Initialise root
                root = HMNode(
                    node_id=root_data[0],
                    observed=obs,
                    hmmodel=forest,
                    s=root_data[-1] if has_true_s else None
                )
                # Add daughters recursively
                root.from_numpy(X=tree_X, has_true_s=has_true_s, obs_func=obs_func)

                # Add tree root to forest
                forest.roots.append(root)
                forest.leaves += root.get_leaves()
            return forest
        
        # Without tree IDs given, check for nodes without mothers - these are the tree roots
        root_inds = list(np.where(pd.isna(X[:, 1]))[0])
        if not root_inds:
            # Check for nodes where cell_id == mother_id
            root_inds = list(np.where(X[:, 0] == X[:, 1])[0])

        for curr_ind, next_ind in zip(root_inds, root_inds[1:] + [None]):
            tree_X = X[curr_ind : next_ind]
            root_data = tree_X[0]
            obs = root_data[2 : 2 + forest.n_obs].astype(float)
            if obs_func is not None:
                obs = obs_func(obs)
            # Initialise root
            root = HMNode(
                node_id=root_data[0],
                observed=obs,
                hmmodel=forest,
                s=root_data[-1] if has_true_s else None
            )
            # Add daughters recursively
            root.from_numpy(X=X, has_true_s=has_true_s, obs_func=obs_func)

            # Add tree root to forest
            forest.roots.append(root)
            forest.leaves += root.get_leaves()
        
        return forest


    @staticmethod
    def from_pandas(df, n_hidden, n_obs, obs_func=None):
        has_true_s = 's' in df.columns.tolist()
        has_tree_id = 'Tree ID' in df.columns.tolist()
        return VBHMForest.from_numpy(df.to_numpy(na_value=np.nan), n_hidden, n_obs, has_true_s, has_tree_id, obs_func=obs_func)


    @staticmethod
    def from_csv(path, n_hidden, n_obs, obs_func=None, **csv_kwargs):
        df = pd.read_csv(path, **csv_kwargs)
        return VBHMForest.from_pandas(df, n_hidden, n_obs, obs_func=obs_func)