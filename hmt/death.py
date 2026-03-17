import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.stats import multivariate_normal
from hmt.emissions import Emission, MVN, VBMVN
from hmt.exceptions import HMTError
from hmt.utils import rowwise_outer, digamma, dirichlet_KL
import warnings

class DeathEmission(Emission):
    def __init__(self, n_hidden, n_obs, p_death=None, distrs=[], hmmodel=None):
        """
        Args
        ----
        distrs: list of hmt.emissions.Emissions
            List of emission distributions for different death states
            0: no death
            1: death during first observation
            2: death during second observation
            etc.
        """
        super().__init__(n_hidden, n_obs, hmmodel)
        self.p_death = p_death.copy() if p_death is not None else None
        self._distrs = distrs
        for death_ind, distr in enumerate(self.distrs):
            distr.death_ind = death_ind
        self.check_params()
    
    @property
    def distrs(self):
        return self._distrs
    
    @distrs.setter
    def distrs(self, distrs):
        for distr in distrs:
            distr.hmmodel = self.hmmodel
            distr.n_hidden = self.n_hidden
        self._distrs = distrs
    
    def clear_params(self):
        for distr in self.distrs:
            distr.clear_params()
        self.p_death = None
    
    def get_params(self):
        return {
            'p_death': self.p_death.copy() if self.p_death is not None else None,
            'distr_params': {distr.death_ind: distr.get_params() for distr in self.distrs}
        }
    
    def set_params(self, p_death=None, distr_params=None, check_params=True):
        if p_death is not None:
            self.p_death = p_death
        if distr_params is not None:
            if self.distrs is None:
                self.init_death_distrs()
            for death_ind, distr_params in distr_params.items():
                distr = [distr for distr in self.distrs if distr.death_ind == death_ind]
                assert distr
                distr[0].set_params(**distr_params)
        if check_params:
            self.check_params()
    
    def permute(self, perm):
        for distr in self.distrs:
            distr.permute(perm)
        self.p_death = self.p_death[perm, :]
    
    def init_death_distrs(self, distr_type=MVN, death_inds=None):
        self.distrs = []
        if death_inds is None:
            death_inds = np.unique([leaf.d for leaf in self.hmmodel.leaves])
            death_inds.sort()

        for death_ind in death_inds:
            n_obs = self.n_obs if not death_ind else death_ind
            distr = distr_type(n_hidden=self.n_hidden, n_obs=n_obs, hmmodel=self.hmmodel)
            distr.death_ind = death_ind
            self.distrs.append(distr)

    def init_params(self, init_method='kmeans', distr_type=MVN):
        if self.p_death is None:
            self.p_death = np.zeros((self.n_hidden, self.n_obs + 1))
            self.p_death[:, 0] = len(self.hmmodel.where('d', lambda d: d == 0))
            for death_ind in range(1, self.n_obs + 1):
                self.p_death[:, death_ind] = len([node for node in self.hmmodel.leaves if node.d == death_ind])
            self.p_death = (self.p_death.T / self.p_death.sum(axis=1)).T
        if not self.distrs:
            self.init_death_distrs(distr_type=distr_type)
    
        for distr in self.distrs:
            distr.init_params(init_method)
        self.check_params()
    
    def check_params(self):
        if self.p_death is not None:
            assert self.p_death.shape == (self.n_hidden, self.n_obs + 1)
            assert np.allclose(self.p_death.sum(axis=1), 1)
        if self.distrs:
            # assert self.distrs[0].n_obs == self.n_obs
            # self.distrs[0].check_params()
            for distr in self.distrs:
                distr.check_params()

    def death_xi_sum(self):
        death_xis = np.zeros((self.n_hidden, self.n_obs + 1))
        death_xis[:, 0] = self.hmmodel.sum_where('xi', 'd', lambda d: d == 0)
        assert np.allclose(death_xis[:, 0], np.sum([
            node.xi for node in self.hmmodel.where('d', lambda d: d == 0)
        ], axis=0))
        for death_ind in range(1, self.n_obs + 1):
            death_xis[:, death_ind] = np.array([
                list(node.xi) for node in self.hmmodel.leaves if node.d == death_ind
            ]).sum(axis=0)
        return death_xis
    
    def update_params(self):
        death_xis = self.death_xi_sum()
        self.p_death = (death_xis.T / death_xis.sum(axis=1)).T
        assert np.allclose(self.p_death.sum(axis=1), 1)

        for distr in self.distrs:
            distr.update_params()
    
    def sample(self, hidden_state):
        death = np.random.choice(np.arange(self.n_obs + 1), p=self.p_death[hidden_state])
        if death:
            x = np.full(self.n_obs, np.nan)
            x[:death] = self.distrs[death].sample(hidden_state)
        else:
            x = self.distrs[death].sample(hidden_state)
        return death, x
    
    def sample_truncated(self, t, hidden_state):
        death = np.random.choice(np.arange(self.n_obs + 1), p=self.p_death[hidden_state])
        if death:
            x = np.full(self.n_obs, np.nan)
            x[:death] = self.distrs[death].sample_truncated(t, hidden_state)
        else:
            x = self.distrs[death].sample_truncated(t, hidden_state)
        return death, x
    
    def emission(self, node):
        """
        P( X, D | S ) = P( X | D, S ) P( D | S )
        """
        return self.distrs[node.d].emission(node) * self.p_death[:, node.d]

    
    def call_distr_func(self, func_name, *args, **kwargs):
        for distr in self.distrs:
            getattr(distr, func_name)(*args, **kwargs)
    
    def precalculate_matrices(self):
        for distr in self.distrs:
            distr.precalculate_matrices()

    

class VBDeathEmission(DeathEmission):
    def __init__(self, n_hidden, n_obs, hmmodel=None, p_death=None, distrs=[]):
        super().__init__(n_hidden=n_hidden, n_obs=n_obs, hmmodel=hmmodel, p_death=p_death, distrs=distrs)
        self.prior_death_weights = None
        self.death_weights = None
    
    def set_priors(self, prior_death_weights=None, distr_priors=None):
        """
        Args
        ----
        prior_death_weights: np.ndarray, shape (n_hidden, n_obs + 1)
            Prior weights for the death probabilities.
        distr_priors: list of dicts
            List of prior parameters for each emission distribution.
        """
        if prior_death_weights is not None:
            self.prior_death_weights = prior_death_weights
        for distr, priors in zip(self.distrs, distr_priors):
            distr.set_priors(**priors)
    
    def clear_priors(self):
        self.prior_death_weights = None
        for distr in self.distrs:
            distr.clear_priors()
    
    def default_priors(self):
        self.prior_death_weights = np.ones((self.n_hidden, self.n_obs + 1))
    
    def set_hyperparams(self, death_weights=None, distr_hyperparams=None):
        if death_weights is not None:
            self.death_weights = death_weights.copy()
        for distr, hyperparams in zip(self.distrs, distr_hyperparams):
            distr.set_hyperparams(**hyperparams)
    
    def clear_hyperparams(self):
        self.death_weights = None
        for distr in self.distrs:
            distr.clear_hyperparams()
    
    def init_hyperparams(self, init_method='kmeans', seed=0, distr_type=VBMVN, **init_kwargs):
        if not self.distrs:
            self.init_death_distrs(distr_type=distr_type)
        if self.death_weights is None:
            self.prior_death_weights = np.ones((self.n_hidden, len(self.distrs)))
            self.death_weights = np.ones((self.n_hidden, len(self.distrs)))
        for distr in self.distrs:
            distr.init_hyperparams(init_method, seed, **init_kwargs)
    
    def update_params(self):
        self.p_death = np.exp(
            digamma(self.death_weights).T - digamma(self.death_weights.sum(axis=1))
        ).T
        self.p_death = (self.p_death.T / self.p_death.sum(axis=1)).T
        for distr in self.distrs:
            distr.update_params()
    
    def update_hyperparams(self):
        self.death_weights = self.prior_death_weights + self.death_xi_sum()

        ### START HERE!
        for distr in self.distrs:
            distr.update_hyperparams()
    
    def get_hyperparams(self):
        return {
            'death_weights': self.death_weights,
            'distr_hyperparams': [distr.get_hyperparams() for distr in self.distrs]
        }
    
    def number_of_params(self):
        n_params = 0
        for r in range(self.n_hidden):
            death_prob = self.p_death[r]
            n_params += len(death_prob) - 1 - len(np.where(np.isclose(death_prob, 0))[0])
        for distr in self.distrs:
            distr_n_params = distr.number_of_params()
            n_degenerate = len(np.where(np.isclose(self.p_death[:, distr.death_ind], 0))[0])
            if n_degenerate:
                distr_n_params = distr_n_params // self.n_hidden * (self.n_hidden - n_degenerate)
            n_params += distr_n_params
        return n_params
        # no_survival_states = np.where(np.isclose(self.p_death[:, 0], 0))[0]
        # if not no_survival_states:
        #     return n_params
        #TODO remove number of params for each of the transition matrices for cells that dont transition
    
    # def death_permutation(self, n_iter):
    #     max_loglk = self.hmmodel.Estep()
    #     params = self.hmmodel.get_params()

    
    def prior_KL_div(self):
        q_KL = dirichlet_KL(self.death_weights, self.prior_death_weights)
        for distr in self.distrs:
            q_KL += distr.prior_KL_div().sum() # sum here as not all distrs have same n_hidden
        return q_KL