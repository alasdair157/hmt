"""Emission probabilities are organised here in a modular fashion"""

import numpy as np
from scipy.special import gamma, loggamma
from scipy.stats import multivariate_normal, norm, truncnorm
# from scipy.stats import multivariate_normal
from hmt.utils import rowwise_outer, div0, log_addition, log_subtraction
from scipy.special import digamma, polygamma
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from hmt.exceptions import HMTError
from hmt.minimax_tilting_sampler import TruncatedMVN
import warnings


class Emission:
    """
    Base class for emission distributions.
    """
    def __init__(self, n_hidden, n_obs, hmmodel=None):
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.hmmodel = hmmodel
        self.missing_pattern = {}
        self.death_ind = None
        self.degenerate_states = []
        self.generate_states = []
    
    def set_degenerate_state(self, degenerate_states):
        if not degenerate_states:
            return
        if isinstance(degenerate_states, int):
            degenerate_states = [degenerate_states]
        self.degenerate_states = degenerate_states
        self.generate_states = tuple(set(range(self.n_hidden)).difference(degenerate_states))
        self.n_hidden -= len(degenerate_states)
        self.update_emission()

    def update_emission(self):
        if not self.degenerate_states:
            return
        original_emission = self.emission
        def degenerate_emission(self, *args, **kwargs):
            
            p = np.zeros(self.hmmodel.n_hidden + len(self.degerate_states))
            
            p[self.generate_states] = original_emission(*args, **kwargs)
            return p
        self.emission = degenerate_emission

    def degenerate_state_prompt(self, degenerate_states):
        print(f"Degenerate state(s) detected: {degenerate_states}")
        self.set_degenerate_state(degenerate_states)
        return
        while True:
            x = input(
                f"Degenerate state detected, no nodes detected in state(s) {degenerate_states}, \
                    would you like to continue? (y/n)\n"
                )
            if x.lower() in ["y", "yes"]:
                self.set_degenerate_state(degenerate_states)
                return
            if x.lower() in ["n", "no"]:
                raise HMTError("Degenerate state detected")
            


    def init_params(self):
        raise NotImplementedError("init_params not implemented for emission distribution")


    def check_params(self):
        raise NotImplementedError("check_params not implemented for emission distribution")
    

    def set_params(self):
        raise NotImplementedError("set_params not implemented for emission distribution")
    

    def get_params(self):
        raise NotImplementedError("get_params not implemented for emission distribution")


    def clear_params(self):
        raise NotImplementedError("clear_params not implemented for emission distribution")


    def number_of_params(self):
        raise NotImplementedError("number_of_params not implemented for emission distribution")
    

    def permute(self, perm):
        raise NotImplementedError("permute not implemented for emission distribution")


    def pdf(self, x):
        raise NotImplementedError("pdf not implemented for emission distribution")


    def update_params(self, hmmodel):
        raise NotImplementedError("update_params not implemented for emission distribution")
    

    def sample(self, hidden_state):
        raise NotImplementedError("sample not implemented for emission distribution")


class MVN(Emission):
    def __init__(self, n_hidden, n_obs, mus=None, sigmas=None, hmmodel=None):
        super().__init__(n_hidden, n_obs, hmmodel)
        self.type = "ML"
        self.mus = mus
        self.sigmas = sigmas
        self.check_params()
        self.sigma_inv_det()
    
    def set_degenerate_state(self, degenerate_states):
        super().set_degenerate_state(degenerate_states)
        self.mus = self.mus[self.generate_states, :]
        self.sigmas = self.sigmas[self.generate_states, :, :]
        self.sigmainv = self.sigmainv[self.generate_states, :, :]
        self.detsigma = self.detsigma[self.generate_states, ...]
        self.check_params()
        

    def sigma_inv_det(self):
        if self.sigmas is not None:
            if self.n_obs > 1:
                self.sigmainv = np.linalg.inv(self.sigmas)
                self.detsigma = np.linalg.det(self.sigmas)
            else:
                self.sigmainv = 1 / self.sigmas
                self.detsigma = self.sigmas
        else:
            self.sigmainv = None
            self.detsigma = None

    def __str__(self):
        return "MVN (ML)"
    

    def kmeans_start(self):
        X = self.hmmodel.to_numpy(attrs='x')[:, 2:2+self.n_obs]
        # Impute any null values
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
        X = imp_mean.transform(X)

        # Run k-means
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=0).fit(X)
        self.mus = kmeans.cluster_centers_

        self.sigmas = np.zeros((self.n_hidden, self.n_obs, self.n_obs))
        for r in range(self.n_hidden):
            self.sigmas[r] = np.cov(X[kmeans.labels_ == r], rowvar=False)
        self.sigmas *= 5 # Scale up to avoid underflow errors
        try:
            self.sigmainv = np.linalg.inv(self.sigmas)
            self.detsigma = np.linalg.det(self.sigmas)
        except:
            print([len(X[kmeans.labels_ == r]) for r in range(self.n_hidden)])
            print(self.sigmas)
            print(self.death_ind)
    

    def random_start(self):
        if self.mus is None:
            # Randomly initialise r means of size n
            self.mus = np.random.randn(self.n_hidden, self.n_obs)
        
        if self.sigmas is None:
            if self.hmmodel is not None:
                var = self.hmmodel.var()
            else:
                var = np.ones(self.n_obs) # If no tree just use unit variance for each observation
            if not isinstance(var, np.ndarray):
                var = np.array([var])
            self.sigmas = np.stack([np.diag(var) for _ in range(self.n_hidden)])
            # Ensure positive definite
            # self.sigmas = self.sigmas @ np.transpose(self.sigmas, axes=(0, 2, 1))
            eigs = np.linalg.eigvals(self.sigmas) if self.n_obs > 1 else self.sigmas
            if (eigs <= 0).any():
                print(self.n_hidden, self.n_obs)
                print(self.sigmas)
                raise HMTError("Initial sigma should be positive definite")
            if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
                print(self.n_hidden, self.n_obs)
                print(self.sigmas)
                raise HMTError("Initial sigma should be symmetric")

            self.sigmainv = np.linalg.inv(self.sigmas)
            self.detsigma = np.linalg.det(self.sigmas)


    def init_params(self, init_method='kmeans'):
        # TODO initialisation with censored values too to avoid underflow
        if init_method == 'kmeans':
            self.kmeans_start()
        elif init_method == 'random':
            self.random_start()
        else:
            raise HMTError("Invalid initialisation method")
        if self.hmmodel.has_null and self.missing_pattern:
            self.precalculate_matrices()
    
        self.check_params()


    def check_params(self):
        if self.mus is not None:
            if self.mus.shape[0] != self.n_hidden:
                raise HMTError(f"mus must have shape n_hidden x n_obs = {(self.n_hidden, self.n_obs)} not {self.mus.shape}")
            
            # TODO check when nobs is 1D
            if self.n_obs > 1 and self.mus.shape[1] != self.n_obs:
                raise HMTError(f"mus must have shape n_hidden x n_obs = {(self.n_hidden, self.n_obs)} not {self.mus.shape}")
            

        if self.sigmas is not None:
            if self.sigmas.shape[0] != self.n_hidden:
                raise HMTError("Emission covaraince matrix must have shape rxnxn where r is the number of hidden states and n is the number of observed states")
            eigs = np.linalg.eigvals(self.sigmas) if self.n_obs > 1 else self.sigmas
            if not (eigs > 0).all():
                raise HMTError("Sigma should be positive definite")

            if self.n_obs > 1 and not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
                raise HMTError("Sigma should be symmetric")

    
    def set_params(self, mus=None, sigmas=None, sigmainv=None, detsigma=None, check_params=True):
        if mus is not None:
            self.mus = mus.copy()

        if sigmas is not None:
            self.sigmas = sigmas.copy()
        else:
            if sigmainv is not None:
                self.sigmas = np.linalg.inv(sigmainv)

        if sigmainv is None or detsigma is None:
            self.sigma_inv_det()
        else:
            self.sigmainv = sigmainv.copy()
            self.detsigma = detsigma.copy()
        
        if check_params:
            self.check_params()
    

    def get_params(self):
        return {
            'mus': self.mus,
            'sigmas': self.sigmas,
            'sigmainv': self.sigmainv,
            'detsigma': self.detsigma
        }
    

    def clear_params(self):
        self.mus = None
        self.sigmas = None
        self.sigmainv = None
        self.detsigma = None
    

    def number_of_params(self):
        # mus
        n_params = self.n_hidden * self.n_obs
        # sigmas
        n_params += self.n_hidden * self.n_obs * (self.n_obs + 1) // 2
        return n_params
    

    def permute(self, perm):
        """Permutes the hidden states (axis=0) of model parameters"""
        if self.mus is not None:
            self.mus = self.mus[perm, ...]
        if self.sigmas is not None:
            self.sigmas = self.sigmas[perm, ...]
        if self.sigmainv is not None:
            self.sigmainv = self.sigmainv[perm, ...]
        if self.detsigma is not None:
            self.detsigma = self.detsigma[perm, ...] 


    def pdf(self, x, mus=None, sigmas=None, sigmainv=None, detsigma=None, n_obs=None):
        """
        MVN.pdf(x)[i] = P(X = x | S = i)
        """
        mus = mus or self.mus
        sigmas = sigmas or self.sigmas
        sigmainv = sigmainv or self.sigmainv
        detsigma = detsigma or self.detsigma
        n_obs = n_obs or self.n_obs # changes if we have death

        if n_obs == 1:
            # print(norm.pdf(x, loc=mus.squeeze(), scale=np.sqrt(sigmas.squeeze())).shape)
            return norm.pdf(x, loc=mus.squeeze(), scale=np.sqrt(sigmas.squeeze()))

        if np.isnan(x).all():
            raise HMTError("Cannot calculate probability of null vector")

        if not np.isnan(x).any():
            centered_x = x - mus
            exponent = np.diag(np.diagonal(centered_x @ sigmainv @ centered_x.T))
            res = np.exp(-exponent / 2) / np.sqrt(detsigma * (2 * np.pi) ** n_obs)
            # true_p = np.array([
            #     multivariate_normal.pdf(x, mean=self.mus[0], cov=self.sigmas[0]),
            #     multivariate_normal.pdf(x, mean=self.mus[1], cov=self.sigmas[1])
            # ])
            # assert np.allclose(res, true_p)
            return res
    
        # If only a subset of x is null then we marginalise the probability
        non_null_indices = (~np.isnan(x)).nonzero()[0]
        marg_x = x[non_null_indices]
        marg_mus = mus[:, non_null_indices]
        marg_sigmas = sigmas[:, non_null_indices][:, :, non_null_indices]
        try:
            marg_sigmainv = np.linalg.inv(marg_sigmas)
        except:
            raise HMTError("x contains null values and marginalised sigma matrix is singular.")
        marg_detsigma = np.linalg.det(marg_sigmas)

        centered_x = marg_x - marg_mus
        exponent = np.diag(np.diagonal(centered_x @ marg_sigmainv @ centered_x.T))
        return np.exp(-exponent / 2) / np.sqrt(marg_detsigma * (2 * np.pi) ** non_null_indices.shape[0])

    def trunc_pdf(self, x, t):
        """
        f(X|X>T)
        """
        raise ValueError("Should not be calling")
        # obs_mask = np.where(~np.isnan(x))[0]
        # if len(obs_mask) == 1:
        #     obs_mask = obs_mask.item()
        # # print(obs_mask)
        # trunc_ind = np.where(t > -np.inf)[0].item()
        # trunc_val = t[trunc_ind]
        # # print(trunc_val, lognorm_addition[trunc_ind])
        # if x[trunc_ind] < trunc_val:
        #     raise HMTError("Observed value less than truncation bound")
        # assert not np.all(np.isnan(x[obs_mask]))
        # logpdf = np.zeros(self.n_hidden)
        # for r in range(self.n_hidden):
        #     logpdf[r] = multivariate_normal.logpdf(
        #         x=x[obs_mask],
        #         mean=self.mus[r, obs_mask],
        #         cov=self.sigmas[r, obs_mask, obs_mask]
        #     )
        # logsurvival = norm.logsf(
        #     x=trunc_val,
        #     loc=self.mus[:, trunc_ind],
        #     scale=np.sqrt(self.sigmas[:, trunc_ind, trunc_ind])
        # ).squeeze()
        # return np.exp(logpdf - logsurvival)
    
    def survival_pdf(self, t):
        raise NotImplementedError
        # trunc_ind = np.where(t > -np.inf)[0].item()
        # trunc_val = t[trunc_ind]
        # return norm.sf(
        #     x=trunc_val,
        #     loc=self.mus[:, trunc_ind],
        #     scale=self.sigmas[:, trunc_ind, trunc_ind]
        # ) / self.mus[:, trunc_ind]

    def emission(self, node):
        """
        P(X_o = x_o, X_c > c, X_m) = P(X_c > c | X_o = x_o) * f(X_o = x_o)
                                   = int_c^inf f(x) dx * f(x_o)
        P(X_o, X_c, X_t) = P(X)
        """
        x = node.x if (node.d is None or not node.d) else node.x[:node.d]
        c = node.c if (node.c is None or node.d is None or not node.d) else node.c[:node.d]
        t = node.t if (node.t is None or node.d is None or not node.d) else node.t[:node.d]

        if node.t is not None and not np.isnan(t).all():
            trunc_ind = np.where(t > -np.inf)[0]
            trunc_val = t[trunc_ind]
            if len(trunc_ind) > 1:
                raise HMTError("Multiple truncation inds not supported (mathematically this is no longer lognomal)")
            if np.isnan(x[trunc_ind]):
                if np.isnan(c[trunc_ind]):
                    raise HMTError(f"Null values can't be truncated: {x = }, {c = }, {t = }")
                c[trunc_ind] = log_addition(c[trunc_ind], trunc_val)
            else:
                x[trunc_ind] = log_addition(x[trunc_ind], trunc_val)
            denominator = norm.sf(
                x=trunc_val,
                loc=self.mus[:, trunc_ind],
                scale=np.sqrt(self.sigmas[:, trunc_ind, trunc_ind])
            )
        else:
            denominator = 1
            

        assert x.shape == (self.n_obs, )
        if c is None or np.isnan(c).all():
            if node.t is None or np.isnan(node.t).all():
                return self.pdf(x)
            return self.trunc_pdf(log_addition(x, node.t), node.t)
        
        if node.t is not None and not np.isnan(node.t).all():
            raise NotImplementedError("Observations with both censoring and truncation has not been implemented")

        nulls = np.argwhere(np.isnan(x)).flatten()
        s21_s11inv = self.missing_pattern[tuple(nulls)]['s21_s11^-1']
        cond_mus = self.mus[:, nulls] + np.einsum(
            'ijk,ik->ij', s21_s11inv, (x - self.mus)[:, ~np.isnan(x)]
            )
        cond_sigmas = self.missing_pattern[tuple(nulls)]['Es22']
        missing = c[np.isnan(x)]

        cond_cdf = np.ones(self.n_hidden)

        for r in range(self.n_hidden):
            cond_cdf[r] = multivariate_normal.cdf(
                lower_limit=np.where(np.isnan(missing), -np.inf, missing),
                x=np.full_like(missing, np.inf),
                mean=cond_mus[r],
                cov=cond_sigmas[r]
            )
        # print(cond_cdf)
        if np.isnan(x).all():
            return cond_cdf

        return cond_cdf * self.pdf(x)

    def precalculate_matrices(self):
        """
        For each pattern of missingness, calculates and stores the matrices required for imputing
        missing values
        """
        hidden_idxs = np.arange(self.n_hidden)
        for nulls in self.missing_pattern.keys():
            not_nulls = np.delete(np.arange(self.n_obs), nulls)
            s = np.linalg.inv(self.sigmas[np.ix_(hidden_idxs, not_nulls, not_nulls)])
            s = self.sigmas[np.ix_(hidden_idxs, nulls, not_nulls)] @ s
            self.missing_pattern[nulls]['s21_s11^-1'] = s
            s22 = self.sigmas[np.ix_(hidden_idxs, nulls, nulls)]
            s21_s11inv_s12 = s @ self.sigmas[np.ix_(hidden_idxs, not_nulls, nulls)]
            self.missing_pattern[nulls]['Es22'] = s22 - s21_s11inv_s12

    def impute(self, xi, x, c=None, t=None, d=None):
        """
        Impute missing values in a single observation x_u using the conditional normal distribution

        Returns E[x_u | observations] and E[S | observations] = E[x_u^Tx_u | observations] for each hidden state

        Args
        ----
        xi: np.ndarray
            The weights for each hidden state
        x: np.ndarray
            The observed values
        c: np.ndarray
            The censored values
        t: np.ndarray
            The truncation points
        """
        ################# TRUNCATION #################
        # Deal with this first as it is an additive constant that we can set to 0 if not used and move on
        
        x = x.copy()
        
        if t is not None and not np.isnan(t).all() and np.any(t > -np.inf):
            x = log_addition(x, t)
        
        x = x[:d] if d is not None and d else x
        c = c[:d] if c is not None and d is not None and d else c
        t = t[:d] if t is not None and d is not None and d else t

        if not np.isnan(x).any(): # All values present
            s = rowwise_outer(x - self.mus, x - self.mus)
                
            x = np.tile(x, (self.n_hidden, 1))
            Ex = x 
            Es = s
            return (Ex.T * xi).T, (Es.T * xi).T
        
        ################# MISSING #################
        # If there is no censoring and just missing values
        #TODO figure out how this works with censoring, looks like this is <y_m|y_o>, not <y_m|y_o, y_c>

        nulls = np.argwhere(np.isnan(x)).flatten()
        Ex = np.tile(x, (self.n_hidden, 1))

        # Calculate conditional mean and variance given observed values x
        s21_s11inv = self.missing_pattern[tuple(nulls)]['s21_s11^-1']
        cond_mus = self.mus[:, nulls] + np.einsum(
            'ijk,ik->ij', s21_s11inv, (x - self.mus)[:, ~np.isnan(x)]
            )
        cond_sigmas = self.missing_pattern[tuple(nulls)]['Es22']
        Ex[:, np.isnan(x)] = cond_mus

        if c is None or np.isnan(c).all():
            Es = rowwise_outer(Ex - self.mus, Ex - self.mus)
            Es[np.ix_(np.arange(self.n_hidden), nulls, nulls)] += cond_sigmas
            return (Ex.T * xi).T, (Es.T * xi).T
            # if res[0].shape != (self.n_hidden, self.n_obs):
            #     raise HMTError("Error on 431")
            # if res[1].shape != (self.n_hidden, self.n_obs, self.n_obs):
            #     raise HMTError("Error on 433")
            # return res
        
        ################# CENSORING #################
        ## Impute with censoring values
        c0 = c[np.isnan(x)] - cond_mus
        censoring_inds = np.where(~np.isnan(c0[0]))[0]
        if len(censoring_inds) > 1:
            # Often we only have one censored value, so we can save time by avoiding these calculations
            raise NotImplementedError("Multiple censored values not implemented")

        # Find the normalising constant numerically
        # TODO does not have to be multivariate, can just find the survival of univariate normal at censoring ind
        cond_cdf = np.zeros(self.n_hidden)
        for r in range(self.n_hidden):
            cond_cdf[r] = multivariate_normal.cdf(
                lower_limit=np.where(np.isnan(c0[r]), -np.inf, c0[r]),
                x=np.full_like(c0[r], np.inf),
                mean=np.zeros_like(c0[r]),
                cov=cond_sigmas[r]
            )

        sigma_kk = np.diagonal(cond_sigmas, axis1=1, axis2=2) 
        sigma_kk = sigma_kk[:, censoring_inds]
        # Fka = (np.exp(-(c0[:, censoring_inds] / sigma_kk) ** 2 / 2) / (np.sqrt(2 * np.pi) * sigma_kk)).flatten()
        Fka = (np.exp(-(c0[:, censoring_inds]) ** 2 / (2 * sigma_kk)) / (np.sqrt(2 * np.pi) * sigma_kk)).flatten()
        cond_sigma_row = cond_sigmas[:, censoring_inds, :].squeeze()
        
        if np.isclose(cond_cdf, 0).any():
            if np.isclose(cond_cdf, 0).all():
                print(x)
                print(self.mus)
                print(self.sigmas)
                raise ValueError("All cdfs are 0")
            with warnings.catch_warnings(): # Don't report divide by 0 error. 
                # We ignore this as the contribution to the sum is 0 since the cdf is 0 so xi is 0
                warnings.filterwarnings('ignore')
                zero_inds = np.where(np.isclose(cond_cdf, 0))[0]
                Ex0 = (cond_sigma_row.T * Fka / cond_cdf).T
                Ex0[zero_inds] = 0
        else:
            Ex0 = (cond_sigma_row.T * Fka / cond_cdf).T
        if Ex0.ndim == 1:
            Ex0 = Ex0.reshape(-1, 1)
            cond_sigma_row = cond_sigma_row.reshape(-1, 1)

        Ex[:, np.isnan(x)] += Ex0

        Ex0x0T = rowwise_outer(cond_sigma_row, cond_sigma_row)
        if np.isclose(cond_cdf, 0).any():
            with warnings.catch_warnings(): # Don't report divide by 0 error. 
                # We ignore this as the contribution to the sum is 0 since the cdf is 0 so xi is 0
                warnings.filterwarnings('ignore')
                Ex0x0T = (Ex0x0T.T * c[~np.isnan(c)][0] * Fka / (sigma_kk.flatten() * cond_cdf)).T
                Ex0x0T[zero_inds] = 0
        else:
            # TODO double check this works!
            Ex0x0T = (Ex0x0T.T * c[~np.isnan(c)][0] * Fka / (sigma_kk.flatten() * cond_cdf)).T
            Ex0x0T += cond_sigmas
        
    
        R = Ex0x0T - rowwise_outer(Ex0, Ex0)
        Es = rowwise_outer(Ex - self.mus, Ex - self.mus)
        Es[np.ix_(np.arange(self.n_hidden), nulls, nulls)] += R
        
        return (Ex.T * xi).T, (Es.T * xi).T
    
    def calc_mk_Hk(self, t):
        if t is None or np.isnan(t).all():
            raise HMTError("Null t values")
        trunc_ind = np.where(t > -np.inf)[0]
        if len(trunc_ind) > 1:
            raise NotImplementedError("Multiple truncation values not implemented")
        trunc_ind = trunc_ind[0]
        trunc_val = t[trunc_ind]
        # Truncated values at start of env have P_2(T_1 + T_2) / S_2(T_1)

        pdf = norm.pdf(
            x=trunc_val - self.mus[:, trunc_ind],
            loc=0,
            scale=np.sqrt(self.sigmas[:, trunc_ind, trunc_ind])
            )
        survival = norm.sf(
            x=trunc_val - self.mus[:, trunc_ind],
            loc=0,
            scale=np.sqrt(self.sigmas[:, trunc_ind, trunc_ind])
        )
        mk = (self.sigmas[:, trunc_ind].squeeze().T * div0(pdf, survival)).T
        assert mk.shape == (self.n_hidden, self.n_obs), f"mk: {mk.shape}"

        Hk = rowwise_outer(mk, self.sigmas[:, trunc_ind])
        Hk = (Hk.T * (self.mus[:, trunc_ind] - trunc_val) / self.sigmas[:, trunc_ind, trunc_ind]).T
        assert np.allclose(Hk, np.transpose(Hk, axes=(0, 2, 1)))
        assert Hk.shape == (self.n_hidden, self.n_obs, self.n_obs), f"Hk: {Hk.shape}"
        return mk, Hk
    
    def truncation_correction(self, hmmodel):
        mk_sum = np.zeros((self.n_hidden, self.n_obs))
        Hk_sum = np.zeros((self.n_hidden, self.n_obs, self.n_obs))
        
        if hasattr(hmmodel, 'prev_env'):
            # Assumes all truncated nodes are at the beggining of environment
            trunc_nodes = hmmodel.trans_roots
        else:
            trunc_nodes = hmmodel.where('t', lambda t: t is not None)

        if self.death_ind is not None:
            trunc_nodes = [node for node in trunc_nodes if node.d == self.death_ind]

        for root in trunc_nodes:
            mk, Hk = self.calc_mk_Hk(root.t)
            mk_sum += (mk.T * root.xi).T
            Hk_sum += (Hk.T * root.xi).T
        return mk_sum, Hk_sum
        

    def update_params(self, hmmodel):
        """
        Args
        ----
        hmmodel: HMTree / HMForest / VBHMTree / VBHMForest
            The HMT model that is being trained
        """
        if hmmodel.has_null and not self.missing_pattern: # Comupute matrices if they're missing
            # Calculate and store inverse matrices needed for conditional normal
            self.precalculate_matrices()

        
        # print(hmmodel.idx, xi_sum)
        # print(xi_sum)

        if self.death_ind is None:
            xi_sum = hmmodel.sum('xi')
            mus_sum, sigmas_sum = hmmodel.multiple_sum(('xi', 'x', 'c', 't', 'd'), func=self.impute)
        else:
            xi_sum = hmmodel.sum_where('xi', 'd', lambda d: d == self.death_ind)
            mus_sum, sigmas_sum = hmmodel.multiple_sum_where(
                attrs=('xi', 'x', 'c', 't', 'd'),
                func=self.impute,
                cond_attr='d',
                cond=lambda d: d == self.death_ind
                )
            # print(sigmas_sum.squeeze())
            # print(xi_sum)
            # if self.death_ind:
            #     death_nodes = [node for node in hmmodel.leaves if node.d == self.death_ind]
            #     mus_sum_test = np.zeros_like(self.mus)
            #     for node in death_nodes:
            #         mus_sum_test += self.impute(node.xi, node.x, node.c, node.t, node.d)[0]
            #     assert np.allclose(mus_sum, mus_sum_test)

        if hasattr(hmmodel, "prev_env") and hmmodel.prev_env is not None:
            mk_sum, Hk_sum = self.truncation_correction(hmmodel)
            mus_sum += mk_sum
            sigmas_sum -= Hk_sum
        # if not np.allclose(mk_sum, 0):
        #     print(mk_sum[~np.isclose(mk_sum, 0)])
        
        # print(self.mus)
        # print(self.sigmas)
        # print(xi_sum)
        # print("\n")

        # print(self.mus)
        # print(self.sigmas, "\n")
        
        self.mus, self.sigmas = (mus_sum.T / xi_sum).T, (sigmas_sum.T / xi_sum).T
        # self.sigmas_test = self.sigmas.copy()
        
        # print("Param Update")
        # print(self.sigmas, "\n")

        if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
            raise HMTError("Sigma should be symmetric")

        eigs = np.linalg.eigvals(self.sigmas) if self.n_obs > 1 else self.sigmas
        # print("Sigmas updated", eigs)
        if (eigs < 0).any() or np.isclose(eigs, 0).any():
            print(hmmodel.it)
            print(eigs)
            print(sigmas_sum)
            print(self.sigmas)
            raise HMTError("Sigma should be positive definite")
        
        self.detsigma = np.linalg.det(self.sigmas)
        self.sigmainv = np.linalg.inv(self.sigmas)

        if hmmodel.has_null: # Compute matrices for use with next E step (prob + imputing) 
            # Calculate and store inverse matrices needed for conditional normal
            self.precalculate_matrices()

    def sample(self, hidden_state):
        if self.n_obs > 1:
            return np.random.multivariate_normal(self.mus[hidden_state], self.sigmas[hidden_state])
        return np.random.normal(self.mus[hidden_state], np.sqrt(self.sigmas[hidden_state]))

    def sample_truncated(self, t, hidden_state):
        return TruncatedMVN(
            mu=self.mus[hidden_state],
            cov=self.sigmas[hidden_state],
            lb=t,
            ub=np.ones_like(t) * np.inf
        ).sample(1).squeeze()


class VBMVN(MVN):
    def __init__(self, n_hidden, n_obs, mus=None, sigmas=None):
        super().__init__(n_hidden, n_obs, mus, sigmas)
        self.type = "VB"
        self.prior_ms = None
        self.prior_betas = None
        self.prior_nus = None
        self.prior_W_invs = None

        self.ms = None
        self.betas = None
        self.nus = None
        self.null_nus = None
        self.W_invs = None
    

    def __str__(self):
        return "MVN (VB)"
    

    def set_priors(self, ms=None, betas=None, nus=None):
        self.prior_ms = ms
        self.prior_betas = betas
        self.prior_nus = nus
        self.prior_W_invs = None
    

    def kmeans_start(self, hmmodel, seed=0):
        X = hmmodel.to_numpy()[:, 2 : 2 + self.n_obs]

        # Impute any null values
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
        X = imp_mean.transform(X)

        # Run k-means
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=seed).fit(X)
        self.ms = kmeans.cluster_centers_
    

    def random_start(self, seed, init_ms=None, init_var=1):
        """
        Initialise priors to least informative and initialise random initial value for each hyperparameter
        """
        # Multivariate normal means
        if self.prior_ms is None:
            self.prior_ms = np.zeros((self.n_hidden, self.n_obs))

        if self.ms is None:
            if init_ms is None:
                init_ms = np.array([[i] * self.n_obs for i in range(self.n_hidden)])
            if init_ms.shape != (self.n_hidden, self.n_obs):
                raise HMTError("init_ms must have shape (n_hidden, n_obs)")
            self.ms = init_ms

        # Multivariate normal covariance scaling
        if self.prior_betas is None:
            self.prior_betas = np.zeros(self.n_hidden)
        if self.betas is None:
            self.betas = np.ones(self.n_hidden)

        # Multivariate normal covariance degrees of freedom
        if self.prior_nus is None:
            self.prior_nus = np.full(self.n_hidden, fill_value=self.n_obs)
        if self.nus is None:
            self.nus = np.abs(np.random.randn(self.n_hidden))
        
        if self.prior_W_invs is None:
            self.prior_W_invs = np.stack([np.eye(self.n_obs) for _ in range(self.n_hidden)]) * self.n_obs
        
        if self.W_invs is None:
            self.W_invs = np.stack([np.eye(self.n_obs) for _ in range(self.n_hidden)]) * init_var
        # print(self.W_invs)
        # print(self.nus)


    def init_hyperparams(self, hmmodel=None, init_method='kmeans', seed=0, **init_kwargs):
        if init_method == 'kmeans':
            self.random_start(seed, **init_kwargs)
            # Overwrite mus with kmeans
            self.kmeans_start(hmmodel, seed)
        elif init_method == 'random':
            self.random_start(seed, **init_kwargs)
        else:
            raise HMTError("Invalid initialisation method")
        
        if hmmodel.has_null:
            self.update_params()
            self.precalculate_matrices()

    def update_params(self):
        """Overwrite ML update_params to have VB updates"""
        self.mus = self.ms
        self.sigmas = (self.W_invs.T / self.nus).T 
        # print(f"{self.sigmas.shape = }")

        self.sigmainv = np.linalg.inv(self.sigmas)
        self.detsigma = 1 / np.linalg.det(self.sigmainv)

    def mu_sum(self, xi, x, d=None):

        x = x[:d] if d is not None and d else x

        if not np.isnan(x).any():
            # print("Mu_sum:\n", np.outer(xi, x))
            return np.outer(xi, x) # Single x vector so use outer to get a value for each hidden state
        nulls = np.argwhere(np.isnan(x)).flatten()
        s21_s11inv = self.missing_pattern[tuple(nulls)]['s21_s11^-1']
        mu1_y1 = (x - self.mus)[:, ~np.isnan(x)]
        Ex = np.tile(x, (self.n_hidden, 1))
        Ex[:, np.isnan(x)] = self.mus[:, nulls] + np.einsum('ijk,ik->ij', s21_s11inv, mu1_y1)
        return (Ex.T * xi).T # Each hidden state has an E[x] vector associated to it, so don't use outer

    def cov_sum(self, xi, x, xbar, d=None):#, _Ex):

        x = x[:d] if d is not None and d else x

        if not np.isnan(x).any():
            Es = rowwise_outer(x - xbar, x - xbar)
            return (Es.T * xi).T
        nulls = np.argwhere(np.isnan(x)).flatten()

        # Calculate E[x]
        s21_s11inv = self.missing_pattern[tuple(nulls)]['s21_s11^-1']
        mu1_y1 = (x - self.mus)[:, ~np.isnan(x)]
        Ex = np.tile(x, (self.n_hidden, 1))
        Ex[:, np.isnan(x)] = self.mus[:, nulls] + np.einsum('ijk,ik->ij', s21_s11inv, mu1_y1)
        assert Ex.shape == (self.n_hidden, self.n_obs)
        
        # print(_Ex)
        # print(_Ex.dtype, Ex.dtype)
        # np.putmask(_Ex, np.ones_like(_Ex), Ex)
        # print(_Ex)

        # Calculate E[S]
        Es = rowwise_outer(Ex - xbar, Ex - xbar)
        Es[np.ix_(np.arange(self.n_hidden), nulls, nulls)] += self.missing_pattern[tuple(nulls)]['Es22']
        return (Es.T * xi).T
    

    def update_hyperparams(self, vb_hmmodel):
        if vb_hmmodel.has_null:
            self.precalculate_matrices()
        
        if self.death_ind is None:
            # Auxilliary values for caluclating updates for mu and sigma hyperparams
            # Note any null value will add 0 to the total
            xi_sum = vb_hmmodel.sum(attrs='xi')

            # Nixbar = vb_hmmodel.sum(attrs=('xi', 'x'), func=np.outer)
            Nixbar = vb_hmmodel.sum(attrs=('xi', 'x'), func=self.mu_sum)

            xbar = (Nixbar.T / xi_sum).T
            
            NiSi = vb_hmmodel.sum(attrs=('xi', 'x'), func=lambda xi, x: self.cov_sum(xi, x, xbar))
        else:
            xi_sum = vb_hmmodel.sum_where('xi', 'd', lambda d: d == self.death_ind)
            if np.any(xi_sum == 0):
                print(f"{xi_sum = }")
                raise HMTError("Degenerate hidden state")
            Nixbar = vb_hmmodel.sum_where(
                attrs=('xi', 'x', 'd'),
                cond_attr='d',
                cond=lambda d: d == self.death_ind,
                func=self.mu_sum
                )
            xbar = (Nixbar.T / xi_sum).T
            NiSi = vb_hmmodel.sum_where(
                attrs=('xi', 'x', 'd'),
                cond_attr='d',
                cond=lambda d: d == self.death_ind,
                func=lambda xi, x, d: self.cov_sum(xi, x, xbar, d)
                )

        self.nus = self.prior_nus + xi_sum
        self.betas = self.prior_betas + xi_sum

        if self.prior_betas.any(): # If not save time by not multiplying matrices by 0
            self.ms = Nixbar + (self.prior_ms.T * self.prior_betas).T
            self.ms = (self.ms.T / (xi_sum + self.prior_betas)).T

            Winv_term = rowwise_outer(xbar - self.prior_ms, xbar - self.prior_ms)
            Winv_term = (Winv_term.T * self.prior_betas * xi_sum / (self.prior_betas + xi_sum)).T
        else: 
            self.ms = xbar
            Winv_term = 0


        self.W_invs = self.prior_W_invs + NiSi + Winv_term

    def clear_params(self):
        super().clear_params()
        self.clear_hyperparams()

    def clear_hyperparams(self):
        self.ms = None
        self.betas = None
        self.nus = None
        self.null_nus = None
        self.W_invs = None

    def get_hyperparams(self):
        return {
            'ms': self.ms,
            'betas': self.betas,
            'nus': self.nus,
            'W_invs': self.W_invs
        }

    def set_hyperparams(self, ms=None, betas=None, nus=None, W_invs=None, check_params=True):
        if ms is not None:
            self.ms = ms
        if betas is not None:
            self.betas = betas
        if nus is not None:
            self.nus = nus
        if W_invs is not None:
            self.W_invs = W_invs
        if check_params:
            self.check_hyperparams()
    
    def set_params_and_hyperparams(
            self,
            mus=None,
            sigmas=None,
            sigmainv=None,
            detsigma=None,
            ms=None,
            betas=None,
            nus=None,
            W_invs=None,
            check_params=True
            ):
        self.set_params(mus, sigmas, sigmainv, detsigma, check_params=check_params)
        self.set_hyperparams(ms, betas, nus, W_invs, check_params=check_params)

    def permute(self, perm):
        if self.ms is not None:
            self.ms = self.ms[perm, ...]
        if self.betas is not None:
            self.betas = self.betas[perm, ...]
        if self.nus is not None:
            self.nus = self.nus[perm, ...]
        if self.W_invs is not None:
            self.W_invs = self.W_invs[perm, ...]
        super().permute(perm)

    def number_of_params(self):
        # betas
        n_params = self.n_hidden
        # ms
        n_params += self.n_hidden * self.n_obs
        # nus
        n_params += self.n_hidden
        # W_invs
        n_params += self.n_hidden * self.n_obs * (self.n_obs + 1) // 2
        return n_params

    def check_hyperparams(self):
        ... # TODO



class MVGamma(Emission):
    """
    Base class for what an emission probability needs to work with the hmt package.
    """
    def __init__(self, n_hidden, n_obs, alphas=None, betas=None, tol=1e-6, maxits=20):
        super().__init__(n_hidden, n_obs)
        self.type = "ML"
        self.alphas = np.asarray(alphas)
        self.betas = np.asarray(betas)
        self.tol = tol
        self.maxits = maxits
    

    def __str__(self):
        return "MV-Gamma (ML)"


    def init_params(self, hmmodel=None):
        if self.alphas is None:
            self.alphas = np.random.randint(1, 10, size=(self.n_hidden, self.n_obs))

        if self.betas is None:
            self.betas = np.random.rand(self.n_hidden, self.n_obs) * 2 # Sample from uniform distribution over [0, 2]


    def check_params(self):
        if self.alphas is not None:
            if self.alphas.shape[0] != self.n_hidden:
                raise HMTError("alphas must have n_hidden values / vectors")
            if self.alphas.shape[1] != self.n_obs:
                raise HMTError("each vector of alphas must have length n_obs")
            if np.any(self.alphas <= 0):
                print(self.alphas)
                raise HMTError("All elements of alphas must be > 0")
        if self.betas is not None:
            if self.betas.shape[0] != self.n_hidden:
                raise HMTError("betas must have n_hidden values / vectors")
            if self.betas.shape[1] != self.n_obs:
                raise HMTError("each vector of betas must have length n_obs")
            if np.any(self.betas <= 0):
                raise HMTError("All elements of betas must be > 0")
    

    def set_params(self, alphas=None, betas=None, check_params=True):
        if alphas is not None:
            self.alphas = np.asarray(alphas)
        if betas is not None:
            self.betas = np.asarray(betas)
        if check_params:
            self.check_params()
    

    def get_params(self):
        return {
            'alphas': self.alphas,
            'betas': self.betas
        }


    def clear_params(self):
        self.alphas = None
        self.betas = None
    

    def number_of_params(self):
        # alpha and beta for each hidden state and observation
        return self.n_hidden * self.n_obs * 2
    

    def permute(self, perm):
        self.alphas = self.alphas[perm, ...]
        self.betas = self.betas[perm, ...]


    def pdf(self, x):
        # # Work out the pdf for each observation individually
        # pdf = x ** (self.alphas - 1) * np.exp(-self.betas * x)
        # pdf *= self.betas ** self.alphas / gamma(self.alphas)
        # # Marginalise out null values by setting proabability equal to 1
        # pdf = np.where(np.isnan(pdf), 1, pdf)
        # return pdf.prod(axis=1)

        log_pdf = (self.alphas - 1) * np.log(x) - self.betas * x 
        log_pdf += self.alphas * np.log(self.betas) - loggamma(self.alphas)
        log_pdf = np.where(np.isnan(log_pdf), 0, log_pdf)
        log_pdf = log_pdf.sum(axis=1)
        return np.exp(log_pdf)



    def update_params(self, hmmodel):
        hmmodel.Estep()
        xbar = hmmodel.sum(('xi', 'x'), func=lambda xi, x: np.outer(xi, x))
        logxbar = hmmodel.sum(('xi', 'x'), func=lambda xi, x: np.outer(xi, np.log(x)))

        # xi_sum = hmmodel.sum('xi')
        # print((xbar.T / xi_sum).T)

        if hmmodel.has_null:
            xi_null_sum = hmmodel.xi_null_sum()
            xbar /= xi_null_sum
            logxbar /= xi_null_sum
            # print("Here")
        else:
            xi_sum = hmmodel.sum('xi')
            xbar = (xbar.T / xi_sum).T
            logxbar = (logxbar.T / xi_sum).T
        
        # print(xbar)
        # assert False

        s = np.log(xbar) - logxbar
        # print(s)
        curr_alphas = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        # print(curr_alphas)
        converged = False
        for it in range(self.maxits): # Newton-Raphson approximation to improve estimate
            prev_alphas = curr_alphas.copy()
            curr_alphas -= (np.log(prev_alphas) - digamma(prev_alphas) - s) / (1 / prev_alphas - polygamma(1, prev_alphas))
            # print(abs(prev_alphas - curr_alphas))
            if abs(prev_alphas - curr_alphas).max() < self.tol:
                converged = True
                break
        # print()
        if not converged:
            warnings.warn(
                f"Gamma alphas did not converge in {self.maxits} iterations, try changing the tol and maxits parameters."
                )
        self.alphas = curr_alphas
        self.betas = self.alphas / xbar
        self.check_params()
    

    def sample(self, hidden_state):
        return np.squeeze(np.random.gamma(self.alphas[hidden_state], 1 / self.betas[hidden_state]))


# class VB_MVGamma(MVGamma):
#     """Llera, Beckmann (2016)"""
#     def __init__(self, n_hidden, n_obs, alphas=None, betas=None):
#         super().__init__(n_hidden, n_obs, alphas, betas)
#         self.type = "VB"

#         self.prior_aas = None
#         self.prior_bs = None
#         self.prior_cs = None
#         self.prior_ds = None

#         self.aas = None
#         self.bs = None
#         self.cs = None
#         self.ds = None
    

#     def __str__(self):
#         return "Gamma (VB)"
    

#     def set_priors(self, aas=None, bs=None, cs=None, ds=None):
#         if aas is not None:
#             self.prior_aas = aas
#         if bs is not None:
#             self.prior_bs = bs
#         if cs is not None:
#             self.prior_cs = cs
#         if ds is not None:
#             self.prior_ds = ds
    

#     def permute(self, perm):
#         """Permutes the hidden states (axis=0) of model parameters"""
#         self.aas = self.aas[perm]
#         self.bs = self.bs[perm]
#         self.cs = self.cs[perm]
#         self.ds = self.ds[perm]
#         # TODO do the same for priocs: build this into emission root class
#         # with setattr(getattr(param)[perm]) for param in self.params, and 
#         # make params a list of the parametecs (vb.params += hyperparams)
#         super().permute() # Permute parametecs as well as hyperparametecs
    

#     def init_hyperparams(self):
#         """
#         Initialise priocs to least informative and initialise random initial value for each hyperparameter
#         """
#         if self.prior_aas is not None:
#             self.prior_aas = 1
#         if self.prior_bs is not None:
#             self.prior_bs = 0
#         if self.prior_cs is not None:
#             self.prior_cs = 0
#         if self.prior_ds is not None:
#             self.prior_ds = 0


#     def update_params(self):
#         """Overwrite ML update_params to have VB updates"""
#         # TODO figure out update formulae - not as obvious as others I fear
#         self.alphas = ...
#         self.betas = ...


#     def update_hyperparams(self, vb_hmmodel):
#         ...


#     def clear_hyperparams(self):
#         self.aas = None
#         self.bs = None
#         self.cs = None
#         self.ds = None
    

#     def check_hyperparams(self):
#         ...


class VBMVGamma(MVGamma):
    """From Miller (1980)"""
    def __init__(self, n_hidden, n_obs, alphas=None, betas=None):
        super().__init__(n_hidden, n_obs, alphas, betas)
        self.type = "VB"

        self.prior_ps = None
        self.prior_logps = None
        self.prior_qs = None
        self.prior_rs = None
        self.prior_ss = None

        self.ps = None
        self.qs = None
        self.rs = None
        self.ss = None
    

    def __str__(self):
        return "Gamma (VB)"
    

    def set_priors(self, ps=None, qs=None, rs=None, ss=None):
        if ps is not None:
            self.prior_ps = ps
            self.prior_logps = np.log(ps)
        if qs is not None:
            self.prior_qs = qs
        if rs is not None:
            self.prior_rs = rs
        if ss is not None:
            self.prior_ss = ss
    

    def permute(self, perm):
        """Permutes the hidden states (axis=0) of model parameters"""
        self.ps = self.ps[perm, ...]
        self.qs = self.qs[perm, ...]
        self.rs = self.rs[perm, ...]
        self.ss = self.ss[perm, ...]
        # TODO do the same for priors: build this into emission root class
        # with setattr(getattr(param)[perm]) for param in self.params, and 
        # make params a list of the parameters (vb.params += hyperparams)
        super().permute() # Permute parameters as well as hyperparameters
    

    def init_hyperparams(self):
        """
        Initialise priors to least informative and initialise random initial value for each hyperparameter
        """
        if self.prior_ps is None:
            self.prior_ps = np.ones((self.n_hidden, self.n_obs))
            self.prior_logps = np.zeros((self.n_hidden, self.n_obs))
        if self.prior_qs is None:
            self.prior_qs = np.zeros((self.n_hidden, self.n_obs))
        if self.prior_rs is None:
            self.prior_rs = np.zeros((self.n_hidden, self.n_obs))
        if self.prior_ss is None:
            self.prior_ss = np.zeros((self.n_hidden, self.n_obs))


    # def update_params(self):
    #     """Overwrite ML update_params to have VB updates"""
    #     # TODO figure out update formulae - not as obvious as others I fear
    #     self.alphas = ...
    #     self.betas = ...
    # NOTE keeping the original ML update_params 


    def update_hyperparams(self, vb_hmmodel):
        # NOTE we might have to convert everything to log space
        # since p = prod(x_u^xi_u) is going to blow up
        # Another option is to scale the data before hand? This would work
        # since a scaled gamma is also gamma - but because its basically x^n
        # for n -> \infty then we either get overflow or under flow :)))))))
        # (stick to log idea)
        self.logps = self.prior_logps + vb_hmmodel.sum(
            attrs=('xi', 'x'),
            func=lambda xi, x: np.outer(xi, np.log(x))
        )
        self.ps = np.exp(self.logps)

        self.qs = self.prior_ps + vb_hmmodel.sum(attrs=('xi', 'x'), func=np.outer)

        Ni = vb_hmmodel.sum(attrs='xi')
        self.rs = (self.prior_rs.T + Ni).T
        self.ss = (self.prior_ss.T + Ni).T


    def clear_hyperparams(self):
        self.ps = None
        self.qs = None
        self.rs = None
        self.ss = None
    

    def check_hyperparams(self):
        ...