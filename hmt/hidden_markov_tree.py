"""
Implementation of Hidden Markov Trees (HMTree and HMNode classes).
"""

import numpy as np
from timeit import default_timer
from itertools import permutations

from hmt.core import Node, Tree, Forest
from hmt.exceptions import HMTError
from hmt.utils import div0, rowwise_outer, mvn_pdf, normal_round, digamma


class HMModel:
    """
    Base class for parameter methods for hidden Markov trees. Used to implement
    basic checks when setting parameters
    
    Methods
    -------
    init_params()
        Initialises parameters
    check_params()
        Checks parameters and raises exceptions if there are problems with any
    set_params()
        Sets parameters specified in arguments
    get_params()
        Returns dictionary with current parameters
    clear_params()
        Sets all parameters to None
    """
    def __init__(self):
        pass


    def init_params(self):
        if self.init_s_distr is None:
            # Randomly initialise initial distribution
            self.init_s_distr = np.random.uniform(0, 1, self.n_hidden)
            # Normalise
            self.init_s_distr /= np.sum(self.init_s_distr)

        if self.P is None:
            # Randomly initialise transistion matrix
            if self.sister_dep:
                self.P = np.random.uniform(0, 1, (self.n_hidden, self.n_hidden, self.n_hidden))
                # Normalise for each value of i
                self.P = (self.P.T / np.sum(self.P, axis=(1, 2))).T
                # self.P = np.full((self.n_hidden, self.n_hidden, self.n_hidden), 1 / self.n_hidden ** 2)
                if not self.daughter_order:
                    self.P = (self.P + np.transpose(self.P, axes=(0, 2, 1))) / 2
            else:
                self.P = np.random.uniform(0, 1, (self.n_hidden, self.n_hidden))
                # Normalise for each value of i
                self.P = (self.P.T / np.sum(self.P, axis=1)).T

        if self.mus is None:
            # Randomly initialise r means of size n
            self.mus = np.random.randn(self.n_hidden, self.n_obs)
        if self.sigmas is None:
            # Initialise sigma based on variance in the data NOTE useless if normalising first
            var = self.var()
            if not isinstance(var, np.ndarray):
                var = np.array([var])
            self.sigmas = np.stack([np.diag(var) for _ in range(self.n_hidden)])
            # Ensure positive definite
            # self.sigmas = self.sigmas @ np.transpose(self.sigmas, axes=(0, 2, 1))

            if (np.linalg.eigvals(self.sigmas) <= 0).any():
                raise HMTError("Initial sigma should be positive definite")
            if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
                raise HMTError("Initial sigma should be symmetric")

            self.sigmainv = np.linalg.inv(self.sigmas)
            self.detsigma = np.linalg.det(self.sigmas)


    def check_params(self, init_s_distr=None, P=None, emit_mat=None, mus=None, sigmas=None):
        if init_s_distr is not None:
            if init_s_distr.shape[0] != self.n_hidden:
                raise HMTError("Initial distribution must be defined for all s values")
            if not np.isclose(sum(init_s_distr), 1):
                raise HMTError('init_s_distr must be a probability distribution.')

        if P is not None:
            if P.shape[0] != self.n_hidden:
                raise HMTError('P must be rxr or rxrxr where r is the number of hidden states.')
            if P.ndim == 2:
                if not np.allclose(P.sum(axis=1), 1):
                    raise HMTError('Rows of P must sum to 1.')
                if P.shape[0] != P.shape[1]:
                    raise HMTError('P must be a square matrix.')
                if P.shape[0] != init_s_distr.shape[0]:
                    raise HMTError('P must have the number of columns as init_s_distr.')
            if P.ndim == 3:
                # Sister dependence
                if P.shape[0] != P.shape[1] or P.shape[0] != P.shape[2]:
                    raise HMTError('P must be a cube.')
                if not np.allclose(P.sum(axis=(1, 2)), 1):
                    raise HMTError('Matrices in of P must sum to 1.')

        if emit_mat is not None:
            if not np.allclose(emit_mat.sum(axis=1), 1):
                raise HMTError('Rows of emit must sum to 1.')
            if emit_mat.shape[0] != P.shape[0]:
                raise HMTError('Emit must have the same number of rows as P')
        
        if mus is not None:
            
            if mus.shape[0] != self.n_hidden:
                raise HMTError("mu must have shape rxn where r is the number of hidden states and n is the number of observed states")
            
            # TODO check when nobs is 1D
            if self.n_obs > 1 and mus.shape[1] != self.n_obs:    
                raise HMTError("mu must have shape rxn where r is the number of hidden states and n is the number of observed states")
            

        if sigmas is not None:
            if sigmas.shape[0] != self.n_hidden:
                raise HMTError("Emission covaraince matrix must have shape rxnxn where r is the number of hidden states and n is the number of observed states")
            if not (np.linalg.eigvals(sigmas) > 0).all():
                raise HMTError("Sigma should be positive definite")

            if not np.allclose(sigmas, np.transpose(sigmas, axes=(0, 2, 1))):
                raise HMTError("Sigma should be symmetric")
            # TODO check when nobs is 1D
            # if sigmas.ndim != 3: 
            #     raise HMTError("Emission covaraince tensor must be 3d, i.e. r 2d covariance matrices where r is the number of hidden states")
            
            # if sigmas.shape[1] != self.n_obs or sigmas.shape[2] != self.n_obs:
            #     raise HMTError("Emission covaraince matrix must have shape rxnxn where r is the number of hidden states and n is the number of observed states")


    def set_params(self, init_s_distr=None, P=None, emit_mat=None, mus=None,
                         sigmas=None, sigmainv=None, detsigma=None, sister_dep=None, daughter_order=None,
                         check_params=True):
        if sister_dep is not None:
            self.sister_dep = sister_dep
        
        if sister_dep is not None:
            self.daughter_order = daughter_order
        
        if check_params:
            self.check_params(init_s_distr, P, emit_mat, mus, sigmas)

        if init_s_distr is not None:
            self.init_s_distr = init_s_distr.copy()

        if P is not None:
            self.P = P.copy()

        if emit_mat is not None:
            self.emit_mat = emit_mat.copy()

        if mus is not None:
            self.mus = mus.copy()

        if sigmas is not None:
            self.sigmas = sigmas.copy()

        if sigmainv is None:
            if self.sigmas is not None:
                self.sigmainv = np.linalg.inv(self.sigmas)
            else:
                self.sigmainv = None
        else:
            self.sigmainv = sigmainv.copy()
        if detsigma is None:
            if self.sigmas is not None:
                self.detsigma = np.linalg.det(self.sigmas)
            else:
                self.detsigma = None
        else:
            self.detsigma = detsigma.copy()
    

    def get_params(self):
        return {
            'init_s_distr': self.init_s_distr,
            'P': self.P,
            'mus': self.mus,
            'sigmas': self.sigmas,
            'sister_dep': self.sister_dep,
            'daughter_order': self.daughter_order
        }
    

    def clear_params(self):
        self.init_s_distr = None
        self.P = None
        self.mus = None
        self.sigmas = None
        self.sigmainv = None
        self.detsigma = None
    

    def permute(self):
        """
        Takes permutation of *true* hidden states and uses the inverse of that
        to permute the predicted states and parameters
        """
        best_acc, perm = self.best_permutation()
        if perm is None:
            return best_acc
        perm = list(perm)

        # Permute ml_s
        self.permute_ml_s(perm)

        # Permute parameters
        params = [
            'mus', 'sigmas', 'sigmainv', 'detsigma', 'init_s_distr'
            ]
        for attr in params:
            self.permute_attr(attr, perm)

        # Permute P (permute in all dimensions in order)
        self.P = self.P[perm][:, perm]
        if self.P.ndim == 3:
            self.P = self.P[:, :, perm]
        return best_acc


class HMNode(Node):
    """
    A node in a hidden Markov tree. The methods impement the E step recursively via the
    upward-downward algorithm given in [1].

    Attributes
    ----------
    id, y, d0, d1, mother
        See documentation for tree.Node.
    tree: HMTree
        tree is an instance of HMTree and contains the model parameters including:
        init_s_distr, P, mus, sigmas, sister_dependence and daughter_order
    s_distr: np.ndarray
        Distibution of S_u; s_distr[j] = P(S_u = j)
    alpha, beta, m_d_beta, m_d_delta: np.ndarrays
        Intermediate distributions used in later calculations
        see [1] for more details
    xi: np.ndarray
        Distribution of hidden states xi[j] = P(S_u = j | observed data)
    m_d_xi: np.ndarray
        Distribution of hidden states for this node and mother
        m_d_xi[i, j] = P(S_u = j, S_m = i | observed data)
        where m is the mother node.
    ml_s: int
        Maximum likelihood estimate for the value of S_u, takes values of
        0, ..., n where n is the number of possible hidden states
    
    (FROM SDHMNODE - TODO CLEAN UP)
    id, y, d0, d1, mother, tree
        See documentation for tree.Node
    s_distr, sc_distr: np.ndarrays
        Distibutions of hidden states for current node and its children: 
        s_distr[i] = P(S_u = i)
        sc_distr[j, k] = P(S_c(u) = (j, k))
    beta_c, beta_c_u: np.ndarrays
        Intermediate distributions used in later calculations:
        beta_c[j, k] = P
        beta_c_u[i] = 
        see [1] for more details
    xi_f, xi_c, xi: np.ndarray
        Conditional distributions of hidden states given data:
        xi_f[i, j, k] = P(S_f(u) = (i, j, k) | observed data)
        xi[i] = P(S_u = i | observed data)
        xi_c[j, k] = P(S_c(u) = (j, k) | observed data)
    m_d_xi: np.ndarray
        Distribution of hidden states for this node and mother
        m_d_xi[i, j] = P(S_u = j, S_m = i | observed data)
        where m is the mother node.
    ml_s: int
        Maximum likelihood estimate for the value of S_u, takes values of
        0, ..., n where n is the number of possible hidden states
    
    Methods
    -------

    References
    ----------
    [1]
    [2] Me :)
    """
    def __init__(self, node_id, observed, tree=None, s=None):
        super().__init__(node_id, observed, tree)
        self.s = s
        self.ml_s = None

        self.s_distr = None
        self.xi = None

        # No sister dependence
        self.beta = None
        self.m_d_beta = None
        self.m_d_delta = None
        self.m_d_xi = None

        # Sister dependence
        self.sc_distr = None
        self.beta_c = None
        self.beta_c_u = None
        self.xi_c = None
        self.xi_f = None
        
    

    def __repr__(self):
        return f'HMNode({self.id})'


    def clean(self):
        self.ml_s = None

        self.s_distr = None
        self.xi = None

        # No sister dependence
        self.beta = None
        self.m_d_beta = None
        self.m_d_delta = None
        self.m_d_xi = None

        # Sister dependence
        self.sc_distr = None
        self.beta_c = None
        self.beta_c_u = None
        self.xi_c = None
        self.xi_f = None
        if self.d0 is not None:
            self.d0.clean()
        if self.d1 is not None:
            self.d1.clean()



    # Initial downward pass
    def calculate_s_distr(self, drec=False):
        if self.mother is None and self.s_distr is None:
            raise HMTError('Root node has no initial S distribution.')

        sc_distr = self.s_distr @ self.tree.P
        if not np.isclose(sum(sc_distr), 1):
            raise HMTError('S distribution probabilities do not sum to 1.')
        

        # Continue recursion
        if self.d0 is not None:
            self.d0.s_distr = sc_distr
            if drec:
                self.d0.calculate_s_distr(drec)
        if self.d1 is not None:
            self.d1.s_distr = sc_distr
            if drec:
                self.d1.calculate_s_distr(drec)


    # Upward pass
    def upward_pass(self, urec=False):
        if urec:
            # Recursion using post order tree traversal
            if self.d0 is not None:
                self.d0.upward_pass(urec)
            if self.d1 is not None:
                self.d1.upward_pass(urec)
    
        self.beta = self.tree.emit(self.x) * self.s_distr
        if self.d0 is not None:
            self.beta *= self.d0.m_d_beta
        if self.d1 is not None:
            self.beta *= self.d1.m_d_beta
        
        N = np.sum(self.beta)
        self.tree.loglikelihood += np.log(N)
        
        # Normalise beta
        self.beta = self.beta / N
        if not np.isclose(sum(self.beta), 1):
            raise HMTError("Probabilities in beta don't sum to 1.")
        
        # Calculate mother-daughter beta
        self.m_d_beta = self.tree.P @ div0(self.beta, self.s_distr)


    ## Downward Pass
    def downward_pass(self, drec=False):
        """
        Calculates xi_u(j) = P(S_u = j | tree data).
        """
        if self.mother is None:
            self.xi = self.beta
        else:
            row_mult = div0(self.beta, self.s_distr)
            col_mult = div0(self.mother.xi, self.m_d_beta)

            self.m_d_xi = self.tree.P * row_mult
            self.m_d_xi = (self.m_d_xi.T * col_mult).T
        
            # print(np.linalg.norm(self.xi - self.m_d_xi.sum(axis=0))) 
        
            self.xi = self.m_d_xi.sum(axis=0)

        # Ensure probabilities sum to 1
        if not np.isclose(np.sum(self.xi), 1):
            print(f'Xi sum = {np.sum(self.xi)}')
            raise HMTError("Smoothed probabilities do not sum to 1.")
        
        if self.mother is not None and not np.isclose(np.sum(self.m_d_xi), 1):
            print(f'Xi sum = {np.sum(self.m_d_xi)}')
            raise HMTError("Smoothed probabilities do not sum to 1.")
        
        # Continue recursion
        if drec:
            if self.d0 is not None:
                self.d0.downward_pass(drec)
            if self.d1 is not None:
                self.d1.downward_pass(drec)
    

    def mother_xi_sum(self):
        if self.mother is None:
            total = 0
        else:
            total = self.mother.xi.copy()
        if self.d0 is not None:
            total += self.d0.mother_xi_sum()
        if self.d1 is not None:
            total += self.d1.mother_xi_sum()
        return total


    ## Viterbi
    def viterbi(self, urec=False):
        if urec:
            # Recursion using post order tree traversal
            if self.d0 is not None:
                self.d0.viterbi(urec)
            if self.d1 is not None:
                self.d1.viterbi(urec)

        delta = self.tree.emit(self.x)
        if self.d0 is not None:
            delta *= self.d0.m_d_delta
        if self.d1 is not None:
            delta *= self.d1.m_d_delta

        self.m_d_delta = np.amax(
            self.tree.P.T * delta, axis=0
            )
        return delta


    ## Backtrack to find optimal hidden state S_u
    def extract_ml_s(self, drec=False):
        if self.mother is None:
            self.ml_s = np.argmax(self.xi)
        else:
            self.ml_s = np.argmax(self.m_d_xi[self.mother.ml_s, :])

        # Continue recursion
        if drec:
            if self.d0 is not None: 
                self.d0.extract_ml_s(drec)
            if self.d1 is not None:
                self.d1.extract_ml_s(drec)
    

    """=================== SISTER DEPENDENCE STARTS HERE ==================="""


    def calculate_sd_s_distr(self, drec=False):
        if self.mother is None and self.s_distr is None:
            raise HMTError('Root node has no initial distribution.')
        
        if not np.isclose(np.sum(self.s_distr), 1):
            raise HMTError('S distribution probabilities do not sum to 1.')

        # Calculate children distribution
        self.sc_distr = (self.tree.P.T @ self.s_distr).T

        if not np.isclose(np.sum(self.sc_distr), 1):
            raise HMTError('S children distribution probabilities do not sum to 1.')

        # Calculate s distribution of daughter cells
        if self.d0 is not None:
            self.d0.s_distr = np.sum(self.sc_distr, axis=1)
            # Continue recursion
            if drec:
                self.d0.calculate_sd_s_distr(drec)
        if self.d1 is not None:
            self.d1.s_distr = np.sum(self.sc_distr, axis=0)
            # Continue recursion
            if drec:
                self.d1.calculate_sd_s_distr(drec)   
    

    def sd_upward_pass(self, urec=False):
        if urec:
            # Recursion using post order tree traversal
            if self.d0 is not None:
                self.d0.sd_upward_pass(urec)
            if self.d1 is not None:
                self.d1.sd_upward_pass(urec)

        if self.d0 is None and self.d1 is None:
            self.beta_c_u = 1
            return

        self.beta_c = self.sc_distr.copy()
        if self.d0 is not None:
            # Multiply columns
            self.beta_c = (self.beta_c.T * self.tree.emit(self.d0.x)).T
            self.beta_c = (self.beta_c.T * self.d0.beta_c_u).T
        if self.d1 is not None:
            # Multiply rows
            self.beta_c *= self.tree.emit(self.d1.x)
            self.beta_c *= self.d1.beta_c_u

        N = np.sum(self.beta_c)
        self.tree.loglikelihood += np.log(N)
        self.beta_c = self.beta_c / N
        ## Calculate beta_{c(u), u}
        self.beta_c_u = self.tree.P * div0(self.beta_c, self.sc_distr)
        # Sum over j and k
        self.beta_c_u = np.sum(self.beta_c_u, axis=(1, 2))

        if self.mother is None:
            self.tree.loglikelihood += np.log(np.dot(
                self.tree.emit(self.x),
                self.s_distr * self.beta_c_u
                ))
    

    def sd_downward_pass(self, drec=False):
        if self.d0 is None and self.d1 is None:
            return

        if self.mother is None:
            # Root node
            self.xi_f = div0(self.beta_c, self.sc_distr) * self.tree.P
            self.xi_f = (self.xi_f.T * self.tree.emit(self.x)).T
            self.xi_f /= np.sum(self.xi_f)
            
            # Sum over i and j
            self.xi = np.sum(self.xi_f, axis=(1, 2))
        else:
            self.xi_f = (self.tree.P.T * div0(self.xi, self.beta_c_u)).T
            self.xi_f *= div0(self.beta_c, self.sc_distr)

        # Quick check to see probabilities are the same
        # print(np.allclose(self.xi - np.sum(self.xi_f, axis=(1, 2))))
        self.xi_c = np.sum(self.xi_f, axis=0)

        # if self.id == 1:
        #     print(self.tree.emit(self.x))
        
        if not np.isclose(np.sum(self.xi), 1):
            print(self.id, np.sum(self.xi))
            raise HMTError("Smoothed probabilities do not sum to 1")

        if not np.isclose(np.sum(self.xi_c), 1):
            print(self.id, np.sum(self.xi_c))
            raise HMTError("Smoothed children probabilities do not sum to 1")
        
        ## Calculate xi values of daughter cells and continue recursion
        if self.d0 is not None:
            # Sum over k
            self.d0.xi = np.sum(self.xi_c, axis=1)
            if drec:
                self.d0.sd_downward_pass(drec)
        if self.d1 is not None:
            # Sum over j
            self.d1.xi = np.sum(self.xi_c, axis=0)
            if drec:
                self.d1.sd_downward_pass(drec)
    

    def sd_extract_ml_s(self, drec=False):
        if self.d0 is None and self.d1 is None:
            return

        if self.mother is None:
            self.ml_s = np.argmax(self.xi)

        ml_sc = np.unravel_index(np.argmax(self.xi_f[self.ml_s]), self.xi_c.shape)

        if self.d0 is not None:
            self.d0.ml_s = ml_sc[0]
            # Continue recursion
            if drec:
                self.d0.sd_extract_ml_s(drec)
        if self.d1 is not None:
            self.d1.ml_s = ml_sc[1]
            # Continue recursion
            if drec:
                self.d1.sd_extract_ml_s(drec)


    """=================== SISTER DEPENDENCE ENDS HERE ==================="""


    def n_accurate_nodes(self):
        if self.s is None:
            raise HMTError("Cannot calculate accuracy without true values of s.")
        accuracy = int(self.ml_s == self.s)
        if self.d0 is not None:
            accuracy += self.d0.n_accurate_nodes()
        if self.d1 is not None:
            accuracy += self.d1.n_accurate_nodes()
        return accuracy
    

    def sample_corr(self, s_list, measure="true_s"):
        if self.d0 is None or self.d1 is None:
            # We need both daughters
            return
        if measure == "true_s":
            if self.s is None:
                raise HMTError("True hidden states must be known, try measure='ml_s'.")
            s_list.append((self.s, self.d0.s, self.d1.s))
            
        if measure == "ml_s":
            if self.ml_s is None:
                raise HMTError("Find ML hidden states first.")
            
            s_list.append((self.ml_s, self.d0.ml_s, self.d1.ml_s))

        # Continue recursion
        self.d0.sample_corr(s_list, measure)
        self.d1.sample_corr(s_list, measure)
    

    def xi_null_sum(self, total):
        total[~np.isnan(self.x)] += self.xi
        if self.d0 is not None:
            self.d0.xi_null_sum(total)
        if self.d1 is not None:
            self.d1.xi_null_sum(total)
    

    def sample(self, N, p):
        if N == 1:
            self.tree.leaves.append(self)
            return
        # Randomly choose p daughter
        q0 = np.random.choice((0, 0.5, 1), p=(p/2, 1 - p, p/2))
        N0 = normal_round(q0 * (N - 1))
        N1 = N - 1 - N0
        
        if self.tree.sister_dep:
            c_distr = self.tree.P[self.s]
            s0, s1  = np.unravel_index(
                np.random.choice(np.arange(self.tree.n_hidden ** 2), 1, p=c_distr.flatten())[0],
                c_distr.shape
                )
        else:
            curr_distr = self.tree.P[self.s]
            s0, s1 = np.random.choice(np.arange(self.tree.n_hidden), 2, p=curr_distr, replace=True)

        if N0 != 0:
            x0 = np.squeeze(np.random.multivariate_normal(self.tree.mus[s0], self.tree.sigmas[s0], 1))
            if self.tree.n_obs == 1:
                x0 = float(x0)
            d0 = HMNode(self.id * 2, observed=x0, tree=self.tree, s=s0)
            d0.mother = self
            d0._path = self._path + '0'
            self.d0 = d0
            self.d0.sample(N0, p)
        if N1 != 0:
            x1 = np.squeeze(np.random.multivariate_normal(self.tree.mus[s1], self.tree.sigmas[s1], 1))
            if self.tree.n_obs == 1:
                x1 = float(x1)
            d1 = HMNode(self.id * 2 + 1, observed=x1, tree=self.tree, s=s1)
            d1.mother = self
            d1._path = self._path + '1'
            self.d1 = d1
            self.d1.sample(N1, p)


class HMTree(Tree, HMModel):
    """
    A structure to encapsulate a hidden Markov tree. The structure of the
    tree itself, and attributes of individual nodes are stored in self.root 
    (a HMNode) while the attributes of the tree (parameters and values) are 
    stored here. The methods are responsible for executing the upward-downward 
    algorithm in the correct order.

    Attributes
    ----------
    root: HMNode
        The root node which contains all nodes in the tree
    init_s_distr: np.ndarray
        Initial distribution of the S values, P(S_1 = j)
    P: np.ndarray
        Transition matrix for the hidden states defined by
        P[i, j] = P(S_u = j | S_m = i)
        where m is the mother node of node u
        or 
        P[i, j, k] = P(S_d0 = j, S_d1 = k | S_m = i)
        if sister_dep == True
    emit: np.ndarray
        Emission matrix, x and s and returns P(X = x | S = s)
    loglikelihood: float
        Loglikelihood of the optimal tree.
    optimal_tree_prob: float
        The probability of the hidden state tree given the observed data
    """
    def __init__(self, n_hidden, n_obs, sister_dep=True, daughter_order=False, root=None):
        super().__init__(root)
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.sister_dep = sister_dep
        self.daughter_order = daughter_order

        self.time = np.zeros(10)

        self.loglikelihood = None
        self.optimal_tree_prob = None
        
        self.init_s_distr = None
        self.P = None
        self.emit_mat = None
        self.mus = None

        self.sigmas = None
        self.sigmainv = None
        self.detsigma = None

        # Variational Bayesian parameters
        self.prior_init_s_weights = None
        self.prior_P_weights = None
        self.prior_mis = None
        self.prior_betas = None
        self.prior_sigma_dof = None
        self.prior_sigma_Winv = None

        self.init_s_weights = None
        self.P_weights = None
        self.mis = None
        self.betas = None
        self.sigma_dof = None
        self.sigma_W = None


    def __repr__(self):
        return f'HMTree(root={self.root})'  


    def __str__(self):
        if self.root is not None:
            self.root.printTree()
            return('')
        return("HMTree()")


    def emit(self, x):
        """
        Returns the vector of emmission probabilities s.t.
        emit(x)[i] = P( X = x | S = i )
        If null values are encountered then probability is marginalised over all
        non-null entries of x.
        """
        if self.emit_mat is not None:
            return self.emit_mat[:, x]
        if not np.isnan(x).any():
            return mvn_pdf(x, self.mus, self.sigmainv, self.detsigma, self.n_obs)
        if np.isnan(x).all():
            raise HMTError("Cannot calculate probability of null vector")

        # If only a subset of x is null then we marginalise the probability
        non_null_indices = (~np.isnan(x)).nonzero()[0]
        # print(non_null_indices)
        marg_x = x[non_null_indices]
        marg_mus = self.mus[:, non_null_indices]
        # print(self.sigmas.shape)
        marg_sigmas = self.sigmas[:, non_null_indices][:, :, non_null_indices]
        try:
            marg_sigmainv = np.linalg.inv(marg_sigmas)
        except:
            raise HMTError("x contains null values and marginalised sigma matrix is singular.")
        marg_detsigma = np.linalg.det(marg_sigmas)
        return mvn_pdf(marg_x, marg_mus, marg_sigmainv, marg_detsigma, non_null_indices.shape[0])


    def viterbi(self):
        for leaf in self.leaves:
            leaf.viterbi(urec=True)
        self.optimal_tree_prob = max(self.init_s_distr * self.root.delta)


    def s_distr(self):
        # Initialise root values S distribution
        self.root.s_distr = 1.0 * self.init_s_distr
        # Recursively calculate S distributions
        if self.sister_dep:
            self.root.calculate_sd_s_distr(drec=True)
        else:
            self.root.calculate_s_distr(drec=True)        


    def upward_pass(self):
        # Reset loglikelihood
        self.loglikelihood = 0
        if self.sister_dep:
            self.root.sd_upward_pass(urec=True)
        else:            
            self.root.upward_pass(urec=True)
        return self.loglikelihood
    

    def downward_pass(self):
        if self.sister_dep:
            self.root.sd_downward_pass(drec=True)
        else:
            self.root.downward_pass(drec=True)


    def Estep(self):
        # Handle Errors
        if self.init_s_distr is None:
            raise HMTError('Tree has no initial S distribution.')
        if self.P is None:
            raise HMTError('Tree has no transition matrix P.')
        if self.emit is None:
            raise HMTError('Tree has no emission distribution.')

        # Caluculate the hidden state distribution for each node
        start = default_timer()
        self.s_distr()
        stop = default_timer()
        self.time[0] += stop - start
        
        # Perform the upward and downward passes to calculate smoothed probabilities
        start = default_timer()
        log_lk = self.upward_pass()
        stop = default_timer()
        self.time[1] += stop - start

        start = default_timer()
        self.downward_pass()
        stop = default_timer()
        self.time[2] += stop - start
        return log_lk


    def Mstep(self):
        ## Update initial S distribution
        self.init_s_distr = self.root.xi.copy()
        if not np.isclose(np.sum(self.init_s_distr), 1):
            raise HMTError("Updating initial distribution has gone wrong.")

        xi_sum = self.sum('xi')
        leaf_sum = np.sum([l.xi for l in self.leaves], axis=0)

        ## Update P
        if self.P.ndim == 2:
            m_d_xi_sum = self.sum('m_d_xi')
            mother_xi_sum = self.root.mother_xi_sum()
            self.P = (m_d_xi_sum.T / mother_xi_sum).T
            if not np.allclose(np.sum(self.P, axis=1), 1):
                print(np.sum(self.P, axis=1))
                raise HMTError("Updating P matrix has gone wrong.")
        if self.P.ndim == 3:
            xi_f_sum = self.sum('xi_f')
            if self.daughter_order:
                self.P = (xi_f_sum.T / (xi_sum - leaf_sum)).T
            else:
                xi_f_sum = (xi_f_sum + np.transpose(xi_f_sum, axes=(0, 2, 1))) / 2
                self.P = (xi_f_sum.T / (xi_sum - leaf_sum)).T
                if not np.allclose(self.P, np.transpose(self.P, axes=(0, 2, 1))):
                    raise HMTError("P not symmetric in update step")
            
            # Ensure P sums to 1
            if not np.allclose(np.sum(self.P, axis=(1, 2)), 1):
                print(f"Iteration {self.it}")
                print(np.sum(self.P, axis=(1, 2)))
                raise HMTError("Updating P matrix has gone wrong.")


        ## Update mu
        self.mus = (self.sum(('xi', 'x'), func=lambda xi, x: np.outer(xi, x)).T / xi_sum).T

        ## Update sigma using updated mu
        self.sigmas = self.sum(
            ('xi', 'x'),
            func=lambda xi, x: (rowwise_outer(x - self.mus, x - self.mus).T * xi).T
            )
        # self.sigmas = self.root.sigma_update_sum()
        self.sigmas = (self.sigmas.T / xi_sum).T

        if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
            raise HMTError("Sigma should be symmetric")

        
        eigs = np.linalg.eigvals(self.sigmas)
        if (eigs < 0).any() or np.isclose(eigs, 0).any():
            # print(f"Iteration {self.it}")
            # print(np.linalg.eigvals(self.sigmas), '\n')
            raise HMTError("Sigma should be positive definite")

            # Use psuedo determinant
            # print(self.sigmas)
            # print(eigs)
            # self.detsigma = np.prod(np.where(eigs > 1.0e-8, eigs, 1), axis=1)
            # print(self.detsigma)
            # Use pseudo inverse
            # self.sigmainv = np.linalg.pinv(self.sigmas)
        # else:
        self.detsigma = np.linalg.det(self.sigmas)
        self.sigmainv = np.linalg.inv(self.sigmas)


    def EMstep(self):
        log_lk = self.Estep()
        self.Mstep()
        return log_lk
    

    """ =========================== Variational Bayesian Algorithm =========================== """

    
    def vb_init_hyperparams(self):
        # Setting initial dirichlet weights to 1 is equivalent to uniform probabilities
        # i.e. a flat prior
        self.prior_init_s_weights = np.full(self.n_hidden, 1)
        self.init_s_weights = self.prior_init_s_weights.copy()

        self.prior_P_weights = np.full((self.n_hidden, self.n_hidden, self.n_hidden), 1)
        self.P_weights = self.prior_P_weights.copy()

        # self.mis = np.random.randn(self.n_hidden, self.n_obs)
        # self.prior_mis = np.zeros((self.n_hidden, self.n_obs))
        mean = self.mean('x')
        self.prior_mis = np.array([mean for _ in range(self.n_hidden)])
        if self.prior_mis.ndim == 1:
            self.prior_mis = self.prior_mis.reshape(-1, 1)
        self.mis = self.prior_mis.copy()

        # self.prior_betas = np.(self.n_hidden, 1/self.n_hidden)
        self.prior_betas = np.zeros(self.n_hidden)
        self.betas = self.prior_betas.copy()

        # Precision hyperparameters
        self.prior_sigma_dof = np.full(self.n_hidden, 1)
        self.sigma_dof = self.prior_sigma_dof.copy()

        # self.prior_sigma_Winv = np.stack([np.eye(self.n_obs) for _ in range(self.n_hidden)])
        
        sample_cov = self.cov()
        if not isinstance(sample_cov, np.ndarray):
            prior_W = np.array([sample_cov for _ in range(self.n_hidden)]).reshape(-1, 1, 1)
        else:
            prior_W = np.stack([sample_cov for _ in range(self.n_hidden)])
        
        # Educated guess
        self.prior_sigma_Winv = np.linalg.inv(prior_W)

        # Flat prior
        # self.prior_sigma_Winv = np.zeros((self.n_hidden, self.n_obs, self.n_obs))

        self.sigma_Winv = self.prior_sigma_Winv.copy()

    
    def vb_parameter_update(self):
        # Update initial S distribution
        self.init_s_distr = np.exp(digamma(self.init_s_weights) - digamma(self.init_s_weights.sum()))
        self.init_s_distr /= self.init_s_distr.sum()
        if not np.allclose(np.sum(self.init_s_distr), 1):
            print(self.init_s_distr)
            raise HMTError("VB update of init_s_distr has gone wrong")
        
        # Update P
        self.P = np.exp((digamma(self.P_weights).T - digamma(self.P_weights.sum(axis=(1, 2)))).T)
        self.P = (self.P.T / self.P.sum(axis=(1, 2))).T
        if not np.allclose(np.sum(self.P, axis=(1, 2)), 1):
            print(np.sum(self.P, axis=(1, 2)))
            raise HMTError("VB update of P matrix has gone wrong.")
        
        # # Update mu
        self.mus = self.mis

        # # Update sigmas
        self.sigmas = (self.sigma_Winv.T / self.sigma_dof).T

        if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
            raise HMTError("Sigma should be symmetric")
        
        eigs = np.linalg.eigvals(self.sigmas)
        if (eigs < 0).any() or np.isclose(eigs, 0).any():
            # print(f"Iteration {self.it}")
            # print(np.linalg.eigvals(self.sigmas), '\n')
            raise HMTError("Sigma should be positive definite")
        
        self.sigmainv = np.linalg.inv(self.sigmas)
        self.detsigma = np.linalg.det(self.sigmas)


    def vb_Estep(self):
        # Update model parameters
        self.vb_parameter_update()
        # Perfom normal E step
        loglk = self.Estep()
        return loglk


    def vb_Mstep(self):
        xi_sum = self.sum('xi')
        xi_f_sum = self.sum('xi_f')
        xi_x_T_sum = self.sum(('xi', 'x'), np.outer)
        x_bar = (xi_x_T_sum.T / xi_sum).T

        # Update initial S distribution weights
        self.init_s_weights = self.prior_init_s_weights + xi_sum

        # Update P weights
        self.P_weights = self.prior_P_weights + xi_f_sum
        # print(self.P_weights)

        # Update mu distribution parameters
        self.betas = xi_sum + self.prior_betas
        self.mis = (self.prior_mis.T * self.prior_betas).T + xi_x_T_sum
        self.mis = (self.mis.T / self.betas).T

        # Update sigma distribution parameters
        self.sigma_dof = self.prior_sigma_dof + xi_sum
        self.sigma_Winv = self.prior_sigma_Winv + self.sum(
            ('xi', 'x'),
            func=lambda xi, x: (rowwise_outer(x - x_bar, x - x_bar).T * xi).T
            )
        self.sigma_Winv += (
            rowwise_outer(x_bar - self.prior_mis, x_bar - self.prior_mis).T * 
            (self.prior_betas * xi_sum) / (self.prior_betas + xi_sum)
            ).T


    def vb_EMstep(self):
        loglk = self.vb_Estep()
        self.vb_Mstep()
        return loglk
    
    " ====== End of Variational Bayes ====== "


    def train(self, tol, maxits, store_log_lks=False, permute=True, overwrite_params=False):
        if overwrite_params:
            self.clear_params()
        self.init_params()
        self.it = 0
        prev_log_lk = self.EMstep()
        self.it += 1
        curr_log_lk = self.EMstep()
        self.it += 1
        if store_log_lks:
            log_lks = np.zeros(maxits)
            log_lks[:2] = prev_log_lk, curr_log_lk

        it = 2

        while abs((curr_log_lk - prev_log_lk) / prev_log_lk) > tol and it < maxits:
            prev_log_lk = curr_log_lk
            curr_log_lk = self.EMstep()
            self.it += 1
            if store_log_lks:
                log_lks[it] = curr_log_lk
            it += 1

        if permute and self.root.s is not None:
            start = default_timer()
            self.permute()
            stop = default_timer()
            self.time[-1] = stop - start

        if store_log_lks:
            return it, log_lks[:it]
        return it

    def extract_ml_s(self):
        if self.sister_dep:
            self.root.sd_extract_ml_s(drec=True)
        else:
            self.root.extract_ml_s(drec=True)


    def best_permutation(self):
        """
        Calculates best permutation of *true* values of hidden states
        """
        # Calculate smoothed probabilities
        self.Estep()
        # Extract MLE S values
        self.extract_ml_s()

        best_acc = self.root.n_accurate_nodes() / len(self)
        best_perm = None
        for perm in permutations(range(self.n_hidden)):
            # Apply permutation
            self.apply(lambda x: perm[x], 's')
            acc = self.root.n_accurate_nodes() / len(self)
            # Apply inverse permutation
            self.apply(lambda x: np.argsort(perm)[x], 's')
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
        return best_acc, best_perm


    def permute_ml_s(self, perm):
        self.apply(lambda ml_s: perm[ml_s], 'ml_s')


    def predict_hidden_states(self):
        """
        The main function.

        Returns
        -------
        accuracy: float
            Percentage of accurately predicted hidden states
        """
        # Calculate smoothed probabilities with current parameters
        self.Estep()
        # Extract MLE S values
        if self.sister_dep:
            self.root.sd_extract_ml_s(drec=True)
        else:
            self.root.extract_ml_s(drec=True)
        if self.root.s is not None:
            # Permute
            best_acc = self.permute()
            return best_acc
    

    def sample_corr(self, measure="true_s"):
        if measure != "true_s" and measure != "ml_s":
            raise HMTError("measure can be 'true_s' or 'ml_s'")
        s_list = []
        self.root.sample_corr(s_list, measure)
        
        # Array of state and daughter states for all nodes with 2 daughters
        s = np.array(s_list)
        # Initialise correlation array 
        r = np.zeros(self.n_hidden)

        for i in range(self.n_hidden):
            # Filter so that mother node is in state i
            si = s[s[:, 0] == i]
            n = si.shape[0]
            _, d0_mean, d1_mean = np.mean(si, axis=0)
            
            # Sample standard deviation
            d0_std = np.std(si[:, 1], ddof=1)
            d1_std = np.std(si[:, 2], ddof=1)

            r[i] = np.dot(si[:, 1], si[:, 2]) - n * d0_mean * d1_mean
            if d0_std == 0 or d1_std == 0:
                r[i] = np.nan
            else:
                r[i] /= (n - 1) * d0_std * d1_std

        return r
    

    def xi_null_sum(self, total):
        self.root.xi_null_sum(total)


    def clear_params(self):
        super().clear_params()
        self.root.clean()


    def number_estimated_parameters(self):
        n, r = self.n_obs, self.n_hidden

        # Initial distribution
        total = r - 1

        # Mu
        total += n * r

        # Sigma
        total += r * n * (n + 1) / 2

        # Transition matrix
        if not self.sister_dep:
            return total + r * (r - 1)
        # Sister dependent
        if self.daughter_order:
            return total + r * (r ** 2 - 1)
        # Sister dependent without daughter order
        return total + r * (r ** 2 + r - 2) / 2
    

    def sample(self, N, p, root_id=1):
        s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
        x = np.squeeze(np.random.multivariate_normal(self.mus[s], self.sigmas[s], 1))
        if self.n_obs == 1:
            x = float(x)
        self.root = HMNode(root_id, observed=x, tree=self, s=s)
        self.root.sample(N, p)
    

    # def sample_both(self, N, p):
    #     tree = HMTree(self.n_hidden, self.n_obs, self.sister_dep, self.daughter_order)
    #     s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
    #     x = np.squeeze(np.random.multivariate_normal(self.mus[s], self.sigmas[s], 1))
    #     self.root = SDHMNode(1, observed=x, tree=self, s=s)
    #     tree.root = HMNode(1, observed=x, tree=tree, s=s)
    #     self.root.sample(N, p, tree.root)
    #     return tree


class HMForest(Forest, HMModel):
    """
    Class for a group of hidden Markov trees that can be trained as a group or
    individually. See HMTree for more details about training and [2] for more details
    about ensemble updates.

    Attributes
    ----------
    n_hidden, n_obs: int
        Number of hidden states and observed values per node
    sister_dep, daughter_order: bool
        Indicates whether to model data with sister dependence or not. If 
        sister dependence is true, daughter_order can be set to true if cells
        have uneven divisions - see [2].
    
    Methods
    -------
    update_trees()
        Ensures all trees have the same parameters as the forest (used for
        ensemble training)

    References
    ----------
    [1]
    [2] - Me :)
    """
    def __init__(self, n_hidden, n_obs, sister_dep=True, daughter_order=False):
        tree_kwargs = {
            "n_hidden" : n_hidden,
            "n_obs" : n_obs,
            "sister_dep" : sister_dep,
            "daughter_order" : daughter_order
        }
        super().__init__(HMTree, HMNode, tree_kwargs)

        # Model parameters
        self.n_hidden = n_hidden
        self.n_obs = n_obs

        self.sister_dep = sister_dep
        self.daughter_order = daughter_order

        # TODO implement automatic test or specify as parameter
        self.has_null = True

        # Tree Parameters
        self.init_s_distr = None
        self.P = None
        self.emit_mat = None
        self.mus = None
        self.sigmas = None
        self.sigmainv = None
        self.detsigma = None
        self.loglikelihood = None


    def update_trees(self):
        for tree in self.trees:
            tree.n_hidden = self.n_hidden
            tree.n_obs = self.n_obs
            tree.sister_dep = self.sister_dep
            tree.daughter_order = self.daughter_order
            tree.set_params(**self.get_params())
    

    def set_params(self, init_s_distr=None, P=None, emit_mat=None, mus=None,
                         sigmas=None, sigmainv=None, detsigma=None, **unused):
        super().set_params(init_s_distr, P, emit_mat, mus, sigmas, sigmainv, detsigma, **unused)
        self.update_trees()
    

    def clear_params(self):
        super().clear_params()
        for tree in self.trees:
            tree.clear_params()


    def init_params(self, training='ensemble'):
        if training == 'ensemble':
            # Initialise parameters and ensure all trees have same parameters
            super().init_params()
            self.update_trees()
            return
        if training == 'individual':
            # Initialise parameters on each tree individually
            for tree in self.trees:
                tree.init_params()
            return
        raise HMTError("Invalid training option, options are 'ensemble' and 'individual'")


    def Estep(self):
        self.loglikelihood = 0
        for tree in self.trees:
            self.loglikelihood += tree.Estep()
        return self.loglikelihood
    

    def Mstep(self):
        self.init_s_distr = np.mean([tree.root.xi for tree in self.trees], axis=0)

        xi_sum = self.sum('xi')

        if self.has_null:
            xi_null_sum = np.zeros((self.n_obs, self.n_hidden))
            for tree in self.trees:
                tree.xi_null_sum(xi_null_sum)

        leaf_sum = 0
        for tree in self.trees:
            leaf_sum += np.sum([l.xi for l in tree.leaves], axis=0)
        
        if self.P.ndim == 2:
            m_d_xi_sum = self.sum('m_d_xi')
            m_d_xi_sum = np.sum([tree.sum('m_d_xi') for tree in self.trees], axis=0)
            mother_xi_sum = np.sum([tree.root.mother_xi_sum() for tree in self.trees], axis=0)
            self.P = (m_d_xi_sum.T / mother_xi_sum).T
            if not np.allclose(np.sum(self.P, axis=1), 1):
                print(np.sum(self.P, axis=1))
                raise HMTError("Updating P matrix has gone wrong.")
        if self.P.ndim == 3:
            xi_f_sum = self.sum('xi_f')
            if self.daughter_order:
                # Symmetry not enforced
                self.P = (xi_f_sum.T / (xi_sum - leaf_sum)).T
            else:
                # Enforce symmetry
                xi_f_sum = (xi_f_sum + np.transpose(xi_f_sum, axes=(0, 2, 1))) / 2
                self.P = (xi_f_sum.T / (xi_sum - leaf_sum)).T
                if not np.allclose(self.P, np.transpose(self.P, axes=(0, 2, 1))):
                    raise HMTError("P not symmetric in update step")
            if not np.allclose(np.sum(self.P, axis=(1, 2)), 1):
                print(np.sum(self.P, axis=(1, 2)))
                raise HMTError("Updating P matrix has gone wrong.")
        
        # Update mu
        self.mus = self.sum(('xi', 'x'), func=lambda xi, x: np.outer(xi, x))
        if self.has_null:
            for m in range(self.n_obs):
                self.mus[:, m] /= xi_null_sum[m]
        else:
            self.mus = (self.mus.T / xi_sum).T

        # Update sigma using updated mu
        self.sigmas = self.sum(
            ('xi', 'x'),
            func=lambda xi, x: (rowwise_outer(x - self.mus, x - self.mus).T * xi).T
            )
        if self.has_null:
            for m in range(self.n_obs):
                scale = xi_null_sum[m]
                self.sigmas[:, m, :] /= scale.reshape(-1, 1)
                self.sigmas[:, m, m] *= scale
                self.sigmas[:, :, m] /= scale.reshape(-1, 1)
        else:
            self.sigmas = (self.sigmas.T / xi_sum).T

        if not (np.linalg.eigvals(self.sigmas) > 0).all():
            print(np.linalg.eigvals(self.sigmas), '\n')
            raise HMTError("Sigma should be positive definite")

        if not np.allclose(self.sigmas, np.transpose(self.sigmas, axes=(0, 2, 1))):
            raise HMTError("Sigma should be symmetric")

        self.sigmainv = np.linalg.inv(self.sigmas)
        self.detsigma = np.linalg.det(self.sigmas)

        self.update_trees()

    
    def EMstep(self, training='ensemble'):
        if training == 'ensemble':
            loglikelihood = self.Estep()
            self.Mstep()
            return loglikelihood
        if training == 'individual':
            self.loglikelihood = 0
            for tree in self.trees:
                self.loglikelihood += tree.EMstep()
            return self.loglikelihood
        raise HMTError("Invalid training option, options are 'ensemble' and 'individual'")


    def train(self, tol, maxits, training='ensemble', store_log_lks=False, overwrite_params=False, permute=False):
        if overwrite_params:
            self.clear_params()
        self.init_params(training)

        prev_log_lk = self.EMstep(training)
        curr_log_lk = self.EMstep(training)

        if store_log_lks:
            log_lks = np.zeros(maxits)
            log_lks[:2] = prev_log_lk, curr_log_lk

        it = 2

        while abs((curr_log_lk - prev_log_lk) / prev_log_lk) > tol:
            prev_log_lk = curr_log_lk
            curr_log_lk = self.EMstep(training)
            if store_log_lks:
                log_lks[it] = curr_log_lk
            it += 1

        if store_log_lks:
            return it, log_lks[:it]    
        
        return it

    
    def predict_hidden_states(self):
        """
        The main function.

        Returns
        -------
        accuracy: float
            Percentage of accurately predicted hidden states
        """
        # Calculate smoothed probabilities with current parameters
        self.Estep()

        # Extract MLE S values
        for tree in self.trees:
            if self.sister_dep:
                tree.root.sd_extract_ml_s(drec=True)
            else:
                tree.root.extract_ml_s(drec=True)

        if self.trees[0].root.s is not None:
            # Permute
            best_acc = self.permute()
            return best_acc
    
    
    def acc(self):
        n_acc = np.sum([tree.root.n_accurate_nodes() for tree in self.trees], axis=0)
        n_tot = np.sum([len(tree) for tree in self.trees], axis=0)
        return n_acc / n_tot


    def best_permutation(self):
        """
        Calculates best permutation of *true* values of hidden states
        """
        # Calculate smoothed probabilities
        self.Estep()
        # Extract MLE S values
        for tree in self.trees:
            tree.extract_ml_s()


        best_acc = self.acc()
        best_perm = None
        for perm in permutations(range(self.n_hidden)):
            # Apply permutation
            for tree in self.trees:
                tree.apply(lambda s: perm[s], 's')
            
            # Find accuracy with this permutation
            acc = self.acc()
            
            # Apply inverse permutation
            for tree in self.trees:
                tree.apply(lambda s: np.argsort(perm)[s], 's')
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
        return best_acc, best_perm
    

    def permute_ml_s(self, perm):
        for tree in self.trees:
            tree.permute_ml_s(perm)


    def sample_corr(self, measure="true_s"):
        if measure != "true_s" and measure != "ml_s":
            raise HMTError("measure can be 'true_s' or 'ml_s'")
        s_list = []

        for tree in self.trees:
            tree.root.sample_corr(s_list, measure)
        
        # Array of state and daughter states for all nodes with 2 daughters
        s = np.array(s_list)
        # Initialise correlation array 
        r = np.zeros(self.n_hidden)
        n = np.zeros(self.n_hidden, dtype=int)

        for i in range(self.n_hidden):
            # Filter so that mother node is in state i
            si = s[s[:, 0] == i]
            n[i] = si.shape[0]
            _, d0_mean, d1_mean = np.mean(si, axis=0)
            
            # Sample standard deviation
            d0_std = np.sqrt(np.dot(si[:, 1] - d0_mean, si[:, 1] - d0_mean) / (n[i] - 1))
            d1_std = np.sqrt(np.dot(si[:, 2] - d1_mean, si[:, 2] - d1_mean) / (n[i] - 1))
            # print(d0_std, d1_std)
            r[i] = np.dot(si[:, 1], si[:, 2]) - n[i] * d0_mean * d1_mean
            if np.isclose(d0_std, 0) or np.isclose(d1_std, 0):
                r[i] = 0
            else:
                r[i] /= (n[i] - 1) * d0_std * d1_std

        return r, n


    def sample(self, n_trees, n_nodes, dropout):
        params = self.get_params()

        tree0  = HMTree(self.n_hidden, self.n_obs, self.sister_dep, self.daughter_order)
        tree0.set_params(**params)
        tree0.sample(n_nodes, dropout, root_id=1)
    
        self.trees.append(tree0)
        for i in range(n_trees - 1):
            tree = HMTree(self.n_hidden, self.n_obs, self.sister_dep, self.daughter_order)
            tree.set_params(**params)
            tree.sample(n_nodes, dropout, root_id = 2 * tree0.leaves[i // 2].id + i%2)
            self.trees.append(tree)
