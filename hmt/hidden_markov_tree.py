"""
Implementation of Hidden Markov Trees (HMTree and HMNode classes).
"""

import numpy as np
import pandas as pd
from timeit import default_timer
from itertools import permutations
from collections import defaultdict
import warnings

from hmt.core import Node, Tree, Forest
from hmt.exceptions import HMTError
from hmt.utils import div0, rowwise_outer, normal_round, digamma
import hmt.emissions as emissions

# logging.basicConfig(level=logging.ERROR, format="hmt %(levelname)s: %(message)s")

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
    def __init__(self, node_id, observed, hmmodel=None, c=None, d=None, t=None, hidden_state=None):
        super().__init__(node_id, observed)
        self.hmmodel = hmmodel
        self.s, self.c, self.d, self.t = hidden_state, c, d, t
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

        # Switches during cell life
        self.next = None
        self.prev = None

        self.Ex = np.empty((self.hmmodel.n_hidden, self.hmmodel.n_obs))


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

        sc_distr = self.s_distr @ self.hmmodel.P
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
    
        self.beta = self.hmmodel.emission(self) * self.s_distr
        if self.d0 is not None:
            self.beta *= self.d0.m_d_beta
        if self.d1 is not None:
            self.beta *= self.d1.m_d_beta
        
        N = np.sum(self.beta)
        self.hmmodel.loglikelihood += np.log(N)
        
        # Normalise beta
        self.beta = self.beta / N
        if not np.isclose(sum(self.beta), 1):
            raise HMTError("Probabilities in beta don't sum to 1.")
            # logging.info("Probabilities in beta don't sum to 1")
        
        # Calculate mother-daughter beta
        self.m_d_beta = self.hmmodel.P @ div0(self.beta, self.s_distr)


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

            self.m_d_xi = self.hmmodel.P * row_mult
            self.m_d_xi = (self.m_d_xi.T * col_mult).T
        
            # print(np.linalg.norm(self.xi - self.m_d_xi.sum(axis=0))) 
        
            self.xi = self.m_d_xi.sum(axis=0)

        # Ensure probabilities sum to 1
        if not np.isclose(np.sum(self.xi), 1):
            print(f'Xi sum = {np.sum(self.xi)}')
            raise HMTError("Smoothed probabilities do not sum to 1.")
            # logging.info("Smoothed probabilities do not sum to 1.\nXi sum = %s".format(np.sum(self.xi)))
        
        if self.mother is not None and not np.isclose(np.sum(self.m_d_xi), 1):
            print(f'Xi sum = {np.sum(self.m_d_xi)}')
            raise HMTError("Smoothed probabilities do not sum to 1.")
            # logging.info("Smoothed probabilities do not sum to 1.")
        
        # Continue recursion
        if drec:
            if self.d0 is not None:
                self.d0.downward_pass(drec)
            if self.d1 is not None:
                self.d1.downward_pass(drec)
    

    def mother_xi_sum(self):
        if self.mother is None or self.prev is not None:
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

        delta = self.hmmodel.emission(self)
        if self.d0 is not None:
            delta *= self.d0.m_d_delta
        if self.d1 is not None:
            delta *= self.d1.m_d_delta

        self.m_d_delta = np.amax(
            self.hmmodel.P.T * delta, axis=0
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


    def calculate_sd_s_distr(self, d_drec=True, n_drec=False):
        if self.mother is None and self.s_distr is None:
            raise HMTError(f'Root node has no initial distribution. {self.hmmodel}, {self.id}')
            # logging.info('Root node has no initial distribution.')
        
        if not np.isclose(np.sum(self.s_distr), 1):
            raise HMTError('S distribution probabilities do not sum to 1.')
            # logging.info('S distribution probabilities do not sum to 1.')

        if self.next is not None:
            self.next.s_distr = self.s_distr @ self.hmmodel.life_P
            assert self.next.s_distr.shape == (self.next.hmmodel.n_hidden, )
            assert np.isclose(self.next.s_distr.sum(), 1)
            if n_drec:
                self.next.calculate_sd_s_distr(d_drec, n_drec)
            return

        # Calculate children distribution
        self.sc_distr = (self.hmmodel.P.T @ self.s_distr).T

        if not np.isclose(np.sum(self.sc_distr), 1):
            raise HMTError('S children distribution probabilities do not sum to 1.')
            # logging.info('S children distribution probabilities do not sum to 1.')
        
        # Calculate s distribution of daughter cells
        if self.d0 is not None:
            self.d0.s_distr = np.sum(self.sc_distr, axis=1)
            # Continue recursion
            if d_drec:
                self.d0.calculate_sd_s_distr(d_drec, n_drec)
        if self.d1 is not None:
            self.d1.s_distr = np.sum(self.sc_distr, axis=0)
            # Continue recursion
            if d_drec:
                self.d1.calculate_sd_s_distr(d_drec, n_drec)

    def sd_upward_pass(self, d_urec=True, n_urec=False):
        if d_urec:
            # Recursion using post order tree traversal
            if self.d0 is not None:
                self.d0.sd_upward_pass(d_urec, n_urec)
            if self.d1 is not None:
                self.d1.sd_upward_pass(d_urec, n_urec)
        if n_urec:
            if self.next is not None:
                self.next.sd_upward_pass(d_urec, n_urec)
        
        if self.next is not None:
            if self.mother is None:
                raise NotImplementedError("Root node with changes not implemented")
            self.hmmodel.urec_next(self)
            return
            self.beta_c_u = self.hmmodel.life_P @ div0(self.next.beta, self.next.s_distr)
            # self.beta = self.hmmodel.emission(self) * self.beta_c_u * self.s_distr
            # self.beta /= np.sum(self.beta) # N_u^k
            # assert self.beta.shape == (self.hmmodel.n_hidden,)
            assert self.beta_c_u.shape == (self.hmmodel.n_hidden, )
            # assert np.allclose(self.beta.sum(), 1), self.beta
            return
        
        if self.d0 is None and self.d1 is None:
            self.beta_c_u = 1
            return

        self.beta_c = self.sc_distr.copy()
        if self.d0 is not None:
            # Multiply columns
            self.beta_c = (self.beta_c.T * self.hmmodel.emission(self.d0)).T
            self.beta_c = (self.beta_c.T * self.d0.beta_c_u).T
        if self.d1 is not None:
            # Multiply rows
            self.beta_c *= self.hmmodel.emission(self.d1)
            self.beta_c *= self.d1.beta_c_u
        N = np.sum(self.beta_c)
        if N == 0:
            print(self.id)
            if self.d0 is not None:
                print(f"d0 emission: {self.hmmodel.emission(self.d0)}")
            if self.d1 is not None:
                print(f"d1 emission: {self.hmmodel.emission(self.d1)}")
            raise HMTError("sum of beta_c is 0")
        self.hmmodel.loglikelihood += np.log(N)
        self.beta_c = self.beta_c / N
        ## Calculate beta_{c(u), u}
        self.beta_c_u = self.hmmodel.P * div0(self.beta_c, self.sc_distr)

        # Sum over j and k
        self.beta_c_u = np.sum(self.beta_c_u, axis=(1, 2))
        assert self.beta_c_u.shape == (self.hmmodel.n_hidden, )

        
        # if self.prev is not None: # (but self.next is None)
        #     self.beta = self.hmmodel.emission(self) * self.beta_c_u * self.s_distr
        #     self.beta /= self.beta.sum()
        #     N = np.sum(self.beta)
        #     self.beta /= N
        #     self.hmmodel.loglikelihood += np.log(N)
        #     return
        
        
        if self.mother is None: # root node
            self.hmmodel.loglikelihood += np.log(np.dot(
                self.hmmodel.emission(self),
                self.s_distr * self.beta_c_u
                ))


    def sd_downward_pass(self, d_drec=True, n_drec=False):
        """
        Args
        ----
        d_drec: bool
            Enables downward recursion for daughters (useful for debugging)
        n_drec: bool
            Enabels downward recursion within chain, turned on or off depending on model used
        """
        if self.next is not None:
            self.hmmodel.drec_next(self)
            if n_drec:
                # Forward pass depends on model being used
                self.next.sd_downward_pass(d_drec, n_drec)
            return
            self.xi_n = (self.hmmodel.life_P.T * div0(self.xi, self.beta_c_u)).T
            self.xi_n *= div0(self.next.beta, self.next.s_distr)
            # self.xi_n = np.full(
            #     (self.hmmodel.n_hidden, self.next.hmmodel.n_hidden),
            #     1 / (self.hmmodel.n_hidden * self.next.hmmodel.n_hidden)
            #     )
            assert self.xi_n.shape == (self.hmmodel.n_hidden, self.next.hmmodel.n_hidden)
            if not np.isclose(self.xi_n.sum(), 1):
                print(self.xi_n)
                print(self.xi_n.sum())
                print(self.xi_n / self.xi_n.sum())
                print(div0(self.next.beta, self.next.s_distr))
                raise HMTError("Smoothed probabilities do not sum to one in chain")
            assert np.allclose(self.xi_n.sum(axis=1), self.xi)
            # print(self.xi, self.xi_n.sum(axis=1))
            # self.xi = self.xi_n.sum(axis=1)
            self.next.xi = self.xi_n.sum(axis=0)
            assert self.next.xi.shape == (self.next.hmmodel.n_hidden,)
            assert np.isclose(self.next.xi.sum(), 1)
            if n_drec:
                self.next.sd_downward_pass(d_drec, n_drec)
            
            
            # self.next.xi = np.full(self.next.hmmodel.n_hidden, 1 / self.next.hmmodel.n_hidden)
            return

        if self.d0 is None and self.d1 is None:
            if self.mother is None:
                # Root node with no daughters
                self.xi = self.hmmodel.emission(self)
                self.xi /= self.xi.sum()
            return
        
        if self.mother is None and self.prev is None:
            # Root node
            self.xi_f = div0(self.beta_c, self.sc_distr) * self.hmmodel.P
            self.xi_f = (self.xi_f.T *  self.hmmodel.emission(self)).T
            self.xi_f /= np.sum(self.xi_f)
            
            # Sum over i and j
            self.xi = np.sum(self.xi_f, axis=(1, 2))
        else:
            self.xi_f = (self.hmmodel.P.T * div0(self.xi, self.beta_c_u)).T
            self.xi_f *= div0(self.beta_c, self.sc_distr)
        # Quick check to see probabilities are the same
        # print(np.allclose(self.xi - np.sum(self.xi_f, axis=(1, 2))))
        self.xi_c = np.sum(self.xi_f, axis=0)

        # if self.id == 1:
        #     print( self.hmmodel.emission(self.x, self.c))
        
        if not np.isclose(np.sum(self.xi), 1):
            print(self.beta_c)
            print(self.id, self.xi, np.sum(self.xi))
            raise HMTError("Smoothed probabilities do not sum to 1")
            # logging.info("Smoothed probabilities do not sum to 1")

        if not np.isclose(np.sum(self.xi_c), 1):
            print(self.id, np.sum(self.xi_c))
            raise HMTError("Smoothed children probabilities do not sum to 1")
            # logging.info("Smoothed children probabilities do not sum to 1")
        
        ## Calculate xi values of daughter cells and continue recursion
        if self.d0 is not None:
            # Sum over k
            self.d0.xi = np.sum(self.xi_c, axis=1)
            assert self.d0.xi.shape == (self.d0.hmmodel.n_hidden,), f"{self.hmmodel.n_hidden, self.d0.hmmodel.n_hidden}"
            if d_drec:
                self.d0.sd_downward_pass(d_drec, n_drec)
        if self.d1 is not None:
            # Sum over j
            self.d1.xi = np.sum(self.xi_c, axis=0)
            assert self.d1.xi.shape == (self.d1.hmmodel.n_hidden,), f"{self.hmmodel.n_hidden, self.d1.hmmodel.n_hidden}"
            if d_drec:
                self.d1.sd_downward_pass(d_drec, n_drec)
    

    def sd_extract_ml_s(self, d_drec=True, n_drec=False):
        if self.next is not None:
            self.hmmodel.extract_next_ml_s(self)
            if n_drec:
                self.next.sd_extract_ml_s(d_drec, n_drec)
            return
        
        if self.mother is None and self.prev is None:
            self.ml_s = np.argmax(self.xi)
        
        if self.d0 is None and self.d1 is None:
            return

        ml_sc = np.unravel_index(np.argmax(self.xi_f[self.ml_s]), self.xi_c.shape)

        if self.d0 is not None:
            self.d0.ml_s = ml_sc[0]
            # Continue recursion
            if d_drec:
                self.d0.sd_extract_ml_s(d_drec, n_drec)
        if self.d1 is not None:
            self.d1.ml_s = ml_sc[1]
            # Continue recursion
            if d_drec:
                self.d1.sd_extract_ml_s(d_drec, n_drec)


    def sd_viterbi(self, d_urec=True):
        if self.next is not None:
            return
        if d_urec:
            # Recursion using post order tree traversal
            if self.d0 is not None:
                # print(self.id, "Left")
                self.d0.sd_viterbi(d_urec)
            if self.d1 is not None:
                # print(self.id, "Right")
                self.d1.sd_viterbi(d_urec)

        # print(f"   **   {self.id}   **")
        self.delta = self.hmmodel.emission(self)
        
        if self.d0 is None and self.d1 is None:
            return
        
        # Store the hidden states of the daughters that are 
        # optimal for each possible hidden state

        # _delta is the part we maximise in the viterbi algorithm
        _delta = self.hmmodel.P
        if self.d0 is not None: # Multiply j component by daughter 0 delta
            _delta = np.transpose(_delta, axes=(0, 2, 1)) * self.d0.delta
            _delta = np.transpose(_delta, axes=(0, 2, 1))
        else: # Marginalise over j component
            _delta = _delta.sum(axis=1)

        if self.d1 is not None: # Multiply k component by daughter 1 delta
            _delta *= self.d1.delta
        else: # Marginalise over k component
            _delta = _delta.sum(axis=2)
        
        # print(f"{_delta.shape = }")
        # print(f"{self.delta.shape = }")
        # End up with either rxrxr tensor (2 daughters) or rxr matrix (1 daughter)

        # For each state i in the current node we find the daughter states that 
        # would maximise the probability of the tree from here
        self.optimal_daughter_states = np.array([
            np.unravel_index(_delta[i].argmax(), _delta[i].shape) for i in range(self.hmmodel.n_hidden)
            ])
        # print(f"{self.optimal_daughter_states.shape = }")
        # print(f"{self.optimal_daughter_states[0].shape = }")
        max_delta = np.zeros(self.hmmodel.n_hidden)
        for i in range(self.hmmodel.n_hidden):
            if self.d0 is not None and self.d1 is not None:
                opt_d0_state, opt_d1_state = self.optimal_daughter_states[i][0], self.optimal_daughter_states[i][0]
                max_delta[i] = _delta[i, opt_d0_state, opt_d1_state]
                continue
            if self.d0 is not None:
                max_delta[i] = _delta[i, self.optimal_daughter_states[i][0]]
            else:
                max_delta[i] = _delta[i, self.optimal_daughter_states[i][1]]

        max_delta = np.array(max_delta)
        # print(f"{max_delta.shape = }")

        self.delta *= max_delta
#       /\
#       |
#      \/
    def sd_viterbi_extract_ml_s(self, drec=False):
        """
        Extracts hidden state of daughter nodes (and current node if self is the root)
        based on Viterbi algorithm
        """
        if self.mother is None: # Root node
            self.ml_s = np.argmax(np.multiply(self.delta, self.s_distr))
        if self.d0 is None and self.d1 is None: # Leaves
            return

        if self.d0 is not None:
            self.d0.ml_s = self.optimal_daughter_states[self.ml_s][0]
            if drec: # Continue recursion
                self.d0.sd_extract_ml_s(drec)
        if self.d1 is not None:
            self.d1.ml_s = self.optimal_daughter_states[self.ml_s][1]
            if drec: # Continue recursion
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
                # logging.info("True hidden states must be known, try measure='ml_s'.")
            s_list.append((self.s, self.d0.s, self.d1.s))
            
        if measure == "ml_s":
            if self.ml_s is None:
                raise HMTError("Find ML hidden states first.")
                # logging.info("Find ML hidden states first.")
            
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
    

    def sample(self, N, p, has_death=False, truncated=False):
        """
        Args
        ---
        truncated: used for debugging   
        """
        if N == 1:
            self.hmmodel.leaves.append(self)
            return 0

        # Randomly choose p daughter
        q0 = np.random.choice((0, 0.5, 1), p=(p/2, 1 - p, p/2))
        N0 = normal_round(q0 * (N - 1))
        N1 = N - 1 - N0
        
        if self.hmmodel.sister_dep:
            c_distr = self.hmmodel.P[self.s]
            s0, s1  = np.unravel_index(
                np.random.choice(np.arange(self.hmmodel.n_hidden ** 2), 1, p=c_distr.flatten())[0],
                c_distr.shape
                )
        else:
            curr_distr = self.hmmodel.P[self.s]
            s0, s1 = np.random.choice(np.arange(self.hmmodel.n_hidden), 2, p=curr_distr, replace=True)
        remaining_0, remaining_1 = 0, 0
        if N0 != 0:
            if has_death:
                death_0, x0 = self.hmmodel.emission_distr.sample(s0) # Assumes death distribution has 2 outputs
            elif truncated:
                trunc_ind = np.random.choice(np.arange(self.hmmodel.n_obs))
                t = np.random.normal(self.hmmodel.emission_distr.mus[s0])
            else:
                x0 = self.hmmodel.emission_distr.sample(s0)
            if self.hmmodel.n_obs == 1:
                x0 = float(x0)
            if isinstance(self.id, str):
                tree, m_id = self.id.split('-')
                cell_id = '-'.join((tree, str(int(m_id) * 2)))
            else:
                cell_id = self.id * 2
            d0 = HMNode(cell_id, observed=x0, hmmodel=self.hmmodel, s=s0)
            d0.mother = self
            d0._path = self._path + '0'
            self.d0 = d0
            if has_death:
                self.d0.d = death_0
                if death_0:
                    self.hmmodel.leaves.append(self.d0)
                    remaining_0 += N0 - 1
                else:
                    remaining_0 += self.d0.sample(N0, p, has_death=has_death)
            else:
                remaining_0 += self.d0.sample(N0, p, has_death=has_death) 
        if N1 != 0:
            if has_death:
                death_1, x1 = self.hmmodel.emission_distr.sample(s1)
            else:
                x1 = self.hmmodel.emission_distr.sample(s1)
            if self.hmmodel.n_obs == 1:
                x1 = float(x1)
            if isinstance(self.id, str):
                tree, m_id = self.id.split('-')
                cell_id = '-'.join((tree, str(int(m_id) * 2 + 1)))
            else:
                cell_id = self.id * 2 + 1
            d1 = HMNode(cell_id, observed=x1, hmmodel=self.hmmodel, s=s1)
            d1.mother = self
            d1._path = self._path + '1'
            self.d1 = d1
            if has_death:
                self.d1.d = death_1
                if death_1:
                    self.hmmodel.leaves.append(self.d1)
                    remaining_1 += N1 - 1
                else:
                    remaining_1 += self.d1.sample(N1, p, has_death=has_death)
            else:
                remaining_1 += self.d1.sample(N1, p, has_death=has_death)
        return remaining_0 + remaining_1 # Return number of remaining cells to sample
    

    def from_numpy(self, X, has_true_s=False, obs_func=None):
        daughter_inds = np.where((X[:, 1] == self.id) & (X[:, 0] != self.id))[0]

        # If no daughters
        if not len(daughter_inds):
            return
        
        ## Add d0
        d0_data = X[daughter_inds[0]]
        obs = d0_data[2 : 2 + self.hmmodel.n_obs].astype(float)
        if obs_func is not None:
            obs = obs_func(obs)

        # Initialise d0 node
        d0 = HMNode(
            node_id=d0_data[0],
            observed=obs,
            hmmodel=self.hmmodel,
            s=d0_data[-1] if has_true_s else None
        )
        # Connect to mother node (i.e. current node = self)
        self.add_daughter(d0)
        
        # Add d0's daughters
        self.d0.from_numpy(X=X, has_true_s=has_true_s, obs_func=obs_func)

        # If only one daughter
        if len(daughter_inds) == 1:
            return
        
        ## Else add d1
        d1_data = X[daughter_inds[1]]
        
        obs = d1_data[2 : 2 + self.hmmodel.n_obs].astype(float)
        if obs_func is not None:
            obs = obs_func(obs)

        # Initialise d1 node
        d1 = HMNode(
            node_id=d1_data[0],
            observed=obs,
            hmmodel=self.hmmodel,
            s=d1_data[-1] if has_true_s else None
        )
        # Connect to mother node (i.e. current node = self)
        self.add_daughter(d1)
        
        # Add d1's daughters
        self.d1.from_numpy(X=X, has_true_s=has_true_s, obs_func=obs_func)
    
    def split(self, start_time, change_time):
        """
        NOTE assumes data is lognormally distributed
        """
        # print("Splitting")
        assert self.hmmodel.next_env is not None
        x = np.where(np.isnan(self.x), self.c, self.x) if self.c is not None else self.x
        cumsum = np.cumsum(np.exp(x))
        phase_ind = np.where(start_time + cumsum > change_time)[0][0]
        next_time = start_time + cumsum[phase_ind] - change_time
        prev_time = change_time - start_time - cumsum[phase_ind - 1] if phase_ind else change_time - start_time
        next_time, prev_time = np.log(next_time), np.log(prev_time)
        # print(start_time + cumsum, change_time)
        # print(np.exp(prev_time))
        # print(next_time)
        next_x, next_c = self.x.copy(), self.c.copy()
        next_x[:phase_ind], next_c[:phase_ind] = np.nan, np.nan
        next_x[phase_ind] = next_time
        next_t = np.full_like(self.x, np.nan)
        next_t[phase_ind] = prev_time
        self.x[phase_ind:], self.c[phase_ind:] = np.nan, np.nan
        self.c[phase_ind] = prev_time

        # print(np.exp(self.x), np.exp(next_x))
        # print(np.exp(self.c), np.exp(next_t), np.exp(next_c))
        next_node = HMNode(
            node_id=self.id,
            observed=next_x,
            hmmodel=self.hmmodel.next_env,
            c=next_c,
            t=next_t,
            d=self.d,
            hidden_state=self.s   
        )
        self.d = 0 if self.d is not None and self.d else self.d
        self.next, next_node.prev = next_node, self
        self.hmmodel.trans_leaves.append(self)
        self.hmmodel.leaves.append(self)
        self.hmmodel.next_env.roots.append(next_node)
        self.hmmodel.next_env.trans_roots.append(next_node)
        
    @staticmethod
    def from_pandas_row(
        row,
        obs_cols,
        hmmodel,
        obs_func=None
        ):
        obs = row[obs_cols].astype(float).to_numpy()
        if obs_func is not None:
            obs = obs_func(obs)
        if "C" in row.index:
            c = np.full_like(obs, np.nan, dtype=float)
            censoring_inds = row["C"]
            if isinstance(censoring_inds, str):
                censoring_inds = eval(censoring_inds)
            if censoring_inds:
                censoring_inds = np.asarray(censoring_inds) - 1
                c[censoring_inds] = obs[censoring_inds]
                obs[censoring_inds] = np.nan
        else:
            c = None
        return HMNode(
            node_id=row['Cell ID'],
            observed=obs,
            c=c,
            d=row["D"] if "D" in row.index else None,
            hmmodel=hmmodel,
            hidden_state=row["hidden_state"] if "hidden_state" in row.index else None
        )

    def from_pandas(
            self,
            df,
            obs_cols,
            obs_func=None,
            curr_time=0,
            change_time=None
            ):
        daughter_inds = df.index[df['Mother ID'] == self.id].tolist()
        # If no daughters
        for i, daughter_ind in enumerate(daughter_inds):
            if i == 2:
                # warnings.warn("Currently only 2 daughters can be read, discarding third")
                break # currently we only allow two daughters to be read
            ## Add daughter
            d = HMNode.from_pandas_row(
                row=df.loc[daughter_ind],
                obs_cols=obs_cols,
                hmmodel=self.hmmodel,
                obs_func=obs_func
            )
            self.add_daughter(d)
            next_time = curr_time + np.sum(np.where(~np.isnan(d.x), np.exp(d.x), 0))
            next_time += np.sum(np.where(~np.isnan(d.c), np.exp(d.c), 0))
            if (
                change_time is not None and
                curr_time < change_time and
                next_time > change_time
                ):
                # print(d.id, round(curr_time), change_time, round(next_time))
                d.split(start_time=curr_time, change_time=change_time)
                d = d.next
            d.from_pandas(
                df=df, obs_cols=obs_cols, curr_time=next_time, change_time=change_time, obs_func=obs_func
                )
            


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
    def __init__(
            self,
            n_hidden,
            n_obs,
            sister_dep=True,
            daughter_order=False,
            has_null=False,
            emission_distr="MVN",
            **emission_kwargs
            ):
        # Model parameters
        self._n_hidden = n_hidden
        self._n_obs = n_obs
        # print(self._n_hidden, self._n_obs)
        self.sister_dep = sister_dep
        self.daughter_order = daughter_order
        self.has_null = has_null

        # Likelihood of the tree / forest to test for convergence
        self.loglikelihood = 0 

        # Model parameters
        self.init_s_distr = None
        self.P = None
        self.set_emissions(emission_distr, **emission_kwargs)

        super().__init__() 
        # This initialises Tree / Forest class when HMTree / HMForest __init__() is called
        # or initialises a python Object class when HMModel.__init__() is directly called
    
    @property
    def n_hidden(self):
        return self._n_hidden
    
    @n_hidden.setter
    def n_hidden(self, value):
        if not isinstance(value, int):
            raise HMTError("Number of hidden states must be an integer.")
        if value < 1:
            raise HMTError("Number of hidden states must be greater than 0.")
        # print(f"Setting n_hidden to {value}")
        self._n_hidden = value
        self.emission_distr.n_hidden = value
        self.clear_params()
    

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


    def init_params(self, init_method='kmeans'):
        if self.init_s_distr is None and not (hasattr(self, "prev_env") and self.prev_env is not None):
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
        
        if self.emission_distr is None:
            raise HMTError("Cannot initialise parameters when emission distribution is None")
        self.emission_distr.init_params(init_method=init_method)


    def check_params(self):
        if self.init_s_distr is not None:
            if self.init_s_distr.shape[0] != self.n_hidden:
                raise HMTError("Initial distribution must be defined for all s values")
            if not np.isclose(sum(self.init_s_distr), 1):
                raise HMTError('init_s_distr must be a probability distribution.')

        if self.P is not None:
            if self.P.shape[0] != self.n_hidden:
                raise HMTError('P must be rxr or rxrxr where r is the number of hidden states.')
            if self.P.ndim == 2:
                if not np.allclose(self.P.sum(axis=1), 1):
                    raise HMTError('Rows of P must sum to 1.')
                if self.P.shape[0] != self.P.shape[1]:
                    raise HMTError('P must be a square matrix.')
                if self.P.shape[0] != self.init_s_distr.shape[0]:
                    raise HMTError('P must have the number of columns as init_s_distr.')
            if self.P.ndim == 3:
                # Sister dependence
                if self.P.shape[0] != self.P.shape[1] or self.P.shape[0] != self.P.shape[2]:
                    raise HMTError('P must be a cube.')
                if not np.allclose(self.P.sum(axis=(1, 2)), 1):
                    raise HMTError('Matrices in of P must sum to 1.')  
        
        self.emission_distr.check_params()


    def set_params(self, init_s_distr=None, P=None, sister_dep=None, daughter_order=None, check_params=True, **emission_params):
        if sister_dep is not None:
            self.sister_dep = sister_dep
        
        if sister_dep is not None:
            self.daughter_order = daughter_order

        if init_s_distr is not None:
            self.init_s_distr = init_s_distr.copy()

        if P is not None:
            self.P = P.copy()
        
        self.emission_distr.set_params(**emission_params, check_params=check_params)
        
        if check_params:
            self.check_params()


    def set_emissions(self, emission_distr="MVN", **emission_kwargs):
        if isinstance(emission_distr, str):
            if emission_distr == "MVN":
                self.emission_distr = emissions.MVN(self.n_hidden, self.n_obs, hmmodel=self, **emission_kwargs)
            elif emission_distr == "Gamma":
                self.emission_distr = emissions.MVGamma(self.n_hidden, self.n_obs, hmmodel=self, **emission_kwargs)
            else:
                raise NotImplementedError("Only MVN and Gamma distributions are implemented by default")
        else:
            self.emission_distr = emission_distr
            emission_distr.hmmodel = self
            if emission_kwargs:
                self.emission_distr.set_params(**emission_kwargs)
        if self.emission_distr.type != "ML":
            raise HMTError("For maximum likelihood (ML) model, emission distribution must also be ML")
    
    def emission(self, node):
        return self.emission_distr.emission(node)

    def get_params(self):
        return {
            'init_s_distr': self.init_s_distr,
            'P': self.P,
            'sister_dep': self.sister_dep,
            'daughter_order': self.daughter_order,
            **self.emission_distr.get_params()
        }

    def clean(self):
        pass

    def clear_params(self):
        self.init_s_distr = None
        self.P = None
        self.emission_distr.clear_params()
        self.clean()
    

    def number_of_params(self):
        # pi
        n_params = self.n_hidden - 1

        # P
        if self.daughter_order:
            n_params += self.n_hidden ** 3 - self.n_hidden
        else:
            n_params += self.n_hidden ** 2 * (self.n_hidden + 1) // 2 - self.n_hidden
        
        # Emission parameters
        n_params += self.emission_distr.number_of_params()
        return n_params        
    

    def permute(self, perm):
        """
        Takes permutation of *true* hidden states and uses the inverse of that
        to permute the predicted states and parameters
        """
        # best_acc, perm = self.best_permutation()
        # if perm is None:
        #     return best_acc
        # perm = list(perm)

        # Permute ml_s
        self.permute_ml_s(perm)
        self.init_s_distr = self.init_s_distr[perm, ...]

        # Permute P (permute in all dimensions in order)
        self.P = self.P[perm, ...][:, perm, ...]
        if self.P.ndim == 3:
            self.P = self.P[:, :, perm]

        # Permute emission parameters=
        self.emission_distr.permute(perm)
        # return best_acc
    

    def permute_ml_s(self, perm):
        self.apply(lambda ml_s: perm[ml_s], 'ml_s')
    

    def best_permutation(self):
        best_acc = self.acc()
        for perm in permutations(range(self.n_hidden)):
            self.permute_ml_s(perm)
            acc = self.acc()
            if acc > best_acc:
                best_acc = acc
                best_perm = perm


    def update_init_s_distr(self):
        pass # Different for tree / forest


    def mother_xi_sum(self):
        pass # Different for tree / forest


    def update_P(self):        
        if self.P.ndim == 2:
            m_d_xi_sum = self.sum('m_d_xi')
            mother_xi_sum = self.mother_xi_sum()
            self.P = (m_d_xi_sum.T / mother_xi_sum).T
            if not np.allclose(np.sum(self.P, axis=1), 1):
                print(np.sum(self.P, axis=1))
                raise HMTError("Updating P matrix has gone wrong.")
            return
        
        # self.P.ndim == 3
        xi_f_sum = self.sum('xi_f')
        if not self.daughter_order:
            # Enforce symmetry
            xi_f_sum = (xi_f_sum + np.transpose(xi_f_sum, axes=(0, 2, 1))) / 2
            if not np.allclose(xi_f_sum, np.transpose(xi_f_sum, axes=(0, 2, 1))):
                raise HMTError("P not symmetric in update step")
        # Normalise
        self.P = (xi_f_sum.T / (self.sum('xi') - self.leaf_sum('xi'))).T
        # if self.it % 10 == 0:
        #     print((f"{self.sum('xi') = }"))
        #     print((f"{self.leaf_sum('xi') = }"))

        # Ensure P sums to 1
        if not np.allclose(np.sum(self.P, axis=(1, 2)), 1):
            print(f"Iteration {self.it}")
            print(np.sum(self.P, axis=(1, 2)))
            print((f"{self.sum('xi') = }"))
            print((f"{self.leaf_sum('xi') = }"))
            print(self.P)
            raise HMTError("Updating P matrix has gone wrong.")
    

    def Mstep(self):
        # Update root nodes hidden state distribution
        self.update_init_s_distr()
        if not np.isclose(np.sum(self.init_s_distr), 1):
            raise HMTError("Updating initial distribution has gone wrong.")

        # Update P
        self.update_P()
        
        # Update emission distribution parameters
        self.emission_distr.update_params(self)
    

    def find_missing_pattern(self):
        if self.find_null():
            self.has_null = True
            if self.has_death():
                self.death_missing_pattern()
            else:
                self.emission_distr.missing_pattern = {idxs: defaultdict() for idxs in self.null_indices()}

    def death_missing_pattern(self):
        for death_ind, distr in enumerate(self.emission_distr.distrs):
            if death_ind == 0:
                distr.missing_pattern = {idxs: defaultdict() for idxs in self.null_indices()}
                continue
            # Just search leaf nodes
            death_nodes = [node for node in self.leaves if node.d == death_ind]
            null_idxs = {
                tuple(
                    np.argwhere(np.isnan(node.x[:death_ind])).flatten()
                    ) for node in death_nodes if np.isnan(node.x[:death_ind]).any()
                }
            distr.missing_pattern = {idxs: defaultdict() for idxs in null_idxs}

    def train(self, n_starts=1, tol=1e-6, maxits=200, store_log_lks=False, permute=False, overwrite_params=False,
              store_params=None):

        if not overwrite_params:
            # NOTE ensure only the parameters you want are stored in the model - all others should be None
            start_params = self.get_params() 

        self.find_missing_pattern()

        best_loglk = -np.inf

        for _ in range(n_starts):
            self.it = 0
            self.clear_params()
            if not overwrite_params:
                self.set_params(**start_params)
            self.init_params()
            
            if store_params is not None:
                if isinstance(store_params, str):
                    stored_params = [self.get_params()[store_params]]
                elif isinstance(store_params, list):
                    stored_params = {key: [self.get_params()[key]] for key in store_params}

            curr_log_lk = self.Estep()
            self.Mstep()

            if store_params is not None:
                curr_params = self.get_params()
                if isinstance(store_params, str):
                    stored_params.append(curr_params[store_params])
                elif isinstance(store_params, list):
                    for key in store_params:
                        stored_params[key].append(curr_params[key])
            
            if store_log_lks:
                log_lks = np.zeros(maxits)
                log_lks[0] = curr_log_lk


            for self.it in range(1, maxits):
                prev_log_lk = curr_log_lk
                curr_log_lk = self.Estep()
                self.Mstep()
                if store_params is not None:
                    curr_params = self.get_params()
                    if isinstance(store_params, str):
                        stored_params.append(curr_params[store_params])
                    elif isinstance(store_params, list):
                        for key in store_params:
                            stored_params[key].append(curr_params[key])
                if store_log_lks:
                    log_lks[self.it] = curr_log_lk
                if abs((prev_log_lk - curr_log_lk) / prev_log_lk) < tol:
                    break
            
            if curr_log_lk > best_loglk:
                best_loglk = curr_log_lk
                best_params = self.get_params()
                best_it = self.it # Used to ensure best run actually converged
                if store_log_lks:
                    best_loglks = log_lks[:self.it]
                if store_params is not None:
                    best_stored_params = stored_params
        
        if best_it == maxits - 1:
            warnings.warn("Loglikelihood did not converge, try changing the tol and maxits arguments")
        
        self.set_params(**best_params)

        if permute:
            self.permute()

        if store_params is not None and store_log_lks:
            return best_it, best_loglks, best_stored_params
        if store_log_lks:
            return best_it, best_loglks
        if store_params is not None:
            return best_it, best_stored_params
        return best_it
    
    def has_death(self):
        death_inds = np.unique([leaf.d for leaf in self.leaves])
        return len(death_inds) > 1


class HMTree(HMModel, Tree):
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
    loglikelihood: float
        Loglikelihood of the optimal tree.
    optimal_tree_prob: float
        The probability of the hidden state tree given the observed data
    """
    def __init__(self, *hmm_args, **hmm_kwargs):
        super().__init__(*hmm_args, **hmm_kwargs)


    def __repr__(self):
        return f'HMTree(root={self.root})'  


    def __str__(self):
        if self.root is not None:
            self.root.printTree()
            return ''
        return "HMTree()"
    

    def Estep(self):
        # Handle Errors
        if self.init_s_distr is None:
            raise HMTError('Tree has no initial S distribution.')
        if self.P is None:
            raise HMTError('Tree has no transition matrix P.')
        if self.emission_distr is None:
            raise HMTError('Tree has no emission distribution.')
        
        self.loglikelihood = 0
        
        self.root.s_distr = 1.0 * self.init_s_distr
        if self.sister_dep:
            self.root.calculate_sd_s_distr(drec=True)
            self.root.sd_upward_pass(urec=True)
            self.root.sd_downward_pass(drec=True)
        else:
            self.root.calculate_s_distr(drec=True)
            self.root.upward_pass(urec=True)
            self.root.downward_pass(drec=True)
        
        return self.loglikelihood


    def update_init_s_distr(self):
        self.init_s_distr = self.root.xi.copy()
    

    def mother_xi_sum(self):
        return self.root.mother_xi_sum()
    

    def xi_null_sum(self, total=None):
        if total is None:
            total = np.zeros((self.n_obs, self.n_hidden))
        self.root.xi_null_sum(total)
        return total


    def clean(self):
        self.root.clean()

    def sample(self, N, p, root_id=1, has_death=False):
        s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
        if has_death:
            death, obs = self.emission_distr.sample(s)
        else:            
            obs = self.emission_distr.sample(s)
        if self.n_obs == 1:
            obs = float(obs)
        self.root = HMNode(root_id, observed=obs, hmmodel=self, s=s)
        if has_death:
            self.root.d = death
            self.leaves.append(self.root)
            if death is not None:
                return
        self.root.sample(N, p, has_death=has_death) # With death we will have fewer than N cells

    def sample_with_death(self, N, p, root_id=1):
        s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
        obs = self.emission_distr.sample(s)
        if self.n_obs == 1:
            obs = float(obs)
        self.root = HMNode(root_id, observed=obs, hmmodel=self, s=s)
        self.root.sample_with_death(N, p)
    

    """" ################# Input / Output ################# """
    

    def to_numpy(self, attrs=None):
        has_true_s = self.root.s is not None
        n_attrs = len(attrs) if attrs is not None else 0
        # Make an empty array with n_nodes rows, and the columns
        # [id, mother_id, x_0, x_1, ..., x_(n_obs), true_hidden_state (if known), attrs]
        X = np.zeros((len(self), 2 + self.n_obs + has_true_s + n_attrs), dtype=object)
        # dtype=object allows strings for the ids
        
        queue = [self.root]
        i = 0
        while queue:
            node = queue.pop(0)
            X[i, 0] = node.id
            X[i, 1] = node.mother.id if node.mother is not None else np.nan
            X[i, 2 : 2 + self.n_obs] = node.x
            if has_true_s:
                X[i, 2 + self.n_obs] = node.s
            if attrs is not None:
                for pos, attr in enumerate(attrs):
                    node_attr = getattr(node, attr)
                    X[i, 2 + self.n_obs + has_true_s + pos] = node_attr

            i += 1
            if node.d0 is not None:
                queue.append(node.d0)
            if node.d1 is not None:
                queue.append(node.d1)
        return X


    def to_pandas(self, attrs=None):
        ndarr = self.to_numpy(attrs=attrs)
        
        column_names = ['cell_id', 'mother_id'] + [f'x{i}' for i in range(self.n_obs)]
        if self.root.s is not None:
            column_names.append('s')
        
        if attrs is not None:
            column_names += attrs
        
        df = pd.DataFrame(
            data=ndarr,
            columns=column_names
        )
        if self.root.s is not None:
            df["s"] = df["s"].astype('Int64')
        if not isinstance(self.root.id, str):
            df[["cell_id", "mother_id"]] = df[["cell_id", "mother_id"]].astype('Int64')
        return df


    def to_csv(self, path, attrs=None, **kwargs):
        df = self.to_pandas(attrs=attrs)
        df.to_csv(path, index=False, **kwargs)
    

    @staticmethod
    def from_numpy(X, n_hidden, n_obs, has_true_s, obs_func=None):
        tree = HMTree(n_hidden, n_obs)
        # Assumes root is the first line in the dataset
        root_data = X[0]
        obs = root_data[2 : 2 + tree.n_obs].astype(float)
        if obs_func is not None:
            obs = obs_func(obs)

        # Initialise root
        tree.root = HMNode(
            node_id=root_data[0],
            observed=obs,
            hmmodel=tree,
            s=root_data[-1] if has_true_s else None
        )
        
        # Add daughters recursively
        tree.root.from_numpy(X=X, has_true_s=has_true_s, obs_func=obs_func)

        # Find leaf nodes
        tree.leaves = tree.root.get_leaves()
        return tree


    @staticmethod
    def from_pandas(df, n_hidden, n_obs, obs_func=None):
        has_true_s = "s" in df.columns.tolist()
        ndarr = df.to_numpy(na_value=np.nan)
        return HMTree.from_numpy(ndarr, n_hidden, n_obs, has_true_s, obs_func=obs_func)


    @staticmethod
    def from_csv(path, n_hidden, n_obs, obs_func=None, **csv_kwargs):
        df = pd.read_csv(path, **csv_kwargs)
        return HMTree.from_pandas(df, n_hidden, n_obs, obs_func=obs_func)
    
    """ ##### Accuracy + Permutations ##### """

    def extract_ml_s(self):
        if self.sister_dep:
            self.root.sd_extract_ml_s(d_drec=True)
        else:
            self.root.extract_ml_s(d_drec=True)


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


class HMForest(HMModel, Forest):
    """
    Class for a group of hidden Markov trees that can be trained as a group or
    individually. See HMTree for more details about training and [2] for more details
    about ensemble updates.

    References
    ----------
    [1]
    [2] - Me :)
    """
    def __init__(self, *hmm_args, **hmm_kwargs):
        super().__init__(*hmm_args, **hmm_kwargs)

    
    def Estep(self):
        # Handle Errors
        if self.init_s_distr is None:
            raise HMTError('Forest has no initial S distribution.')
        if self.P is None:
            raise HMTError('Forest has no transition matrix P.')
        if self.emission_distr is None:
            raise HMTError('Forest has no emission distribution.')
        
        self.loglikelihood = 0

        for tree_root in self.roots:    
            tree_root.s_distr = 1.0 * self.init_s_distr
            if self.sister_dep:
                tree_root.calculate_sd_s_distr(d_drec=True)
                tree_root.sd_upward_pass(d_urec=True)
                tree_root.sd_downward_pass(d_drec=True)
            else:
                tree_root.calculate_s_distr(d_drec=True)
                tree_root.upward_pass(d_urec=True)
                tree_root.downward_pass(d_drec=True)
        
        return self.loglikelihood


    def xi_null_sum(self):
        # Easier to get sum using n_obs x n_hidden
        xi_null_sum = np.zeros((self.n_obs, self.n_hidden))
        for root in self.roots:
            root.xi_null_sum(xi_null_sum)
        # Easier to work with n_hidden x n_obs from here
        return xi_null_sum.T 


    def update_init_s_distr(self):
        self.init_s_distr = np.mean([root.xi for root in self.roots], axis=0)
    

    def mother_xi_sum(self):
        return np.sum([root.mother_xi_sum() for root in self.roots], axis=0)
    

    def clean(self):
        for root in self.roots:
            root.clean()
    

    def sample(self, n_trees, n_nodes, dropout, has_death=False):
        self.roots = [] # Clear any current trees in the forest
        n_nodes_remaining = 0
        for i in range(n_trees):
            s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
            if has_death:
                death, obs = self.emission_distr.sample(s)
            else:
                obs = self.emission_distr.sample(s)
            if self.n_obs == 1:
                obs = float(obs)
            root = HMNode(f"t{i+1}-1", observed=obs, hmmodel=self, s=s)
            if has_death:
                root.d = death
                if death:
                    self.roots.append(root)
                    self.leaves.append(root)
                    n_nodes_remaining += n_nodes - 1
                    continue
            n_nodes_to_add = root.sample(n_nodes, dropout, has_death=has_death)
            # print(n_nodes_to_add)
            n_nodes_remaining += n_nodes_to_add
            self.roots.append(root)

        # Sample remaining nodes that weren't added to original trees due to death
        while n_nodes_remaining > 0:
            # TODO doesn't work when we go in to this while loop a second time
            # print(n_nodes_remaining)
            i += 1
            if i > 10:
                return
            s = np.random.choice(np.arange(self.n_hidden), p=self.init_s_distr)
            if has_death:
                death, obs = self.emission_distr.sample(s)
            else:
                obs = self.emission_distr.sample(s)
            if self.n_obs == 1:
                obs = float(obs)
            root = HMNode(f"t{i+1}-1", observed=obs, hmmodel=self, s=s)
            if has_death:
                root.d = death
                if death:
                    self.roots.append(root)
                    self.leaves.append(root)
                    n_nodes_remaining += n_nodes - 1
                    continue
            n_nodes_remaining = root.sample(n_nodes_remaining, dropout, has_death=has_death)
            self.roots.append(root)

    """ =================== Input / Output =================== """

    def to_numpy(self, attrs=None):
        has_true_s = self.roots[0].s is not None
        
        # Make an empty array with n_nodes rows, and the columns
        # [id, mother_id, x_0, x_1, ..., x_(n_obs), true_hidden_state (if known)]
        n_nodes = sum([len(tree) for tree in self.roots])
        n_attrs = len(attrs) if attrs is not None else 0
        X = np.zeros((n_nodes, 2 + self.n_obs + has_true_s + n_attrs), dtype=object)
        
        i = 0
        for root in self.roots:
            queue = [root]
            while queue:
                node = queue.pop(0)
                X[i, 0] = node.id
                X[i, 1] = node.mother.id if node.mother is not None else np.nan
                X[i, 2 : 2 + self.n_obs] = node.x
                if has_true_s:
                    X[i, 2 + self.n_obs] = node.s
                if attrs is not None:
                    for pos, attr in enumerate(attrs):
                        node_attr = getattr(node, attr)
                        X[i, 2 + self.n_obs + has_true_s + pos] = node_attr

                i += 1
                if node.d0 is not None:
                    queue.append(node.d0)
                if node.d1 is not None:
                    queue.append(node.d1)
        return X


    def to_pandas(self, attrs=None):
        ndarr = self.to_numpy(attrs=attrs)
        
        column_names = ['cell_id', 'mother_id'] + [f'x{i}' for i in range(self.n_obs)]
        if self.roots[0].s is not None:
            column_names.append('S')
        
        if attrs is not None:
            column_names += attrs
        
        df = pd.DataFrame(
            data=ndarr,
            columns=column_names
        )
        if self.roots[0].s is not None:
            df["S"] = df["S"].astype('Int64')
        if not isinstance(self.roots[0].id, str):
            df[["cell_id", "mother_id"]] = df[["cell_id", "mother_id"]].astype('Int64')
        return df


    def to_csv(self, path, attrs=None, **kwargs):
        df = self.to_pandas(attrs=attrs)
        df.to_csv(path, index=False, **kwargs)
    

    @staticmethod
    def from_numpy(X, n_hidden, n_obs, inds_dict, obs_func=None):
        """
        Assumes the array X is in the format:

        [tree1, root1, nan, obs, hidden (if known)]
        [tree1, cell, mother, obs, hidden (if known)]
        [tree1, cell, mother, obs, hidden (if known)]
        ...
        [tree2, root2, nan, obs, hidden (if known)]
        [tree2, cell, mother, obs, hidden (if known)]
        ...
        [tree3, root3, nan, obs, hidden (if known)]
        [tree3, cell, mother, obs, hidden (if known)]
        ...

        NOTE: The tree ID is not necesarily required
        """
        forest = HMForest(n_hidden, n_obs)

        if inds_dict['tree_id'] is not None:
            for tree_id in np.unique(X[:, inds_dict['tree_id']]):
                tree_X = X[X[:, 0] == tree_id, 1:]
                root_data = tree_X[0]
                obs = root_data[inds_dict['obs']].astype(float)
                if obs_func is not None:
                    obs = obs_func(obs)
                if inds_dict['C'] is not None:
                    censoring_inds = eval(inds_dict['C'])
                    if censoring_inds:
                        c = obs[censoring_inds]
                        obs[censoring_inds] = np.nan

                # Initialise root
                root = HMNode(
                    node_id=root_data[0],
                    observed=obs,
                    hmmodel=forest,
                    hidden_state=root_data[inds_dict['hidden_state']] if inds_dict['hidden_state'] is not None else None,
                    d=root_data[inds_dict['D']] if inds_dict['D'] is not None else None
                )
                return root
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
    
    def add_nodes_from_pandas(self, df, obs_cols, obs_func=None):
        for tree_id in df["Tree ID"].unique():
            tree_df = df[df["Tree ID"] == tree_id].reset_index(drop=True)
            ## NOTE assumes the root is the first row in the dataframe

            # Initialise root and read in tree
            root = HMNode.from_pandas_row(tree_df.iloc[0], obs_cols=obs_cols, hmmodel=self, obs_func=obs_func)
            root.from_pandas(df=tree_df, obs_cols=obs_cols, obs_func=obs_func)
            self.roots.append(root)
            self.leaves += root.get_leaves()

    @staticmethod
    def from_pandas(df, n_hidden, n_obs, obs_cols, obs_func=None):
        cols = df.columns.tolist()
        assert set(obs_cols).issubset(set(cols)), "obs_cols must be a subset of the dataframe columns"
        
        forest = HMForest(n_hidden, n_obs)
        forest.add_nodes_from_pandas(df=df, obs_cols=obs_cols, obs_func=obs_func)
        return forest


    @staticmethod
    def from_csv(path, n_hidden, n_obs, obs_cols, obs_func=None, **csv_kwargs):
        df = pd.read_csv(path, **csv_kwargs)
        return HMForest.from_pandas(df=df, n_hidden=n_hidden, n_obs=n_obs, obs_cols=obs_cols, obs_func=obs_func)


    """ ##### Accuracy + Permutations ##### """

    def extract_ml_s(self):
        if self.sister_dep:
            for root in self.roots:
                root.sd_extract_ml_s(d_drec=True)
        else:
            for root in self.roots:
                root.extract_ml_s(d_drec=True)

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
        self.extract_ml_s()

        if self.roots[0].s is not None:
            # Permute
            best_acc = self.permute()
            return best_acc
    
    
    def acc(self):
        n_acc = np.sum([root.n_accurate_nodes() for root in self.roots], axis=0)
        n_tot = len(self)
        return n_acc / n_tot


    def best_permutation(self):
        """
        Calculates best permutation of *true* values of hidden states
        """
        # Calculate smoothed probabilities
        self.Estep()
        # Extract MLE S values
        self.extract_ml_s()


        best_acc = self.acc()
        best_perm = None
        for perm in permutations(range(self.n_hidden)):
            # Apply permutation
            for root in self.roots:
                root.apply(lambda s: perm[s], 's', d_drec=True)
            
            # Find accuracy with this permutation
            acc = self.acc()
            
            # Apply inverse permutation
            for root in self.roots:
                root.apply(lambda s: np.argsort(perm)[s], 's', d_drec=True)
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
        return best_acc, best_perm


    def sample_corr(self, measure="true_s"):
        if measure != "true_s" and measure != "ml_s":
            raise HMTError("measure can be 'true_s' or 'ml_s'")
        s_list = []

        for root in self.roots:
            root.sample_corr(s_list, measure)

        # TODO reimplement with to_numpy method

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