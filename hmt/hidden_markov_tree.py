import numpy as np
import pandas as pd
from timeit import default_timer
from itertools import permutations
from collections import defaultdict
import warnings

from hmt.core import Node, Forest
from hmt.exceptions import HMTError
from hmt.utils import div0, normal_round, digamma, log_addition
import hmt.emissions as emissions

class HMModel:
    pass

class HMNode(Node):
    def __init__(self, node_id, observed=None, hmmodel=None, d=None, c=None, t=None, hidden_state=None):
        super().__init__(node_id, observed)
        self.hmmodel = hmmodel
        
        # Switches during cell life
        self.next, self.prev = None, None

        self.s, self.c, self.d, self.t = hidden_state, c, d, t
        self.ml_s = None

        self.s_distr = None
        self.xi = None

        # No sister dependence
        self.beta = None
        self.m_d_beta = None
        self.m_d_delta = None
        self.m_d_xi = None
    
    def __repr__(self):
        return f'HMNode({self.id})'

    def __eq__(self, other):
        for attr in ['id', 'd', 's', 'ml_s', 'next', 'prev']:
            try:
                if np.isnan(getattr(self, attr)):
                    if not np.isnan(getattr(other, attr)):
                        print(f"{attr} nan")
                        return False
                    continue
            except TypeError:
                continue
            if getattr(self, attr) != getattr(other, attr):
                print(f"{attr} not equal")
                return False
        for npattr in ['x', 'c', 't']:
            if getattr(self, npattr) is None:
                if getattr(other, npattr) is not None:
                    print(f"{npattr} None")
                    return False
                continue
            if np.all(np.isnan(getattr(self, npattr))):
                if not np.all(np.isnan(getattr(other, npattr))):
                    return False
                continue
            if not np.allclose(getattr(self, npattr), getattr(other, npattr)):
                print(f"{npattr} not equal")
                return False
        return True

    def clean(self):
        """Resets calculations on tree."""
        self.ml_s = None
        self.s_distr = None
        self.xi = None
        self.beta = None
        self.m_d_beta = None
        self.m_d_delta = None
        self.m_d_xi = None
    
    def calculate_s_distr(self):
        """Performs calculations on a single node for initial downward pass for hidden state prior."""
        if self.mother is None and self.s_distr is None:
            raise HMTError('Root node has no initial S distribution.')
        
        sc_distr = self.s_distr @ self.hmmodel.P
        if not np.isclose(sum(sc_distr), 1):
            raise HMTError('S distribution probabilities do not sum to 1.')

        if self.d0 is not None:
            self.d0.s_distr = sc_distr
        if self.d1 is not None:
            self.d1.s_distr = sc_distr

    def upward_pass(self):
        """Performs calculations on a single node for the upward pass."""
        self.beta = self.hmmodel.emission(self) * self.s_distr

        self.beta *= self.d0.m_d_beta if self.d0 is not None else 1
        self.beta *= self.d1.m_d_beta if self.d1 is not None else 1
        
        N = np.sum(self.beta)
        self.hmmodel.loglikelihood += np.log(N)

        self.beta = self.beta / N
        if not np.isclose(sum(self.beta), 1):
            raise HMTError("Probabilities in beta don't sum to 1.")

        self.m_d_beta = self.hmmodel.P @ div0(self.beta, self.s_distr)

    def downward_pass(self):
        """Performs calculations on a single node for the downward pass."""
        if self.mother is None:
            self.xi = self.beta
        else:
            row_mult = div0(self.beta, self.s_distr)
            col_mult = div0(self.mother.xi, self.m_d_beta)
            self.m_d_xi = self.hmmodel.P * row_mult
            self.m_d_xi = (self.m_d_xi.T * col_mult).T
            self.xi = self.m_d_xi.sum(axis=0)

        if not np.isclose(np.sum(self.xi), 1):
            raise HMTError("Smoothed probabilities do not sum to 1.")
        
        if self.mother is not None and not np.isclose(np.sum(self.m_d_xi), 1):
            raise HMTError("Smoothed probabilities do not sum to 1.")

    def viterbi(self):
        """Performs calculations on a single node for the upward Viterbi pass."""
        delta = self.hmmodel.emission(self)
        delta *= self.d0.m_d_delta if self.d0 is not None else 1
        delta *= self.d1.m_d_delta if self.d1 is not None else 1
        self.m_d_delta = np.amax(self.hmmodel.P.T * delta, axis=0)
        return delta

    def predict_viterbi(self):
        """Performs calculations on a single node for the downward Viterbi pass."""
        raise NotImplementedError("Viterbi not yet implemented for independent sister cells")

    def predict_greedy(self):
        """Predicts the most likely state of each node given the most likely mother state."""
        if self.mother is None:
            self.ml_s = np.argmax(self.xi)
        else:
            self.ml_s = np.argmax(self.m_d_xi[self.mother.ml_s, :])
    
    def sample_daughter(self, hidden_state, d=0, has_death=False):
        """Samples daughter node given hidden state, and left (d=0) or right (d=1) position."""
        # Daughter ID
        if isinstance(self.id, str):
            tree, m_id = self.id.split('-')
            node_id = '-'.join((tree, str(int(m_id) * 2 + d)))
        else:
            node_id = self.id * 2 + d

        NodeClass = SDHMNode if self.hmmodel.sister_dep else HMNode
        node = NodeClass(node_id, hmmodel=self.hmmodel, hidden_state=hidden_state)

        # Sample observation
        node.x = self.hmmodel.emission_distr.sample(hidden_state)
        if has_death:
            node.d, node.x = node.x
        node.x = float(node.x) if self.hmmodel.n_obs == 1 else node.x
        
        # Attach to tree
        node.mother = self
        setattr(self, f"d{d}", node)
        node._path = self._path + str(d)
        return node
    
    def sample(self, N, p, has_death=False):
        """
        Recursively samples nodes on a tree by dividing them into left and right subtrees.
        Args
        ----
        N : int
            The number of nodes to sample in the current subtree.
        p : float
            The probability of a node having only one daughter (in this case it is 50/50 whether
            the daughter is on the left or right.)
        has_death : bool, optional
            Flag indicating whether cell death events should be considered during sampling.
            Default is False.
        Returns
        -------
        int
            The number of remaining cells that could not be sampled (typically due to
            cell death events). Returns 0 if all cells in the subtree were successfully sampled.
        """
        if N == 1:
            return 0

        # Randomly choose how many children node to go on either side (N0 on left and N1 on right)
        q0 = np.random.choice((0, 0.5, 1), p=(p/2, 1 - p, p/2))
        N0 = normal_round(q0 * (N - 1))
        N1 = N - 1 - N0
        
        # Sample hidden states of daughters
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
            self.sample_daughter(s0, 0, has_death)
            if self.d0.d is not None and self.d0.d: # Cell dies, stop sampling
                remaining_0 += N0 - 1
            else:
                remaining_0 += self.d0.sample(N0, p, has_death=has_death)
        if N1 != 0:
            self.sample_daughter(s1, 1, has_death)
            if self.d1.d is not None and self.d1.d: # Cell dies, stop sampling
                remaining_1 += N1 - 1
            else:
                remaining_1 += self.d1.sample(N1, p, has_death=has_death)
        return remaining_0 + remaining_1 # Return number of remaining cells to sample


class SDHMNode(HMNode):
    def __init__(self, node_id, observed=None, hmmodel=None, d=None, c=None, t=None, hidden_state=None):
        super().__init__(node_id, observed, hmmodel, d, c, t, hidden_state)
        
        self.sc_distr = None
        self.beta_c = None
        self.beta_c_u = None
        self.xi_c = None
        self.xi_f = None
    
    def clean(self):
        """Resets calculations on tree."""
        self.ml_s = None
        self.s_distr = None
        self.xi = None
        self.sc_distr = None
        self.beta_c = None
        self.beta_c_u = None
        self.xi_c = None
        self.xi_f = None
    
    def __repr__(self):
        return f'SDHMNode({self.id})'
    
    def calculate_s_distr(self):
        """Performs calculations on a single node for initial downward pass for hidden state prior."""
        if self.mother is None:
            self.s_distr = self.hmmodel.init_s_distr.copy()
        
        if not np.isclose(np.sum(self.s_distr), 1):
            raise HMTError('S distribution probabilities do not sum to 1.')

        if self.next is not None:
            self.hmmodel.calculate_s_distr_next(self)
            return

        # Calculate children distribution
        self.sc_distr = (self.hmmodel.P.T @ self.s_distr).T

        if not np.isclose(np.sum(self.sc_distr), 1):
            raise HMTError('S children distribution probabilities do not sum to 1.')
        
        # Calculate s distribution of daughter cells
        if self.d0 is not None:
            self.d0.s_distr = np.sum(self.sc_distr, axis=1)
        if self.d1 is not None:
            self.d1.s_distr = np.sum(self.sc_distr, axis=0)

    def upward_pass(self):
        """Performs calculations on a single node for the upward pass."""
        if self.next is not None:
            if self.mother is None:
                raise NotImplementedError("Root node with changes not implemented")
            self.hmmodel.upward_pass_next(self)
            return
        
        if self.d0 is None and self.d1 is None:
            self.beta_c_u = 1
            return

        self.beta_c = self.sc_distr.copy()
        if self.d0 is not None: # Multiply columns
            self.beta_c = (self.beta_c.T * self.hmmodel.emission(self.d0) * self.d0.beta_c_u).T
        if self.d1 is not None: # Multiply rows
            self.beta_c *= self.hmmodel.emission(self.d1) * self.d1.beta_c_u 
        N = np.sum(self.beta_c)
        if N == 0:
            raise HMTError("Sum of beta_c is 0")
        self.hmmodel.loglikelihood += np.log(N)
        self.beta_c = self.beta_c / N

        self.beta_c_u = self.hmmodel.P * div0(self.beta_c, self.sc_distr)
        self.beta_c_u = np.sum(self.beta_c_u, axis=(1, 2))
        assert self.beta_c_u.shape == (self.hmmodel.n_hidden, )
        
        if self.mother is None:
            self.hmmodel.loglikelihood += np.log(np.dot(
                self.hmmodel.emission(self),
                self.s_distr * self.beta_c_u
                ))

    def downward_pass(self):
        """Performs calculations on a single node for the downward pass."""
        if self.next is not None:
            self.hmmodel.downward_pass_next(self)

        if self.d0 is None and self.d1 is None:
            if self.mother is None: # Root node with no daughters
                self.xi = self.hmmodel.emission(self)
                self.xi /= self.xi.sum()
            # else self.xi is calculated by mother node
            return
        
        if self.mother is None and self.prev is None:
            # Root node
            self.xi_f = div0(self.beta_c, self.sc_distr) * self.hmmodel.P
            self.xi_f = (self.xi_f.T *  self.hmmodel.emission(self)).T
            self.xi_f /= np.sum(self.xi_f)

            self.xi = np.sum(self.xi_f, axis=(1, 2))
        else:
            self.xi_f = (self.hmmodel.P.T * div0(self.xi, self.beta_c_u)).T
            self.xi_f *= div0(self.beta_c, self.sc_distr)
    
        self.xi_c = np.sum(self.xi_f, axis=0)
        
        if not np.isclose(np.sum(self.xi), 1):
            raise HMTError("Smoothed probabilities do not sum to 1")

        if not np.isclose(np.sum(self.xi_c), 1):
            raise HMTError("Smoothed children probabilities do not sum to 1")
        
        ## Calculate xi values of daughter cells
        if self.d0 is not None: # Sum over k
            self.d0.xi = np.sum(self.xi_c, axis=1)
            assert self.d0.xi.shape == (self.d0.hmmodel.n_hidden,), f"{self.hmmodel.n_hidden, self.d0.hmmodel.n_hidden}"
        if self.d1 is not None: # Sum over j
            self.d1.xi = np.sum(self.xi_c, axis=0)
            assert self.d1.xi.shape == (self.d1.hmmodel.n_hidden,), f"{self.hmmodel.n_hidden, self.d1.hmmodel.n_hidden}"

    def viterbi(self):
        """Performs calculations on a single node for the upward Viterbi pass."""
        if self.next is not None:
            self.hmmodel.sd_viterbi_next(self)
            return

        self.delta = self.hmmodel.emission(self)

        if self.d0 is None and self.d1 is None:
            return
        
        # Store the hidden states of the daughters that are optimal for each possible hidden state
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

        # For each state i in the current node we find the daughter states that 
        # would maximise the probability of the tree from here
        self.optimal_daughter_states = np.array([
            np.unravel_index(_delta[i].argmax(), _delta[i].shape) for i in range(self.hmmodel.n_hidden)
            ])

        max_delta = np.zeros(self.hmmodel.n_hidden)
        # TODO vectorise this with indexing
        for i in range(self.hmmodel.n_hidden):
            if self.d0 is not None and self.d1 is not None:
                opt_d0_state, opt_d1_state = self.optimal_daughter_states[i][0], self.optimal_daughter_states[i][1]
                max_delta[i] = _delta[i, opt_d0_state, opt_d1_state]
                continue
            if self.d0 is not None:
                max_delta[i] = _delta[i, self.optimal_daughter_states[i][0]]
            else:
                max_delta[i] = _delta[i, self.optimal_daughter_states[i][1]]

        self.delta *= max_delta

    def predict_viterbi(self):
        """Performs calculations on a single node for the downward Viterbi pass."""
        if self.delta is None:
            raise HMTError("Delta probabilities have not been calculated, call hmmodel.viterbi() first")
        if self.mother is None: # Root node
            self.ml_s = np.argmax(np.multiply(self.delta, self.s_distr))
        
        if self.next is not None:
            self.hmmodel.sd_viterbi_extract_ml_s_next(self)
            return 
        if self.d0 is None and self.d1 is None:
            return

        if self.d0 is not None:
            self.d0.ml_s = self.optimal_daughter_states[self.ml_s][0]
        if self.d1 is not None:
            self.d1.ml_s = self.optimal_daughter_states[self.ml_s][1]

    def predict_greedy(self):
        """Predicts the most likely state of each node given the most likely mother state."""
        if self.next is not None:
            self.hmmodel.predict_greedy_next(self)
            return
        
        if self.mother is None and self.prev is None:
            self.ml_s = np.argmax(self.xi)
        
        if self.d0 is None and self.d1 is None:
            return

        ml_sc = np.unravel_index(np.argmax(self.xi_f[self.ml_s]), self.xi_c.shape)

        if self.d0 is not None:
            self.d0.ml_s = ml_sc[0]
        if self.d1 is not None:
            self.d1.ml_s = ml_sc[1]

    # def sample(self):
    #     raise NotImplementedError("TODO: sample")

    def split(self):
        raise NotImplementedError("TODO: split")

    def from_numpy(self):
        raise NotImplementedError("TODO: from_numpy")

    def from_pandas_row(self):
        raise NotImplementedError("TODO: from_pandas_row")    


class HMForest(Forest):
    def __init__(
            self,
            n_hidden,
            n_obs,
            sister_dep=True,
            daughter_order=False,
            emission_distr="MVN",
            seed=0,
            **emission_kwargs
            ):
        super().__init__()

        # Hyper parameters
        self._n_hidden = n_hidden
        self._n_obs = n_obs
        self.sister_dep = sister_dep
        self.daughter_order = daughter_order

        # Model parameters
        self.init_s_distr = None
        self.P = None
        self.set_emissions(emission_distr, **emission_kwargs)

        self.rng = np.random.default_rng(seed)

        self.loglikelihood = 0
    
    @property
    def n_hidden(self):
        return self._n_hidden
    
    @n_hidden.setter
    def n_hidden(self, value):
        if not isinstance(value, int):
            raise HMTError("Number of hidden states must be an integer.")
        if value < 1:
            raise HMTError("Number of hidden states must be greater than 0.")
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
            self.init_s_distr = self.rng.uniform(0, 1, self.n_hidden)
            self.init_s_distr /= np.sum(self.init_s_distr)

        if self.P is None: # Randomly initialise transistion matrix
            if self.sister_dep:
                self.P = self.rng.uniform(0, 1, (self.n_hidden, self.n_hidden, self.n_hidden))
                # Normalise for each value of mother hidden state
                self.P = (self.P.T / np.sum(self.P, axis=(1, 2))).T
                if not self.daughter_order: # enforce P[i, j, k] = P[i, k, j]
                    self.P = (self.P + np.transpose(self.P, axes=(0, 2, 1))) / 2
            else:
                self.P = self.rng.uniform(0, 1, (self.n_hidden, self.n_hidden))
                # Normalise for each value of mother hidden state
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
            if self.sister_dep:
                if self.P.shape[0] != self.P.shape[1] or self.P.shape[0] != self.P.shape[2]:
                    raise HMTError('P must be a cube.')
                if not np.allclose(self.P.sum(axis=(1, 2)), 1):
                    raise HMTError('Matrices in P must sum to 1.')  
            else:
                if not np.allclose(self.P.sum(axis=1), 1):
                    raise HMTError('Rows of P must sum to 1.')
                if self.P.shape[0] != self.P.shape[1]:
                    raise HMTError('P must be a square matrix.')
                if self.P.shape[0] != self.init_s_distr.shape[0]:
                    raise HMTError('P must have the number of columns as init_s_distr.')
        
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
        
        self.emission_distr.set_params(**emission_params, check_params=False)
        
        if check_params:
            self.check_params()
    
    def get_params(self):
        return {
            'init_s_distr': self.init_s_distr.copy() if self.init_s_distr is not None else None,
            'P': self.P.copy() if self.P is not None else None,
            **self.emission_distr.get_params()
        }
    
    def clear_params(self):
        self.init_s_distr = None
        self.P = None
        self.emission_distr.clear_params()

    def clean(self):
        for node in self.preorder():
            node.clean()
    
    def set_emissions(self, emission_distr="MVN", **emission_kwargs):
        """NOTE I think emission_distr should be a property"""
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
        """Finds the emission probability of that node, overwritten in MultiEnv models"""
        return self.emission_distr.emission(node)
    
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
    
    def Estep(self):
        if self.init_s_distr is None:
            raise HMTError('Forest has no initial S distribution.')
        if self.P is None:
            raise HMTError('Forest has no transition matrix P.')
        if self.emission_distr is None:
            raise HMTError('Forest has no emission distribution.')
    
        self.loglikelihood = 0

        # Upward-downward algorithm
        for node in self.preorder():
            node.calculate_s_distr()
        for node in self.postorder():
            node.upward_pass()
        for node in self.preorder():
            node.downward_pass()
    
        return self.loglikelihood

    def Mstep(self):
        self.update_init_s_distr()
        self.update_P()
        self.emission_distr.update_params()
    
    def update_init_s_distr(self):
        self.init_s_distr = np.mean([root.xi for root in self.roots], axis=0)
        if not np.isclose(np.sum(self.init_s_distr), 1):
            raise HMTError("Updating initial distribution has gone wrong.")
    
    def update_P(self):        
        if not self.sister_dep:
            m_d_xi_sum = self.sum('m_d_xi')
            mother_xi_sum = self.mother_xi_sum()
            self.P = (m_d_xi_sum.T / mother_xi_sum).T
            if not np.allclose(np.sum(self.P, axis=1), 1):
                raise HMTError("Updating P matrix has gone wrong.")
            return
        xi_f_sum = self.sum('xi_f')
        if not self.daughter_order:
            xi_f_sum = (xi_f_sum + np.transpose(xi_f_sum, axes=(0, 2, 1))) / 2
            if not np.allclose(xi_f_sum, np.transpose(xi_f_sum, axes=(0, 2, 1))):
                raise HMTError("P not symmetric in update step")
        self.P = (xi_f_sum.T / (self.sum('xi') - self.leaf_sum('xi'))).T
        
        if not np.allclose(np.sum(self.P, axis=(1, 2)), 1):
            raise HMTError("Updating P matrix has gone wrong.")

    def mother_xi_sum(self):
        """Sum the xis of nodes that are mothers."""
        return sum(node.xi for node in self.preorder() if node.d0 is not None or node.d1 is not None)
    
    def xi_null_sum(self):
        xi = np.array([node.xi for node in self.preorder()])
        return np.nansum(xi, axis=0)

    def find_missing_pattern(self):
        # TODO move this to emissions, it makes way more sense there, and can be overwritten by death emissions below
        if self.has_death():
            self.death_missing_pattern()
        if not self.has_null():
            return
        self.emission_distr.missing_pattern = {idxs: defaultdict() for idxs in self.null_indices()}

    def death_missing_pattern(self):
        #TODO if self.shared_parameters
        for death_ind, distr in enumerate(self.emission_distr.distrs):
            if death_ind == 0:
                death_nodes = [node for node in self.preorder() if node.d == 0 ]
                end = None
            else:
                death_nodes = [node for node in self.leaves if node.d == death_ind]
                end = death_ind
            null_idxs = {
                tuple(
                    np.argwhere(np.isnan(node.x[:end])).flatten()
                    ) for node in death_nodes if np.isnan(node.x[:end]).any()
                }
            distr.missing_pattern = {idxs: defaultdict() for idxs in null_idxs}
    
    def has_death(self):
        return bool(len([leaf.d for leaf in self.leaves if leaf.d is not None and leaf.d]))

    def predict(self, method="viterbi"):
        if method == "viterbi":
            self.predict_viterbi()
        elif method == "greedy":
            self.predict_greedy()
        else:
            raise HMTError("Only 'viterbi' and 'greedy' are valid methods")
    
    def predict_greedy(self):
        for node in self.preorder():
            node.predict_greedy()
        return [node.ml_s for node in self.preorder()]
    
    def predict_viterbi(self):
        for node in self.postorder():
            node.viterbi()
        for node in self.preorder():
            node.predict_viterbi()
        return [node.ml_s for node in self.preorder()]

    def accuracy(self):
        if self.roots[0].s is None:
            raise HMTError("Cannot calculate accuracy without true values of s.")
        return np.mean([node.s == node.ml_s for node in self.preorder()])

    def acc(self):
        return self.accuracy()
    
    def permute(self, perm):
        """Permutes all hidden states and respective parameters"""
        if self.roots[0].ml_s is not None:
            for node in self.preorder():
                node.ml_s = perm[node.ml_s]

        self.init_s_distr = self.init_s_distr[perm, ...]

        # Permute P (permute in all dimensions in order)
        if self.sister_dep:
            self.P = self.P[perm, ...][:, perm, ...][:, :, perm]
        else:
            self.P = self.P[perm, ...][:, perm, ...]

        # Permute emission parameters
        self.emission_distr.permute(perm)
    
    def best_permutation(self):
        best_acc = self.accuracy()
        true_s = np.array([node.s for node in self.preorder()])
        pred_s = np.array([node.ml_s for node in self.preorder()])
        for perm in permutations(range(self.n_hidden)):
            perm_s = pred_s[perm]
            acc = np.mean(true_s == perm_s)
            if acc > best_acc:
                best_acc = acc
                best_perm = perm
        self.permute(best_perm)
        assert best_acc == self.accuracy()
        return best_acc()

    def sample_corr(self, true_s=False):
        s_list = [node.s if true_s else node.ml_s for node in self.preorder()]
        s,r, n = np.array(s_list), np.zeros(self.n_hidden), np.zeros(self.n_hidden, dtype=int)
        for i in range(self.n_hidden):
            # Filter so that mother node is in state i
            si = s[s[:, 0] == i]
            n[i] = si.shape[0]
            _, d0_mean, d1_mean = np.mean(si, axis=0)
            
            # Sample standard deviation
            d0_std = np.sqrt(np.dot(si[:, 1] - d0_mean, si[:, 1] - d0_mean) / (n[i] - 1))
            d1_std = np.sqrt(np.dot(si[:, 2] - d1_mean, si[:, 2] - d1_mean) / (n[i] - 1))
            r[i] = np.dot(si[:, 1], si[:, 2]) - n[i] * d0_mean * d1_mean
            if np.isclose(d0_std, 0) or np.isclose(d1_std, 0):
                r[i] = 0
            else:
                r[i] /= (n[i] - 1) * d0_std * d1_std

        return r, n

    def sample(self, n_trees, n_nodes, dropout, has_death=False, keep_sampling=False):
        if dropout == 1:
            raise HMTError("Dropout rate cannot be 100%")
        NodeClass = SDHMNode if self.sister_dep else HMNode
        self.roots = [] # Clear any current trees in the forest
        n_nodes_remaining = 0
        for i in range(n_trees):
            hidden_state = self.rng.choice(np.arange(self.n_hidden), p=self.init_s_distr)
            root = NodeClass(f"t{i+1}-1", hmmodel=self, hidden_state=hidden_state)
            root.x = self.emission_distr.sample(hidden_state)
            if has_death:
                root.d, root.x = root.x
            root.x = float(root.x) if self.n_obs == 1 else root.x
            if root.d is not None and root.d: # Cell dies, stop sampling
                n_nodes_to_add = n_nodes - 1
            else:
                n_nodes_to_add = root.sample(n_nodes, dropout, has_death=has_death)
            n_nodes_remaining += n_nodes_to_add
            self.roots.append(root)
        self.get_leaves()
    
    def to_numpy(self, return_cols=False, combine_t=False, include_ids=True):
        columns = []
        for attr in ['d', 's', 'ml_s']:
            if len(self.where(attr, lambda x: x is not None)):
                columns.append(attr)
        
        forest = []
        for tree_id, root in enumerate(self.roots):
            tree_id += 1
            for node in root.preorder():
                mother_id = np.nan if node.mother is None else node.mother.id
                if combine_t and node.t is not None:
                    x = node.x if np.all(node.t == -np.inf) else log_addition(node.x, node.t)
                else:
                    x = node.x
                if include_ids:
                    data = [tree_id, node.id, mother_id] + list(x) + [getattr(node, attr) for attr in columns]
                else:
                    data = list(x) + [getattr(node, attr) for attr in columns]
                forest.append(data)
    
        if return_cols:
            return np.array(forest), columns
        return np.array(forest)