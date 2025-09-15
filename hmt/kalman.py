import numpy as np
from hmt.core import Node, Tree, Forest
from hmt.exceptions import HMTError, HMTWarning


class KalmanNode(Node):
    P = None
    res = None
    s = None
    def __init__(self, node_id, observed, true_s=None, tree=None):
        super().__init__(node_id, observed, tree)
        self.true_s = true_s


    def update(self, drec=True):
        # Prefit residual
        # y = x - Hs
        y = self.x - self.tree.H @ self.s 

        # S = H P_(k|k-1) H^T + R
        print(self.mother.P.shape, self.tree.H.shape)
        S = self.tree.H @ self.mother.P @ self.tree.H.T #+ self.tree.R

        # K = P_(k|k-1) H^T S^-1
        if isinstance(S, np.ndarray):
            K = self.mother.P @ self.tree.H @ np.linalg.inv(S)
        else:
            K = self.mother.P @ self.tree.H / S

        ## Updated state estimate
        # x = x + Ky
        self.s = self.s + K @ y

        # P_(k|k) = (I - KH) P_(k|k-1)
        IminusKH = - K @ self.tree.H
        for i in IminusKH.shape[0]:
            IminusKH[i, i] += 1
        self.P = IminusKH @ self.mother.P

        # Post-fit residual
        self.res = self.x - self.tree.H @ self.s 

        if drec:
            self.predict_daughters(drec)


    def predict_daughters(self, drec=True):
        if self.d0 is None and self.d1 is None:
            return # Stop recursion

        ## State estimate
        # s = Fs + Bu
        s = self.tree.F @ self.s #+ self.tree.B @ self.u

        ## Update daughter values
        if self.d0 is not None:
            self.d0.s = s[: self.tree.n // 2 + 1]
        if self.d1 is not None:
            self.d1.s = s[self.tree.n // 2 + 1:]
        
        ## Covariance estimate
        # P_(k+1|k) = F P_(k|k) F^T + Q
        self.P = self.tree.F @ self.P @ self.tree.F.T + self.tree.Q

        # Continue recursion 
        if drec:
            if self.d0 is not None:
                self.d0.update(drec)
            if self.d1 is not None:
                self.d1.update(drec)


class KalmanTree(Tree):
    F = None
    Q = None
    H = None
    n = None
    def __init__(self, root=None, model_params=None):
        super().__init__(root)
        if model_params is not None:
            self.set_model_params(**model_params)
    

    def check_model_params(self, F, Q, H):
        if Q.shape[0] != Q.shape[1]:
            raise HMTError('Q must be square.')
        if Q.shape[0] != F.shape[0]:
            raise HMTError('Q must have the same number of rows as F.')
        


    def set_model_params(self, F, Q, H):
        self.check_model_params(F, Q, H)
        self.F = F
        self.Q = Q
        self.H = H
        self.n = F.shape[1]

    
    def run(self, s0, P0):
        if self.H.shape[self.H.ndim - 1] != s0.shape[0]:
            raise HMTError('The shapes of H and x0 must match for matrix multiplication.')
        
        # Set initial state and covariance matrix
        self.root.s = s0
        self.root.P = P0
        
        # Run model
        self.root.predict_daughters(drec=True)


class KalmanForest(Forest):
    def __init__(self):
        super().__init__()
        self.TreeClass = KalmanTree
        self.NodeClass = KalmanNode