"""The main module for Hidden Markov Trees."""

from importlib.metadata import version

from .core import HMTError, HMTWarning, Node, Tree, read_txt
from .hidden_markov_tree import HMForest, HMNode, HMTree

# from .kalman import KalmanNode, KalmanTree, KalmanForest

__version__ = version(__name__)
__all__ = [
    "HMTError",
    "HMTWarning",
    "Node",
    "Tree",
    "read_txt",
    "HMForest",
    "HMNode",
    "HMTree",
]
