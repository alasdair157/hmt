"""The main module for Hidden Markov Trees."""

from importlib.metadata import version

from .core import Tree, Node, HMTError, HMTWarning, read_txt
from .hidden_markov_tree import HMNode, HMTree, HMForest
# from .kalman import KalmanNode, KalmanTree, KalmanForest

__version__ = version(__name__)
