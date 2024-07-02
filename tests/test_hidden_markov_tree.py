"""Tests for the hidden_markov_tree module."""


def test_hmt_tree():
    """Test the HMTTree class."""
    from hmt import HMTree

    n_hidden = 3
    n_obs = 4

    tree = HMTree(n_hidden, n_obs, sister_dep=False)

    assert tree.n_hidden == n_hidden
    assert tree.n_obs == n_obs
    assert tree.sister_dep is False
