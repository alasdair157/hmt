"""
Exension of numpy to do useful things
"""
import numpy as np


def insort(a, x):
    """
    Insert item x into list a, and keep it sorted assuming a is sorted.
    """
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    a.insert(lo, x)


def normal_round(x):
    if x - np.floor(x) < 0.5:
        return np.floor(x)
    return np.ceil(x)


def div0(a, b):
    """A quick function that divides two numpy arrays element wise and returns 0 when dividing by 0."""
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def log0(x):
    return np.log(x, out=np.zeros_like(x), where=x!=0)


def rowwise_outer(a, b):
    """
    Returns the row-wise outer product of a and b both of size rxn
    equivalent to:
    for i in range(a.shape[0])
        c[i] = np.outer(a[i], b[i])
    """
    return np.einsum('ij, ik -> ijk', a, b)


def mvn_pdf(x, mu, sigmainv, detsigma, k):
    """
    Returns the probabilities of obtaining x from different mvn distributions
    """
    centered_x = x - mu
    exponent = np.diag(np.diagonal(centered_x @ sigmainv @ centered_x.T))
    
    res = np.exp(-exponent / 2) / np.sqrt(detsigma * (2 * np.pi) ** k)
    # if res.dtype == 'complex128':
    #     print(detsigma)
    return res


def mutual_information(P):
    """Calculates the mutual information between """
    Id0d1 = np.zeros(P.shape[0])

    for i, Pi in enumerate(P):
        Pi0 = Pi.sum(axis=1)
        Pi1 = Pi.sum(axis=0)
        p = div0(Pi.T, Pi0).T
        p = div0(p, Pi1)
        Id0d1[i] = np.sum(Pi * log0(p))
    return Id0d1



def ppmcc(P):
    """Calculates the mutual information between """
    r = np.zeros(P.shape[0])
    states = np.arange(1, P.shape[0] + 1)
    for i, Pi in enumerate(P):
        d0_mean = np.dot(Pi.sum(axis=1), states)
        d1_mean = np.dot(Pi.sum(axis=0), states)

        d0 = states - d0_mean
        d1 = states - d1_mean
        cov = (Pi.T * d0).T
        cov = cov * d1
        cov = cov.sum()
        vard0 = np.sqrt(np.dot(d0**2, Pi.sum(axis=1)))
        vard1 = np.sqrt(np.dot(d1**2, Pi.sum(axis=0)))
        # if vard0 == 0 or vard1 == 0:
        #     print(P[0, 0, 0], P[0, 1, 1])
        r[i] = cov / (vard0 * vard1)
    return r


def cramV(P):
    v = np.zeros(P.shape[0])
    for i, Pi in enumerate(P):
        r, k = Pi.shape
        ni_ = Pi.sum(axis=1)
        n_j = Pi.sum(axis=0)
        out = np.outer(ni_, n_j)

        chi2 = np.sum((Pi - out) ** 2 / out)
        v[i] = np.sqrt(chi2 / min(k - 1, r - 1))
    return v


def stationary_distribution(P):
    trans_mat = P.sum(axis=1)
    evals, evecs = np.linalg.eig(trans_mat.T)
    eval_inds = np.isclose(evals, 1)
    stat_distr = evecs[:, eval_inds]
    stat_distr /= stat_distr.sum(axis=0)
    if stat_distr.shape[1] > 1:
        print("Multiple stationary distributions")
    return np.real(stat_distr)


def float_digamma(x):
    x_ = x.copy()
    r = 0
    while x_ <= 5:
        r -= 1 / x_
        x_ += 1
    f = 1 / (x_ * x_)
    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0
        + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))))))
    return r + np.log(x_) - 0.5 / x_ + t


def digamma(x):
    """
    Credit to Tom Minka and his lightspeed package
    Faster digamma function - assumes x > 0.
    """
    if not isinstance(x, np.ndarray):
        return float_digamma(x)
    x_ = x.copy()
    # print(x_)
    r = np.zeros_like(x_)
    while (x_ <= 5).any():
        r[x_ <= 5] -= 1 / x_[x_ <= 5]
        x_[x_ <= 5] += 1
    f = 1 / (x_ * x_)
    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0
        + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))))))
    return r + np.log(x_) - 0.5 / x_ + t
