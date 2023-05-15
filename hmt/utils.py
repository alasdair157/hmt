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
    states = np.arange(1, P.shape[0] + 1)[::-1]
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
        if vard0 == 0 or vard1 == 0:
            print(P[0, 0, 0], P[0, 1, 1])
        r[i] = cov / (vard0 * vard1)
    return r


def digamma(x):
    """
    Credit to Tom Minka and his lightspeed package
    Faster digamma function assumes x > 0.
    """
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t
