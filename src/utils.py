import numpy as np


def ask(x):
    """
    Compute best ask of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int of ask price

    """
    pa = np.where(x > 0)[0]
    if len(pa) == 0:
        p = len(x)
    else:
        p = pa[0]
    return p


def bid(x):
    """
    Compute best bid of limit order book

    Parameters
    ----------
    x : numpy array
        volumes of each level of the order book

    Returns
    -------
    int of bid price

    """
    pa = np.where(x < 0)[0]
    if len(pa) == 0:
        p = -1
    else:
        p = pa[-1]
    return p
