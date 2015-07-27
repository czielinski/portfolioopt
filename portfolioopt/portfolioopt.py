# The MIT License (MIT)
#
# Copyright (c) 2015 Christian Zielinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PortfolioOpt: Financial Portfolio Optimization

This module provides a set of functions for financial portfolio
optimization, such as construction of Markowitz portfolios, minimum
variance portfolios and tangency portfolios (i.e. maximum Sharpe ratio
portfolios) in Python. The construction of long-only, long/short and
market neutral portfolios is supported."""

import numpy as np
import pandas as pd
import cvxopt as opt
import cvxopt.solvers as optsolvers
import warnings


__all__ = ['markowitz_portfolio',
           'min_var_portfolio',
           'tangency_portfolio',
           'max_ret_portfolio',
           'truncate_weights']


def markowitz_portfolio(cov_mat, exp_rets, target_ret,
                        allow_short=False, market_neutral=False):
    """
    Computes a Markowitz portfolio.

    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).
    target_ret: float
        Target return of portfolio.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.
    market_neutral: bool, optional
        If 'False' sum of weights equals one.
        If 'True' sum of weights equal zero, i.e. create a
            market neutral portfolio (implies allow_short=True).
            
    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    if not isinstance(target_ret, float):
        raise ValueError("Target return is not a float")

    if not cov_mat.index.equals(exp_rets.index):
        raise ValueError("Indices do not match")

    if market_neutral and not allow_short:
        warnings.warn("A market neutral portfolio implies shorting")
        allow_short=True

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # exp_rets*x >= target_ret and x >= 0
        G = opt.matrix(np.vstack((-exp_rets.values,
                                  -np.identity(n))))
        h = opt.matrix(np.vstack((-target_ret,
                                  +np.zeros((n, 1)))))
    else:
        # exp_rets*x >= target_ret
        G = opt.matrix(-exp_rets.values).T
        h = opt.matrix(-target_ret)

    # Constraints Ax = b
    # sum(x) = 1
    A = opt.matrix(1.0, (1, n))

    if not market_neutral:
        b = opt.matrix(1.0)
    else:
        b = opt.matrix(0.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights


def min_var_portfolio(cov_mat, allow_short=False):
    """
    Computes the minimum variance portfolio.

    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.
    
    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # x >= 0
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(0.0, (n, 1))
    else:
        G = None
        h = None

    # Constraints Ax = b
    # sum(x) = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights


def tangency_portfolio(cov_mat, exp_rets, allow_short=False):
    """
    Computes a tangency portfolio, i.e. a maximum Sharpe ratio portfolio.
    
    Note: As the Sharpe ratio is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral tangency portfolios. This is because for
    a positive initial Sharpe ratio the sharpe grows unbound
    with increasing leverage.
    
    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    if not cov_mat.index.equals(exp_rets.index):
        raise ValueError("Indices do not match")

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # exp_rets*x >= 1 and x >= 0
        G = opt.matrix(np.vstack((-exp_rets.values,
                                  -np.identity(n))))
        h = opt.matrix(np.vstack((-1.0,
                                  np.zeros((n, 1)))))
    else:
        # exp_rets*x >= 1
        G = opt.matrix(-exp_rets.values).T
        h = opt.matrix(-1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)

    # Rescale weights, so that sum(weights) = 1
    weights /= weights.sum()
    return weights


def max_ret_portfolio(exp_rets):
    """
    Computes a long-only maximum return portfolio, i.e. selects
    the assets with maximal return. If there is more than one
    asset with maximal return, equally weight all of them.
    
    Parameters
    ----------
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    weights = exp_rets[:]
    weights[weights == weights.max()] = 1.0
    weights[weights != weights.max()] = 0.0
    weights /= weights.sum()

    return weights


def truncate_weights(weights, min_weight=0.01, rescale=True):
    """
    Truncates small weight vectors, i.e. sets weights below a treshold to zero.
    This can be helpful to remove portfolio weights, which are negligibly small.
    
    Parameters
    ----------
    weights: pandas.Series
        Optimal asset weights.
    min_weight: float, optional
        All weights, for which the absolute value is smaller
        than this parameter will be set to zero.
    rescale: boolean, optional
        If 'True', rescale weights so that weights.sum() == 1.
        If 'False', do not rescale.

    Returns
    -------
    adj_weights: pandas.Series
        Adjusted weights.
    """
    if not isinstance(weights, pd.Series):
        raise ValueError("Weight vector is not a Series")

    adj_weights = weights[:]
    adj_weights[adj_weights.abs() < min_weight] = 0.0

    if rescale:
        if not adj_weights.sum():
            raise ValueError("Cannot rescale weight vector as sum is not finite")
        
        adj_weights /= adj_weights.sum()

    return adj_weights
