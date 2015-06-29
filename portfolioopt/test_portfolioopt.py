#!/usr/bin/python

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

import unittest
import numpy as np
import pandas as pd

import portfolioopt as pfopt


__all__ = ['create_test_data']


def create_test_data(my_seed=42, num_days=100):
    """
    Creates some test returns data together with
    its covariance matrix and average returns.
    
    Parameters
    ----------
    my_seed: integer, optional
        Covariance matrix of asset returns.
    num_days: integer, optional
        Expected asset returns (often historical returns).

    Returns
    -------
    returns: pandas.DataFrame
        Test returns.
    cov_mat: pandas.DataFrame
        Test covariance matrix.
    avg_rets: pandas.Series
        Test average returns.
    """
    np.random.seed(my_seed)

    data = np.random.normal(loc=0.001, scale=0.05, size=(num_days, 5))
    dates = pd.date_range('1/1/2000', periods=num_days, freq='D', tz='UTC')
    assets = ['asset_a', 'asset_b', 'asset_c', 'asset_d', 'asset_e']

    returns = pd.DataFrame(data, columns=assets, index=dates)
    avg_rets = returns.mean()
    cov_mat = returns.cov()

    return returns, cov_mat, avg_rets


class TestMarkowitzPortfolio(unittest.TestCase):
    def test_long_only(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target_ret).values
        exp_weights = [0.23506651774627838, 0.28683592298360255, 0.0015464104918494993,
                       0.3685342846149573, 0.10801686416331224]

        self.assertTrue(np.allclose(calc_weights, exp_weights))

    def test_allow_short(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target_ret, allow_short=True).values
        exp_weights = [0.24132125485063094, 0.28750580615601806, -0.0065950973956575331,
                       0.36542391192949747, 0.11234412445951109]

        self.assertTrue(np.allclose(calc_weights, exp_weights))


class MinVarPortfolio(unittest.TestCase):
    def test_long_only(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.min_var_portfolio(cov_mat).values
        exp_weights = [0.29428271128647232, 0.19221596707310873, 0.13820590476491501,
                       0.20879394627003695, 0.16650147060546697]

        self.assertTrue(np.allclose(calc_weights, exp_weights))

    def test_allow_short(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.min_var_portfolio(cov_mat, allow_short=True).values
        exp_weights = [0.29428401463312454, 0.19221716939564482, 0.13820233202108606,
                       0.20879490895467365, 0.16650157499547097]

        self.assertTrue(np.allclose(calc_weights, exp_weights))


class TangencyPortfolio(unittest.TestCase):
    def test_long_only(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.tangency_portfolio(cov_mat, avg_rets).values
        exp_weights = [0.013637652162222968, 0.37065128018786714, 1.6549667634656901e-09,
                       0.61571105705720952, 8.9377335719926867e-09]

        self.assertTrue(np.allclose(calc_weights, exp_weights))

    def test_allow_short(self):
        returns, cov_mat, avg_rets = create_test_data()
        target_ret = avg_rets.quantile(0.7)

        calc_weights = pfopt.tangency_portfolio(cov_mat, avg_rets, allow_short=True).values
        exp_weights = [0.048052417309504825, 0.63522794399754601, -0.53498204281249118,
                       0.93698599544795846, -0.085284313942518161]

        self.assertTrue(np.allclose(calc_weights, exp_weights))


if __name__ == '__main__':
    unittest.main()
