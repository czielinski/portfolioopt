# Financial Portfolio Optimization

This module provides a set of functions for financial portfolio optimization, such as construction of Markowitz portfolios, minimum variance portfolios and tangency portfolios (i.e. maximum Sharpe ratio portfolios) in Python. Both the construction of long-only and long/short portfolios is supported.

### Installation

Clone the repository via `git clone https://github.com/czielinski/portfolioopt.git` and install the module requirements via `pip install -r requirements.txt`. Please also read the `LICENSE` file. To verify the installation run the tests with `python -m unittest discover` in the module directory or one of its parent folders. If everything is right, all tests should pass.

The module `portfolio` provides the optimization routines, `example.py` provides a simple usage example.

### Example

The example output looks like the following:
```
$ python example.py 

Example returns
---------------
                            asset_a   asset_b   asset_c   asset_d   asset_e
2000-01-01 00:00:00+00:00  0.025836 -0.005913  0.033384  0.077151 -0.010708
2000-01-02 00:00:00+00:00 -0.010707  0.079961  0.039372 -0.022474  0.028128
2000-01-03 00:00:00+00:00 -0.022171 -0.022286  0.013098 -0.094664 -0.085246
2000-01-04 00:00:00+00:00 -0.027114 -0.049642  0.016712 -0.044401 -0.069615
2000-01-05 00:00:00+00:00  0.074282 -0.010289  0.004376 -0.070237 -0.026219
2000-01-06 00:00:00+00:00  0.006546 -0.056550  0.019785 -0.029032 -0.013585
2000-01-07 00:00:00+00:00 -0.029085  0.093614  0.000325 -0.051886  0.042127
2000-01-08 00:00:00+00:00 -0.060042  0.011443 -0.096984 -0.065409  0.010843
2000-01-09 00:00:00+00:00  0.037923  0.009568 -0.004782 -0.014055 -0.072926
2000-01-10 00:00:00+00:00 -0.034992 -0.022032  0.053856  0.018181 -0.087152
...


Average returns
---------------
asset_a   -0.001237
asset_b    0.004848
asset_c   -0.003694
asset_d    0.007403
asset_e   -0.000610
dtype: float64


Covariance matrix
-----------------
          asset_a   asset_b   asset_c   asset_d   asset_e
asset_a  0.002027 -0.000362  0.000099 -0.000220 -0.000305
asset_b -0.000362  0.002421  0.000297  0.000090  0.000151
asset_c  0.000099  0.000297  0.002420  0.000020  0.000113
asset_d -0.000220  0.000090  0.000020  0.002302  0.000047
asset_e -0.000305  0.000151  0.000113  0.000047  0.002877


Minimum variance portfolio
--------------------------
Optimal weights:
asset_a    0.294283
asset_b    0.192216
asset_c    0.138206
asset_d    0.208794
asset_e    0.166501
dtype: float64

Expected return:   0.00150128915014
Expected variance: 0.000443881332631
Expected Sharpe:   0.0712575531382


Markowitz portfolio (target return: 0.00376)
--------------------------------------------
Optimal weights:
asset_a    0.235067
asset_b    0.286836
asset_c    0.001546
asset_d    0.368534
asset_e    0.108017
dtype: float64

Expected return:   0.00375625399053
Expected variance: 0.000587574392946
Expected Sharpe:   0.154961396104


Tangency portfolio (long only)
------------------------------
Optimal weights:
asset_a    1.363765e-02
asset_b    3.706513e-01
asset_c    1.654967e-09
asset_d    6.157111e-01
asset_e    8.937734e-09
dtype: float64

Expected return:   0.00633799763357
Expected variance: 0.00123946652675
Expected Sharpe:   0.180025767612


Tangency portfolio (long/short)
-------------------------------
Optimal weights:
asset_a    0.048052
asset_b    0.635228
asset_c   -0.534982
asset_d    0.936986
asset_e   -0.085284
dtype: float64

Expected return:   0.0119844615356
Expected variance: 0.00354334941516
Expected Sharpe:   0.201331410159
```
