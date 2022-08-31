# Financial Portfolio Optimization

_THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE._

This module provides a set of functions for financial portfolio optimization, such as construction of Markowitz portfolios, minimum variance portfolios and tangency portfolios (i.e. maximum Sharpe ratio portfolios) in Python. The construction of long-only, long/short and market neutral portfolios is supported. Please read the `docstring` of the function you intend to use by typing e.g. `help(portfolioopt.markowitz_portfolio)` in the interactive interpreter. You can also find some documentation [here](https://raw.githubusercontent.com/czielinski/portfolioopt/master/doc/portfolioopt.txt).

### Installation

To manually install the library, clone the repository via `git clone https://github.com/czielinski/portfolioopt.git` and install the module with `python setup.py install`. To install the requirements by hand you can also use `pip install -r requirements.txt`. You can run the tests with `python setup.py test` or with `python -m unittest discover` in the module directory. If everything is right, all tests should pass.

The `portfolioopt` module provides the optimization routines, the file `example.py` provides a simple usage example. Please also read the `LICENSE.txt` file.

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


Minimum variance portfolio (long only)
--------------------------------------
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


Minimum variance portfolio (long/short)
---------------------------------------
Optimal weights:
asset_a    0.294284
asset_b    0.192217
asset_c    0.138202
asset_d    0.208795
asset_e    0.166502
dtype: float64

Expected return:   0.0015013136255
Expected variance: 0.000443881332596
Expected Sharpe:   0.0712587148452


Markowitz portfolio (long only, target return: 0.00376)
-------------------------------------------------------
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


Markowitz portfolio (long/short, target return: 0.00376)
--------------------------------------------------------
Optimal weights:
asset_a    0.241321
asset_b    0.287506
asset_c   -0.006595
asset_d    0.365424
asset_e    0.112344
dtype: float64

Expected return:   0.00375616820372
Expected variance: 0.000587278581077
Expected Sharpe:   0.154996878211


Markowitz portfolio (market neutral, target return: 0.00376)
------------------------------------------------------------
Optimal weights:
asset_a   -0.088226
asset_b    0.158734
asset_c   -0.241207
asset_d    0.260916
asset_e   -0.090217
dtype: float64

Expected return:   0.00375618451738
Expected variance: 0.000397921658527
Expected Sharpe:   0.188299050118


Tangency portfolio (long only)
------------------------------
Optimal weights:
asset_a    0.013638
asset_b    0.370651
asset_c    0.000000
asset_d    0.615711
asset_e    0.000000
dtype: float64

Expected return:   0.00633799771227
Expected variance: 0.00123946655115
Expected Sharpe:   0.180025768076


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
