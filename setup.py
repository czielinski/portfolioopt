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

"""This module provides a set of functions for financial portfolio
optimization, such as construction of Markowitz portfolios, minimum
variance portfolios and tangency portfolios (i.e. maximum Sharpe ratio
portfolios) in Python. The construction of long-only, long/short and
market neutral portfolios is supported."""

from setuptools import setup, find_packages

setup(
    name = 'portfolioopt',
    packages = find_packages(),
    version = '1.0.0',
    description = "PortfolioOpt: Financial Portfolio Optimization",
    long_description = __doc__,
    author = 'Christian Zielinski',
    author_email = 'email@czielinski.de',
    url = 'https://github.com/czielinski/portfolioopt',
    download_url = 'https://github.com/czielinski/portfolioopt/tarball/master',
    test_suite = 'portfolioopt.test_portfolioopt.make_test_suite',
    license = 'MIT',
    install_requires = [
        'numpy',
        'pandas',
        'cvxopt'
    ],
    keywords = [
        'portfolio',
        'optimization',
        'markowitz'
    ],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
