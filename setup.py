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

from distutils.core import setup

setup(
    name = 'portfolioopt',
    packages = ['portfolioopt'],
    version = '0.2',
    description = 'Financial Portfolio Optimization Routines in Python',
    author = 'Christian Zielinski',
    author_email = 'email@czielinski.de',
    url = 'https://github.com/czielinski/portfolioopt',
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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Programming Language :: Python'
    ],
)
