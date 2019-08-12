#!/bin/python
#
# Copyright (c) 2019 Shoaib Ahmed Siddiqui

from __future__ import print_function

import sys
import glob
from distutils.core import setup

if sys.version_info < (3, 6):
    sys.exit("Python versions less than 3.6 are not supported")

# scripts = glob.glob("wl-*[a-z]")
prereqs = """
    numpy
""".split()

setup(
    name='torchbrain',
    version='v0.0',
    author="Shoaib Ahmed Siddiqui",
    description="PyTorch library for spiking NNs.",
    packages=["spiking"],
    scripts=None,
    install_requires=prereqs,
)
