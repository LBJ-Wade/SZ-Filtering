#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

setup(
    name="ilc_tools",
    version="1.0",
    author="Rovina Pinto",
    author_email="pinto.rovina@astro.uni-bonn.de",
    packages=["ilc_tools"],
    url="https://github.com/rovinapinto/ilc_tools",
    license="MIT License",
    description=("A tool box for studying galaxy clusters with the SZ effect."),
    #long_description=open("README.rst").read(),
    package_data={"ilc_tools": ["LICENSE", "data/*.txt", "data/*.fits", "data/*.csv",]},
    include_package_data=True,
    install_requires=["matplotlib", "numpy", "astropy", "scipy", "healpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
