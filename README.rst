===============
Conditional KDE
===============

.. image:: https://img.shields.io/pypi/v/conditional_kde.svg
        :target: https://pypi.python.org/pypi/conditional_kde
        :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/conditional_kde.svg
        :target: https://pypi.python.org/pypi/conditional_kde
        :alt: Python Versions

.. image:: https://github.com/dprelogo/conditional_kde/workflows/CI/badge.svg
        :target: https://github.com/dprelogo/conditional_kde/actions?query=workflow%3ACI
        :alt: CI Status

.. image:: https://codecov.io/gh/dprelogo/conditional_kde/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/dprelogo/conditional_kde
        :alt: Code Coverage

.. image:: https://readthedocs.org/projects/conditional-kde/badge/?version=latest
        :target: https://conditional-kde.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/dprelogo/conditional_kde.svg
        :target: https://github.com/dprelogo/conditional_kde/blob/main/LICENSE
        :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Code Style: Black




Conditional Kernel Density Estimation
-------------------------------------

A Python package for conditional kernel density estimation. This library provides efficient implementations for estimating conditional probability densities using kernel methods.

* **Free software:** MIT license
* **Documentation:** https://conditional-kde.readthedocs.io
* **PyPI:** https://pypi.org/project/conditional_kde/
* **Source Code:** https://github.com/dprelogo/conditional_kde


Installation
------------

Install from PyPI::

    pip install conditional_kde

For development installation::

    git clone https://github.com/dprelogo/conditional_kde.git
    cd conditional_kde
    pip install -e .[dev]


Quick Start
-----------

.. code-block:: python

    from conditional_kde import ConditionalKDE

    # Example usage
    ckde = ConditionalKDE()
    # Add your code example here


Features
--------

* Gaussian and interpolated kernel density estimation
* Support for conditional density estimation
* Efficient implementation using NumPy and SciPy
* Comprehensive test coverage
* Type hints for better IDE support

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
