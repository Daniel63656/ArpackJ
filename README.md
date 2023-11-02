# ArpackJ
Java library that provides routines for quickly finding a few eigenvalues/eigenvectors of large sparse or dense matrices.
All of the functionality can be accessed through the various `eigs()` and `eigsh()` functions located in the MatrixDecomposition class. All functions can either be called by utilizing la4j's various matrix interfaces or by providing the left multiplication of a matrix as lambda function. Only real matrices of type double are supported.
This library is essentially a wrapper function for the ARPACK library, written in Fortran, as well as some openBLAS functions.

## Features

- Solve eigenvalue problems for square and symmetric matrices using `eigsh()` (returns real eigenvectors/eigenvalues)
  or non-symmetric matrices using `eigs()` (which returns complex eigenvalues/eigenvectors).
- Solve standard and general eigenvalue problems.
- Support for different modes and options such as shift-invert, buckling or cayley-transform
- LU decomposition
- Matrix inversion (real and complex).
- Supports any kind of la4j matrix (sparse, dense) or any matrix defined by the user as lambda function.

## Getting Started


### Usage

