
# Overview

`pressio-tools` is a collection of capabilities accessible from Python for:

- computing the QR factorization of a distributed tall-skinny matrix

- computing the SVD of a distributed matrix

- computing sample mesh indices for hyperreduction via

	- Discrete empirical interpolation method (DEIM)

	- other methods TBD

`pressio-tools` is being developed as an auxiliary library to the [pressio C++ library](https://pressio.github.io/pressio/html/index.html) and its [Python bindings library](https://pressio.github.io/pressio4py/html/index.html).

# When to use pressio-tools?

*`pressio-tools` is mainly intended to operate on large data distributed on large-scale machines.*
For example, suppose you want to use the SVD functionality. If you have a "small" matrix that fits on a single node, using pressio-tools to compute its SVD is excessive, and you (likely) can as easily use scipy.svd or other libraries for shared-memory computing like Eigen.
However, if you have a large tall-skinny matrix distributed over a large machine and need to compute its SVD, then `pressio-tools` is right for you.


# Requirements

- CMake 3.11.0 or newer

- A C,C++ (supporting C++14) and Fortran compilers: for example GNU compilers.

- BLAS/LAPACK:
  - default libs typically already exist on your machine
  - if you need to install them yourself, you can use a package manager
  - for example, OpenBLAS contains both BLAS and LAPACK: https://www.openblas.net/

- Python3, pip, numpy, scipy

- MPI:
  - to install from scratch, see e.g. this: https://www.open-mpi.org/software/ompi/v4.1/.
  - or you can use a package manager

- mpi4py:
  - with MPI installed, installing mpi4py is easy: https://mpi4py.readthedocs.io/en/stable/install.html

- Trilinos:
  - is used as the backend to efficiently compute some large-scale operations, e.g., distributed TSQR
  - if you don't have Trilinos, you can try to let pressio-tools build it for you. If you already have Trilinos on your machine, you can tell pressio-tools to use that installation. See below the building section for more details.

**Note**: if you are using large-scale HPC platforms, e.g., NERSC, ALCF or ORLC machines, the above dependencies are pretty much already satisfied and accessible via ready-to-use modules for MPI, Python, Trilinos, PETSc, etc.



# Installing pressio-tools

We envision the following scenarios conditioned on having Trilinos available.

## (a) You already have Trilinos installed somewhere

Make sure that your Trilinos was built with `-DTrilinos_ENABLE_TpetraTSQR=ON`. This is needed for pressio-tools.
```bash
export MPI_BASE_DIR=<full-path-to-your-MPI-installation>
export TRILINOS_ROOT=<full-path-to-your-trilinos-installation>

# if you have ssh keys ready for github (--recursive is needed for git submodules)
git clone --recursive git@github.com:Pressio/pressio-tools.git
# for http: git clone --recursive https://github.com/Pressio/pressio-tools.git

pip install ./pressio-tools
```

## (b) You DO NOT have Trilinos and want to build it on your own

If you want to build Trilinos on your own, look at the [Trilinos website](https://github.com/trilinos/Trilinos) and use their documentation to build Trilinos on your own.
**Note** that Trilinos contains a lot of packages, and pressio-tools does NOT need *all* of them.
To help you out, here is a sample configure script only enabling the minimun set needed for `pressio-tools`.
```bash
#!/bin/bash

TRILINOS_SRC=<point-to-your-trilinos-source>
MPI_BASE_DIR=<full-path-to-your-MPI-installation>
TRILINOS_PFX=<where-you-want-to-install-trilinos>

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE="Release" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
	-DTPL_ENABLE_MPI=ON \
	-DMPI_BASE_DIR=${MPI_BASE_DIR} \
	-DTrilinos_ENABLE_Tpetra=ON \
	-DTrilinos_ENABLE_TpetraTSQR=ON \
	-DTrilinos_ENABLE_Epetra=ON \
	-DCMAKE_INSTALL_PREFIX=${TRILINOS_PFX} \
	${TRILINOS_SRC}

make -j4 install
```
Note the specification of `-DTrilinos_ENABLE_TpetraTSQR=ON`, since the TSQR is critical for pressio-tools.
Trilinos will use the `MPI_BASE_DIR` you set above to find MPI, and will try to find BLAS and LAPACK automatically via CMake. In this case, to ensure Trilinos finds a *specific* BLAS/LAPACK, you need to set appropriate env vars as detailed for [BLAS](https://cmake.org/cmake/help/latest/module/FindBLAS.html) and [LAPACK](https://cmake.org/cmake/help/latest/module/FindLAPACK.html), or look at the Trilinos documentation directly.

If you build Trilinos this way, you can then set `TRILINOS_ROOT` to point the installation you just completed, and then proceed as in case (a).


## (c) You DO NOT have Trilinos but want pressio-tools to build it

If you don't want to build Trilinos yourself, but want `pressio-tools` build it for you, you can proceed as follows:
```bash
# ensure TRILINOS_ROOT is not defined
unset TRILINOS_ROOT

export MPI_BASE_DIR=<full-path-to-your-MPI-installation>
# if you have ssh keys ready for github (--recursive is needed for git submodules)
git clone --recursive git@github.com:Pressio/pressio-tools.git
# for http: git clone --recursive https://github.com/Pressio/pressio-tools.git

cd pressio-tools
python3 setup.py build
python3 setup.py install
```
In this case, a default, basic build of Trilinos is attempted, and automatically used to build the library `pressio-tools`.

### Things that can go wrong

- If the Trilinos build does not succeed:
  - the subdirectory `pressio-tools/build/trilinos/build`, contains the CMakeCache and build of Trilinos
  - check that the configure steps of Trilinos worked and/or what errors were triggered. If there are errors, remove that subdirectory and try again.
  - check that the Trilinos found BLAS/LAPACK
  - make sure the compilers used in all steps are consistent
  - if you cannot make it work, open an issue or reach out to us


# Questions?
Find us on Slack: https://pressioteam.slack.com or open an issue on [github](https://github.com/Pressio/pressio-tools).


# License and Citation
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

The full license is available [here](https://pressio.github.io/various/license/).
