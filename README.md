
# Overview

`pressio-tools` includes Python bindings providing the following functionalities:

- computation of sample mesh indices for hyper-reduction via

	- leverage scores: using algorithm 5.1 of [this paper](https://arxiv.org/pdf/1903.00911.pdf)

	- other methods to come soon

- SVD for shared- or distributed-memory (tall-skinny) matrix

- QR factorization of a shared- or distributed-memory (tall-skinny) matrix

`pressio-tools` is self-contained but mainly developed as an auxiliary tool to [pressio](https://pressio.github.io/).

<!-- # When to use pressio-tools? -->
<!-- *`pressio-tools` is mainly intended to operate on large data distributed on large-scale machines, but we also support sharemem scenarios.* -->
<!-- For example, suppose you want to use the SVD functionality. If you have a "small" matrix that fits on a single node, using pressio-tools to compute its SVD is excessive, and you (likely) can as easily use scipy.svd or other libraries for shared-memory computing like Eigen. -->
<!-- However, if you have a large tall-skinny matrix distributed over a large machine and need to compute its SVD, then `pressio-tools` is right for you. -->

# Installing

We provide two separate build/installation modes:

- to work *only* on shared-memory data:
  * basic mode requiring only Python and a C++ compiler; intended to get you started quickly
  * see the [wiki page](https://github.com/Pressio/pressio-tools/wiki/Sharedmemory-build:-requirements-and-installation).

- to work on potentially distributed-memory data:
  * more *advanced* mode relying on MPI and Trilinos to operate on potentially distributed large-scale data
  * see the [wiki page](https://github.com/Pressio/pressio-tools/wiki/MPI-build:-requirements-and-installation).

# Usage

### Interested in the *leverage scores-based sample mesh* functionality?
  - Look at the *sharedmemory* demo where data is read from a single ascii file
	- the matrix is stored in a single file
    - Run as:
	```bash
	export DEMODIR=pressio-tools/demos/samplemesh_levscores_onnode_via_driver_script
	cd pressio-tools/driver-scripts
	python3 hypred_levscores.py --input ${DEMODIR}/input.yaml --data-dir=${DEMODIR}
	```

  - Look at the *parallel* demo where data is split over multiple ascii files
    - the matrix is block-row distributed, stored as "matrix.txt.3.i", where i=0,1,2 is the MPI rank
    - Run as:
	```bash
	export DEMODIR=pressio-tools/demos/samplemesh_levscores_parallel_via_driver_script
	cd pressio-tools/driver-scripts
	mpirun -n 3 python3 hypred_levscores.py --input ${DEMODIR}/input.yaml --data-dir=${DEMODIR}
	```

### Interested in the *SVD* functionality?
  - Look at the *sharedmemory* demo where data is read from a single ascii file
	- the matrix is stored in a single file
    - Run as:
	```bash
	export DEMODIR=pressio-tools/demos/svd_onnode_via_driver_script
	cd pressio-tools/driver-scripts
	python3 computeSVD.py --input ${DEMODIR}/input.yaml --data-dir=${DEMODIR}
	```

  - Look at the *parallel* demo where data is split over multiple ascii files
    - the matrix is block-row distributed, stored as "matrix.txt.3.i", where i=0,1,2 is the MPI rank
    - Run as:
	```bash
	export DEMODIR=pressio-tools/demos/svd_parallel_via_driver_script
	cd pressio-tools/driver-scripts
	mpirun -n 3 python3 computeSVD.py --input ${DEMODIR}/input.yaml --data-dir=${DEMODIR}
	```

  - Look at the *parallel* demo where the parallel SVD is called within Python
    - the matrix is block-row distributed over 3 MPI ranks
    - Run as:
	```bash
	cd pressio-tools/demos/svd_parallel_call_from_python
	mpirun -n 3 python3 main.py
	```


<!-- - Interested in the *QR* factorizaton? -->
<!--   - You can look at the *distributed* case: -->
<!-- 	- [demo](https://github.com/Pressio/pressio-tools/blob/master/demos/qr.py). -->
<!--     - Run as: `cd demos; mpirun -n 4 python3 qr.py` -->


# Questions?
Find us on Slack: https://pressioteam.slack.com or open an issue on [github](https://github.com/Pressio/pressio-tools/issues).

# License and Citation
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

The full license is available [here](https://pressio.github.io/various/license/).
