
# Overview

`pressio-tools` includes functionalities accessible from Python currently supporting:

- QR factorization of a distributed (tall-skinny) matrix

- SVD of a distributed (tall-skinny) matrix

- computation of sample mesh indices for hyper-reduction via

	- leverage scores using algorithm 5.1 of [this paper](https://arxiv.org/pdf/1903.00911.pdf)

	- more methods to come

`pressio-tools` is being developed as an auxiliary library to the [pressio C++ library](https://pressio.github.io/pressio/html/index.html) and its [Python bindings library](https://pressio.github.io/pressio4py/html/index.html).

# When to use pressio-tools?

*`pressio-tools` is mainly intended to operate on large data distributed on large-scale machines.*

For example, suppose you want to use the SVD functionality. If you have a "small" matrix that fits on a single node, using pressio-tools to compute its SVD is excessive, and you (likely) can as easily use scipy.svd or other libraries for shared-memory computing like Eigen.
However, if you have a large tall-skinny matrix distributed over a large machine and need to compute its SVD, then `pressio-tools` is right for you.

# Installing
See the [wiki page](https://github.com/Pressio/pressio-tools/wiki/Requirements-and-installation).

# Usage

- Interested in the *QR* factorizaton?
  - Look at the [demo](https://github.com/Pressio/pressio-tools/blob/master/demos/qr.py).
  - To run it: `cd demos; mpirun -n 4 python3 qr.py`

- Interested in the *SVD* functionality?
  - Look at the [demo](https://github.com/Pressio/pressio-tools/blob/master/demos/svd.py).
  - To run it: `cd demos; mpirun -n 4 python3 svd.py`

- Interested in the *Sample Mesh* functionality based on leverage scores?
  - You can run the demo as:
  ```bash
  cd pressio-tools/driver-scripts
  mpirun -n 4 python3 hypred_levscores.py --input pressio-tools/demos/levscores/input.yaml`
  ```

# Questions?
Find us on Slack: https://pressioteam.slack.com or open an issue on [github](https://github.com/Pressio/pressio-tools).

# License and Citation
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

The full license is available [here](https://pressio.github.io/various/license/).
