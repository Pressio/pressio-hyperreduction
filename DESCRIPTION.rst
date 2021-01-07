pressio-hyperreduction: Hyper-reduction tools for pressio
=====================================================================

This Python library provides hyper-reduction capabilities
to be used for pressio (website_) or pressio4py (website2_), or elsewhere.
The goal of this library is to TBD.

.. _website: https://pressio.github.io/pressio/html/index.html

.. _website2: https://pressio.github.io/pressio4py/html/index.html


..
   Install
   -------

   You can try to use `pip` directly:

   .. code-block:: bash

     pip install pressio-hyperreduction


   You can double check that everything worked fine by doing:

   .. code-block:: python

     import pressio-hyperreduction as phr
     print(phr.__version__)




..
   Running Demos/Tutorials
   -----------------------

   After installing the library, you can check run the regression tests:

   .. code-block:: bash

     git clone git@github.com:Pressio/pressio4py.git
     cd pressio4py/regression_tests
     pytest -s


   And you can check out the demos:

   .. code-block:: bash

     git clone git@github.com:Pressio/pressio4py.git
     cd pressio4py/demos
     python3 ./<demo-subdir-name>/main.py


   Documentation
   -------------

   The documentation (in progress) can be found (here_) with some demos already available.

   .. _here: https://pressio.github.io/pressio4py/html/index.html


   Citations
   ---------

   If you use this package, please acknowledge our work-in-progress:

   * Francesco Rizzi, Patrick J. Blonigan, Eric. Parish, Kevin T. Carlberg
     "Pressio: Enabling projection-based model reduction for large-scale nonlinear dynamical systems"
     https://arxiv.org/abs/2003.07798
