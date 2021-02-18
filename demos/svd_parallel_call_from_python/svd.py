
import sys
import numpy as np
import pressiotools

np.set_printoptions(linewidth=140)

def run(comm):
  '''
  this demo shows how to compute the parallel SVD
  over N ranks of a block-row distributed random matrix A.

  Suppose A is a 100x5 matrix such that it is
  block-row distributed over multiple MPI ranks.

  For example, suppose have 4 ranks:

       0     j        4
  i=0  ----------------
          rank=0
  24   ----------------
          rank=1
  49   ----------------
          rank=2
  74   ----------------
          rank=3
  i=99 ----------------

  Each rank owns 25 rows and all columns of A.
  '''
  rank = comm.Get_rank()

  # fix seed for reproducibility
  np.random.seed(312367)

  # create the local piece for each rank:
  # here for simplicity we use random numbers, but one can
  # fill each piece by reading a file or some other way.
  # Note: the layout MUST be fortran such that
  # pressiotools can view it without copying data.
  # This is important to reduced the memory footprint
  # especially for large matrices.
  myLocalRandPiece = np.asfortranarray(np.random.rand(25,5))

  # use the local piece to create a distributed multivector
  # the multivector is an abstraction that allows us
  # to easily perform all the computations behind the scenes
  # in native C++ without performance hit
  A = pressiotools.MultiVector(myLocalRandPiece)

  # construct a SVD object
  svdO = pressiotools.svd()
  # comptue thin svd
  svdO.computeThin(A)

  # U is a numpy array viewing the local piece
  U = svdO.viewLeftSingVectorsLocal()

  # S contains the sing values, replicated on each rank
  S = svdO.viewSingValues()

  # VT contains the transpose of the right-vectors, replicated on each rank
  VT = svdO.viewRightSingVectorsT()

  print("Rank = {}, U.shape = {}, S.shape = {}, VT.shape = {}".format(
    rank, U.shape, S.shape, VT.shape))

if __name__ == '__main__':
  '''
  run with:  mpirun -np 4 python3 svd.py
  '''
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  if (comm.Get_size() != 4):
    print("Rerun with 4 ranks")
    sys.exit()

  run(comm)
