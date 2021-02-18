
import pathlib, sys
import numpy as np
import pressiotools as pt
import scipy.linalg as la

np.set_printoptions(linewidth=140)

def pinv_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)
  A = np.asfortranarray(np.random.rand(10,4))

  B = la.pinv(A)
  print(B.T)

  A1 = pt.MultiVector(A)
  piO = pt.pinv()
  piO.compute(A1)
  AstarT = piO.viewTransposeLocal()
  print(AstarT)

  assert(np.allclose(B.T, AstarT, atol=1e-10))

def pinv_apply(comm):
  print("\n")
  np.random.seed(312367)

  # create the matrix
  A = np.asfortranarray(np.random.rand(10,4))

  # compute scipy pseudo-inverse
  B = la.pinv(A)
  print(B.T)
  # apply to vector of ones
  d0 = np.ones(10)
  c = np.dot(B, d0)
  print(c)

  # do same using our code
  A1 = pt.MultiVector(A)
  piO = pt.pinv()
  piO.compute(A1)
  d1 = pt.Vector(d0)
  c1 = piO.apply(d1)
  print(c1)

  assert(np.allclose(c, c1, atol=1e-10))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 1)

  pinv_run(comm)
  pinv_apply(comm)
