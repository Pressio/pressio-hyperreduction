
import pathlib, sys
import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la

np.set_printoptions(linewidth=140)

def pinv_run():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)

  # create matrix
  A = np.asfortranarray(np.random.rand(15,4))

  BT = la.pinv(A).T
  if rank==0:
    print(BT)

  # create distributed A
  myStartRow = rank*5
  A1 = pt.MultiVector(A[myStartRow:myStartRow+5, :])
  piO = pt.pinv()
  piO.compute(A1)
  # view the local part of A^*T
  # remember that pressiotools.pinv stores A^*T NOT A^*
  AstarT = piO.viewTransposeLocal()
  print("rank", rank, AstarT)

  myBT = BT[myStartRow:myStartRow+5, :]
  assert(np.allclose(myBT, AstarT, atol=1e-10))


def pinv_apply():
  print("\n")
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)

  # create the matrix
  A = np.asfortranarray(np.random.rand(15,4))

  # compute scipy pseudo-inverse
  B = la.pinv(A)
  print(B.T)
  # apply to vector of ones
  d0 = np.ones(15)
  c = np.dot(B, d0)
  print(c)

  # do same using our code
  # create distributed A
  myStartRow = rank*5
  A1 = pt.MultiVector(A[myStartRow:myStartRow+5, :])
  piO = pt.pinv()
  piO.compute(A1)
  d1 = pt.Vector(np.ones(5))
  c1 = piO.apply(d1)
  print(c1)

  assert(np.allclose(c, c1, atol=1e-10))


if __name__ == '__main__':
  pinv_run()
  pinv_apply()
