
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla
import scipy.linalg as la

np.set_printoptions(linewidth=140)

def pinv_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)

  # create matrix
  A = np.asfortranarray(np.random.rand(15,4))

  BT = la.pinv(A).T
  if rank==0:
    print(BT)

  # create distributed A
  myStartRow = rank*5
  A1 = ptla.MultiVector(A[myStartRow:myStartRow+5, :])
  piO = ptla.PseudoInverse()
  piO.compute(A1)
  # view the local part of A^*T
  # remember that pressiotools.PseudoInverse stores A^*T NOT A^*
  AstarT = piO.viewTransposeLocal()
  print("rank", rank, AstarT)

  myBT = BT[myStartRow:myStartRow+5, :]
  assert(np.allclose(myBT, AstarT, atol=1e-10))


def pinv_apply(comm):
  print("\n")
  rank = comm.Get_rank()
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
  A1 = ptla.MultiVector(A[myStartRow:myStartRow+5, :])
  piO = ptla.PseudoInverse()
  piO.compute(A1)
  d1 = ptla.Vector(np.ones(5))
  c1 = piO.apply(d1)
  print(c1)

  assert(np.allclose(c, c1, atol=1e-10))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 3)

  pinv_run(comm)
  pinv_apply(comm)
