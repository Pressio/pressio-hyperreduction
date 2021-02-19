
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla

np.set_printoptions(linewidth=140)

def tsqr_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)

  # create the matrix A and use scipy QR
  A = np.asfortranarray(np.random.rand(15,4))
  Q1, R1 = np.linalg.qr(A)
  print(R1)

  # create distributed A
  myStartRow = rank*5
  A1  = ptla.MultiVector(A[myStartRow:myStartRow+5, :])
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)
  R = qrO.viewR()
  Q = qrO.viewQLocal()

  # Q is distributed so check correct rows
  myQ = Q1[myStartRow:myStartRow+5, :]
  assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-10))

  # R is the replicated, so should be the same
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-10))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 3)

  tsqr_run(comm)
