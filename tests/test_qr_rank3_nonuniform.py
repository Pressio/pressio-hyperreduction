
import pathlib, sys
import numpy as np
import pressiotools as pt

np.set_printoptions(linewidth=140)

def tsqr_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)

  # create the matrix A and use scipy QR
  A = np.asfortranarray(np.random.rand(30,4))
  Q1, R1 = np.linalg.qr(A)
  print(R1)

  # create distributed A
  if rank==0:
    A1 = A[0:12, :]
  elif rank==1:
    A1 = A[12:20, :]
  elif rank==2:
    A1 = A[20:30, :]

  A2 = pt.MultiVector(A1)

  qrO = pt.Tsqr()
  qrO.computeThinOutOfPlace(A2)
  R = qrO.viewR()
  Q = qrO.viewQLocal()

  # Q is distributed so check correct rows
  if rank == 0:
    myQ = Q1[0:12, :]
  elif rank == 1:
    myQ = Q1[12:20, :]
  else:
    myQ = Q1[20:30, :]

  assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-10))
  # R is the replicated, so should be the same
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-10))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 3)
  tsqr_run(comm)
