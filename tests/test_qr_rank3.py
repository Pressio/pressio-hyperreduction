
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt

np.set_printoptions(linewidth=140)

def tsqr_run():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)

  # create the matrix A and use scipy QR
  A = np.asfortranarray(np.random.rand(15,4))
  Q1, R1 = np.linalg.qr(A)
  print(R1)

  # create distributed A
  myStartRow = rank*5
  A1 = pt.MultiVector(A[myStartRow:myStartRow+5, :])
  qrO = pt.Tsqr()
  qrO.computeThinOutOfPlace(A1)
  R = qrO.viewR()
  Q = qrO.viewLocalQ()

  # Q is distributed so check correct rows
  myQ = Q1[myStartRow:myStartRow+5, :]
  assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-10))

  # R is the replicated, so should be the same
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-10))


if __name__ == '__main__':
  tsqr_run()
