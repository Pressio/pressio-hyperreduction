
import numpy as np
from pressiotools import linalg as ptla

np.set_printoptions(linewidth=140)

def svd_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)
  # create the matrix A and use scipy QR
  A = np.asfortranarray(np.random.rand(15,4))
  U0,s0,VT0 = np.linalg.svd(A, full_matrices=False)
  if rank==0:
    print(s0)
    print(U0)
    print(VT0)
    print(np.dot(U0[:,0], U0[:,1]))
    print("----")

  # create distributed A
  myStartRow = rank*5
  A1   = ptla.MultiVector(A[myStartRow:myStartRow+5, :])
  svdO = ptla.Svd()
  svdO.computeThin(A1)
  U1 = svdO.viewLeftSingVectorsLocal()
  S1 = svdO.viewSingValues()
  VT1 = svdO.viewRightSingVectorsT()
  # print(S1)
  #print(U1)
  print(VT1)

  # sing values are replicated
  assert(np.allclose(np.abs(s0),np.abs(S1), atol=1e-10))

  # right sing vectors are replicated
  assert(np.allclose(np.abs(VT0),np.abs(VT1), atol=1e-10))

  # left sing vectors are distributed as A is
  myU0 = U0[myStartRow:myStartRow+5, :]
  assert(np.allclose(np.abs(myU0),np.abs(U1), atol=1e-10))

if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 3)
  svd_run(comm)
