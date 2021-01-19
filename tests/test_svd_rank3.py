
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg.lapack as la

np.set_printoptions(linewidth=140)

def svd_run():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)
  # create the matrix A and use scipy QR
  A = np.asfortranarray(np.random.rand(15,4))
  U0,s0,V0 = np.linalg.svd(A, full_matrices=False)
  if rank==0:
    print(s0)
    print(U0)
    print(V0)
    print(np.dot(U0[:,0], U0[:,1]))
    print("----")

  # create distributed A
  myStartRow = rank*5
  A1 = pt.MultiVector(A[myStartRow:myStartRow+5, :])
  # A1 = pt.MultiVector(A)
  svdO = pt.svd()
  svdO.computeThin(A1)
  U1 = svdO.viewLeftSingVectors()
  S1 = svdO.viewSingValues()
  VT1 = svdO.viewRightSingVectorsT()
  # print(S1)
  print(U1)
  # print(VT1)

  # sing values are replicated
  assert(np.allclose(np.abs(s0),np.abs(S1), atol=1e-10))

  myU0 = U0[myStartRow:myStartRow+5, :]
  assert(np.allclose(np.abs(myU0),np.abs(U1), atol=1e-10))

  myV0 = V0[myStartRow:myStartRow+5, :]
  assert(np.allclose(np.abs(V0),np.abs(VT1), atol=1e-10))

if __name__ == '__main__':
  svd_run()
