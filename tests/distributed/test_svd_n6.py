
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla
import scipy.linalg as la

np.set_printoptions(linewidth=140)

#-----------------------------------------
def run1(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(37, numCols))
  U0,s0,VT0 = np.linalg.svd(A0, full_matrices=False)
  if rank==0:
    print(A0)
    #print(BT)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  A1 = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))
  svdO = ptla.Svd()
  svdO.computeThin(A1)
  U1  = svdO.viewLeftSingVectorsLocal()
  S1  = svdO.viewSingValues()
  VT1 = svdO.viewRightSingVectorsT()
  print(rank, S1, s0)
  print(rank, U1.shape)

  # sing values are replicated
  assert(np.allclose(np.abs(s0),np.abs(S1), atol=1e-10))
  # right sing vectors are replicated
  assert(np.allclose(np.abs(VT0),np.abs(VT1), atol=1e-10))

  # left sing vectors are distributed as A is
  myU0 = U0[locRows, :]
  assert(np.allclose(np.abs(myU0),np.abs(U1), atol=1e-10))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 6)
  run1(comm)
