
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla
import scipy.linalg as la

np.set_printoptions(linewidth=140)

#-----------------------------------------
def pinv_run1(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(37, numCols))
  BT = la.pinv(A0).T
  if rank==0:
    print(A0)
    print(BT)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  A1 = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))
  piO = ptla.PseudoInverse()
  piO.compute(A1)
  # # view the local part of A^*T
  # # remember that pressiotools.PseudoInverse stores (A^+)^T NOT A^+
  AstarT = piO.viewTransposeLocal()
  print("rank", rank, AstarT.shape)

  myBT = BT[locRows, :]
  assert(np.allclose(myBT, AstarT, atol=1e-12))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 6)
  pinv_run1(comm)
