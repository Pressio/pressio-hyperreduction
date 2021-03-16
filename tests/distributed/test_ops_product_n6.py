
import numpy as np
import sys
import pressiotools.linalg as ptla

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

#-----------------------------
def run1(comm):
  rank = comm.Get_rank()

  np.random.seed(3274618)
  A0 = np.asfortranarray(np.random.rand(37, 7))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  A = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))

  # compute B = A^T A, B is replicated on all ranks
  B = np.zeros((A0.shape[1], A0.shape[1]), order='F')
  ptla.selfTransposeSelf(1., A, B)
  if rank==0: print(B)

  ## check correctness ##
  Btrue = np.dot(A0.T, A0)
  if rank==0: print(Btrue)
  assert( Btrue.shape == B.shape )
  assert(np.allclose(B, Btrue, atol=1e-12))

#-----------------------------
def run2(comm):
  rank = comm.Get_rank()

  np.random.seed(3274618)
  A0 = np.asfortranarray(np.random.rand(37, 7))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = []
  elif rank==4: locRows  = np.arange(15, 37).tolist()
  elif rank==5: locRows  = []

  A = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))

  # compute B = A^T A, B is replicated on all ranks
  B = np.zeros((A0.shape[1], A0.shape[1]), order='F')
  ptla.selfTransposeSelf(1., A, B)
  if rank==0: print(B)

  ## check correctness ##
  Btrue = np.dot(A0.T, A0)
  if rank==0: print(Btrue)

  ## check correctness ##
  assert( Btrue.shape == B.shape )
  assert(np.allclose(B, Btrue, atol=1e-12))


############
### MAIN ###
############
if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 6)

  run1(comm)
  run2(comm)
