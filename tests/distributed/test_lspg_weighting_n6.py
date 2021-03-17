
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.romoperators.lspgWeighting import computeLspgWeighting

np.set_printoptions(linewidth=340, precision=14)
tol = 1e-14

#-----------------------------
def runDof1(comm):
  rank = comm.Get_rank()

  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(37, 7))
  if rank==0:
    print(psi0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  psi = ptla.MultiVector(np.asfortranarray(psi0[locRows, :]))

  # even if I pass a single list of glob indices,
  # the code will work beucase each rank will pick up
  # only the glob inds that pertain to it
  meshGlobIndices = [0, 8,9,10,11, 15,16,17, 32,34]
  wMat = computeLspgWeighting(residualBasis=psi,
                              dofsPerMeshNode=1,
                              sampleMeshIndices=meshGlobIndices,
                              communicator=comm)
  print(rank, wMat)

  ## check correctness ##
  import scipy.linalg as scipyla
  Zpsi = psi0[meshGlobIndices, :]
  ZpsiPsInv = scipyla.pinv(Zpsi)
  A = np.matmul(psi0, ZpsiPsInv)
  gold = np.transpose(A).dot(A)
  if rank==0:
    print(rank, "gold", gold)


  if rank==0:   myGold = gold[[0], :]
  elif rank==1: myGold = None
  elif rank==2: myGold = gold[np.arange(1,5), :]
  elif rank==3: myGold = gold[np.arange(5,8), :]
  elif rank==4: myGold = None
  elif rank==5: myGold = gold[np.arange(8,10), :]

  if rank not in [1,4]:
    assert( wMat.shape == myGold.shape )
    assert(np.allclose(wMat, myGold, atol=1e-12))
  else:
    assert(wMat == None)


#-----------------------------
def runDof2(comm):
  rank = comm.Get_rank()

  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(38, 7))
  if rank==0: print(psi0)

  # here we use dof/cell = 2 so when we divide,
  # make sure we have an integer num of mesh cells for each rank
  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 16).tolist()
  elif rank==3: locRows  = np.arange(16, 20).tolist()
  elif rank==4: locRows  = np.arange(20, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 38).tolist()

  psi = ptla.MultiVector(np.asfortranarray(psi0[locRows, :]))

  # even if I pass a single list of glob indices,
  # the code will work beucase each rank will pick up
  # only the glob inds that pertain to it
  meshGlobIndices = [0,1,8,9,10,13,15]
  wMat = computeLspgWeighting(residualBasis=psi,
                              dofsPerMeshNode=2,
                              sampleMeshIndices=meshGlobIndices,
                              communicator=comm)
  print(rank, wMat)

  ## check correctness ##
  import scipy.linalg as scipyla
  # note that we have dofs/cell = 2, so here
  # we need to list the DOFs GIDs whcih are not just the mesh GIDs
  Zpsi = psi0[[0,1,2,3,16,17,18,19,20,21,26,27,30,31], :]
  ZpsiPsInv = scipyla.pinv(Zpsi)
  A = np.matmul(psi0, ZpsiPsInv)
  gold = np.transpose(A).dot(A)
  if rank==0: print(rank, "gold", gold)

  # note that we have dofs/cell = 2, so here
  # we need to list the DOFs GIDs whcih are not just the mesh GIDs
  if rank==0:   myGold = gold[[0,1,2,3], :]
  elif rank==1: myGold = None
  elif rank==2: myGold = None
  elif rank==3: myGold = gold[[4,5,6,7], :]
  elif rank==4: myGold = gold[[8,9,10,11], :]
  elif rank==5: myGold = gold[[12,13], :]

  if rank not in [1,2]:
    assert( wMat.shape == myGold.shape )
    assert(np.allclose(wMat, myGold, atol=1e-12))
  else:
    assert(wMat == None)


############
### MAIN ###
############
if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 6)

  runDof1(comm)
  runDof2(comm)
