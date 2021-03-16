
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.romoperators.galerkinProjector import computeGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

#-----------------------------
def runDof1(comm):
  rank = comm.Get_rank()
  np.random.seed(3675638)

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(37, numCols))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  phi = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))


  # even if I pass a single list of glob indices,
  # the code will work beucase each rank will pick up
  # only the glob inds that pertain to it
  meshGlobIndices = [0, 11, 15, 34]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=meshGlobIndices,
                                       communicator=comm)
  print(rank, projector)
  if rank == 0: gold = A0[0,:]
  elif rank==2: gold = A0[11,:]
  elif rank==3: gold = A0[15,:]
  elif rank==5: gold = A0[34,:]

  if rank in [0,2,3,5]:
    assert( projector.shape[0] == 1)
    assert( projector.shape[1] == numCols)
    assert( np.all(projector == gold))


#-----------------------------
def runDof1Gappy(comm, phiNumCols, psiNumCols):
  rank = comm.Get_rank()

  np.random.seed(3675638)
  phi0 = np.asfortranarray(np.random.rand(37, phiNumCols))
  if rank==0:
    print(phi0)

  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(37, psiNumCols))
  if rank==0:
    print(psi0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  phi = ptla.MultiVector(np.asfortranarray(phi0[locRows, :]))
  psi = ptla.MultiVector(np.asfortranarray(psi0[locRows, :]))

  # even if I pass a single list of glob indices,
  # the code will work beucase each rank will pick up
  # only the glob inds that pertain to it
  meshGlobIndices = [0, 8,9,10,11, 15,16,17, 32,34]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       residualBasis=psi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=meshGlobIndices,
                                       communicator=comm)
  print(rank, projector)

  ## check correctness ##
  import scipy.linalg as scipyla
  Zpsi = psi0[meshGlobIndices, :]
  ZpsiPi = scipyla.pinv(Zpsi)
  #print(ZpsiPi.shape)
  B = np.transpose(psi0).dot(phi0)
  #print(B.shape)
  gold = np.transpose(ZpsiPi).dot(B)
  if rank==0:
    print(rank, gold)

  if rank==0:   myGold = gold[[0], :]
  elif rank==1: myGold = None
  elif rank==2: myGold = gold[np.arange(1,5), :]
  elif rank==3: myGold = gold[np.arange(5,8), :]
  elif rank==4: myGold = None
  elif rank==5: myGold = gold[np.arange(8,10), :]

  if rank not in [1,4]:
    assert( projector.shape == myGold.shape )
    assert(np.allclose(projector, myGold, atol=1e-12))
  else:
    assert(projector == None)


#-----------------------------
def runDof1GappyTwo(comm, phiNumCols, psiNumCols):
  rank = comm.Get_rank()

  np.random.seed(3675638)
  phi0 = np.asfortranarray(np.random.rand(21, phiNumCols))
  if rank==0: print(phi0)

  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(21, psiNumCols))
  if rank==0: print(psi0)

  if rank==0:   locRows  = np.arange(0, 4).tolist()
  elif rank==1: locRows  = np.arange(4, 11).tolist()
  elif rank==2: locRows  = np.arange(11, 14).tolist()
  elif rank==3: locRows  = np.arange(14, 16).tolist()
  elif rank==4: locRows  = np.arange(16, 20).tolist()
  elif rank==5: locRows  = np.arange(20, 21).tolist()

  phi = ptla.MultiVector(np.asfortranarray(phi0[locRows, :]))
  psi = ptla.MultiVector(np.asfortranarray(psi0[locRows, :]))

  # even if I pass a single list of glob indices,
  # the code will work beucase each rank will pick up
  # only the glob inds that pertain to it
  meshGlobIndices = [8,9,10, 15, 16,17]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       residualBasis=psi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=meshGlobIndices,
                                       communicator=comm)
  print(rank, projector)

  ## check correctness ##
  import scipy.linalg as scipyla
  Zpsi = psi0[meshGlobIndices, :]
  ZpsiPi = scipyla.pinv(Zpsi)
  B = np.transpose(psi0).dot(phi0)
  gold = np.transpose(ZpsiPi).dot(B)
  if rank==0: print(rank, gold)

  if rank==0:   myGold = None
  elif rank==1: myGold = gold[np.arange(0,3), :]
  elif rank==2: myGold = None
  elif rank==3: myGold = gold[[3], :]
  elif rank==4: myGold = gold[[4,5], :]
  elif rank==5: myGold = None

  if rank not in [0,2,5]:
    print( rank, projector.shape, myGold.shape )
    assert( projector.shape == myGold.shape )
    assert(np.allclose(projector, myGold, atol=1e-12))
  else:
    assert(projector == myGold)


############
### MAIN ###
############
if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 6)

  runDof1(comm)

  # run for various combo of # of phi and psi cols
  runDof1Gappy(comm, 6, 8)
  runDof1Gappy(comm, 8, 6)
  runDof1Gappy(comm, 2, 6)
  runDof1Gappy(comm, 6, 2)
  runDof1Gappy(comm, 1, 6)
  runDof1Gappy(comm, 6, 1)

  runDof1GappyTwo(comm, 2, 3)
