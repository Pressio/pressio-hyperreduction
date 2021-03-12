
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.galerkinProjector import computeGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def runDof1(phi, comm):
  rank = comm.Get_rank()

  if rank == 0: myMeshGlobIndices = [0]
  elif rank==1: myMeshGlobIndices = [9,10]
  elif rank==2: myMeshGlobIndices = [19]

  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=myMeshGlobIndices,
                                       communicator=comm)
  print(rank, projector.data())
  if rank == 0: gold = phi.data()[0,:]
  elif rank==1: gold = phi.data()[[5,6],:]
  elif rank==2: gold = phi.data()[3,:]

  if rank==0 or rank==2:
    assert( projector.data().shape[0] == 1)
  else:
    assert( projector.data().shape[0] == 2)

  assert( projector.data().shape[1] == 4)
  assert( np.all(projector.data() == gold))


def runDof2(phi, comm):
  rank = comm.Get_rank()

  if rank == 0: myMeshGlobIndices = [1]
  elif rank==1: myMeshGlobIndices = [5,7]
  elif rank==2: myMeshGlobIndices = [9]

  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=2,
                                       sampleMeshIndices=myMeshGlobIndices,
                                       communicator=comm)
  print(rank, projector.data())
  if rank == 0: gold = phi.data()[[2,3],:]
  elif rank==1: gold = phi.data()[[6,7,10,11],:]
  elif rank==2: gold = phi.data()[[2,3],:]

  if rank==0 or rank==2:
    assert( projector.data().shape[0] == 2)
  else:
    assert( projector.data().shape[0] == 4)

  assert( projector.data().shape[1] == 4)
  assert( np.all(projector.data() == gold))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)
  phi0 = np.asfortranarray(np.random.rand(24,4))

  if rank==0:
    myNumRows = 4
    myStartRow = 0
  elif rank==1:
    myNumRows = 12
    myStartRow = 4
  elif rank==2:
    myNumRows = 8
    myStartRow = 16

  phi = ptla.MultiVector(phi0[myStartRow:myStartRow+myNumRows, :])
  runDof1(phi, comm)
  runDof2(phi, comm)
