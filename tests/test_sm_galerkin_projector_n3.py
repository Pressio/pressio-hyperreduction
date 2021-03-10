
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.galerkinProjector import createGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run(phi, comm):
  rank = comm.Get_rank()
  dofsPerNode = 2
  if rank == 0:
    indices = [2]
  elif rank==1:
    indices = [4,5]
  elif rank==2:
    indices = [2,3,4]



  projector = createGalerkinProjector(stateBasis=phi,
                                      samplingMatrixLocalInds=indices,
                                      communicator=comm)
  print(rank, indices)
  if rank == 0:
    gold = phi.data()[indices,:]
  elif rank==1:
    gold = phi.data()[indices,:]
  elif rank==2:
    gold = phi.data()[indices,:]
  
  assert( projector.data().shape == gold.shape)
  assert( np.all(projector.data() == gold))

if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)
  phi0 = np.asfortranarray(np.random.rand(24,4))
  myNumRows = 8
  myStartRow = rank*myNumRows
  phi = ptla.MultiVector(phi0[myStartRow:myStartRow+myNumRows, :])
  run(phi, comm)
