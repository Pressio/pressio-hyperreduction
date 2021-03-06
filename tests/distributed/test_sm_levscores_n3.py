
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.withLeverageScores import computeNodes

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run(psi, comm):
  rank = comm.Get_rank()
  dofsPerNode = 2
  indices, pmf = computeNodes(matrix=psi,
                              numSamples=5,
                              dofsPerMeshNode=dofsPerNode,
                              communicator=comm)
  print(indices)
  if rank == 0:
    gold = [2]
  elif rank==1:
    gold = [4,5]
  elif rank==2:
    gold = [9]

  assert( len(indices) == len(gold))
  assert( all(indices == gold))

if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)
  psi0 = np.asfortranarray(np.random.rand(24,4))
  myNumRows = 8
  myStartRow = rank*myNumRows
  psi = ptla.MultiVector(psi0[myStartRow:myStartRow+myNumRows, :])
  run(psi, comm)
