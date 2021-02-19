
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.withLeverageScores import findSampleMeshIndices

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run(psi):
  dofsPerNode = 2
  indices, pmf = findSampleMeshIndices(matrix=psi,
                                       numSamples=5,
                                       dofsPerMeshNode=dofsPerNode)
  print(indices)
  gold = [3,5,6,9]
  assert( len(indices) == len(gold))
  assert( all(indices == gold))

if __name__ == '__main__':
  np.random.seed(312367)
  psi0 = np.asfortranarray(np.random.rand(24,4))
  psi = pt.MultiVector(psi0)
  print(psi0)
  print("---------\n")
  run(psi)
