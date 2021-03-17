
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.romoperators.lspgWeighting import computeLspgWeighting

np.set_printoptions(linewidth=340, precision=14)
tol = 1e-14

#-----------------------------
def runDof1():
  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(37, 7))
  print(psi0)
  psi = ptla.MultiVector(psi0)

  meshGlobIndices = [0, 8,9,10,11, 15,16,17, 32,34]
  wMat = computeLspgWeighting(residualBasis=psi,
                              dofsPerMeshNode=1,
                              sampleMeshIndices=meshGlobIndices)
  print(wMat)

  ## check correctness ##
  import scipy.linalg as scipyla
  Zpsi = psi0[meshGlobIndices, :]
  ZpsiPsInv = scipyla.pinv(Zpsi)
  A = np.matmul(psi0, ZpsiPsInv)
  gold = np.transpose(A).dot(A)
  print("gold", gold)

  assert( wMat.shape == gold.shape )
  assert(np.allclose(wMat, gold, atol=1e-12))

#-----------------------------
def runDof2():
  np.random.seed(3274618)
  psi0 = np.asfortranarray(np.random.rand(38, 7))
  print(psi0)
  psi = ptla.MultiVector(psi0)

  meshGlobIndices = [0,1,8,9,10,13,15]
  wMat = computeLspgWeighting(residualBasis=psi,
                              dofsPerMeshNode=2,
                              sampleMeshIndices=meshGlobIndices)
  print(wMat)

  ## check correctness ##
  import scipy.linalg as scipyla
  # note that we have dofs/cell = 2, so here
  # we need to list the DOFs GIDs whcih are not just the mesh GIDs
  Zpsi = psi0[[0,1,2,3,16,17,18,19,20,21,26,27,30,31], :]
  ZpsiPsInv = scipyla.pinv(Zpsi)
  A = np.matmul(psi0, ZpsiPsInv)
  gold = np.transpose(A).dot(A)
  print("gold", gold)
  assert( wMat.shape == gold.shape )
  assert(np.allclose(wMat, gold, atol=1e-12))

############
### MAIN ###
############
if __name__ == '__main__':
  runDof1()
  runDof2()
