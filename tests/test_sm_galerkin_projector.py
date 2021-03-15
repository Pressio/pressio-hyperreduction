
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.galerkinProjector import computeGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

#-------------------------
def runDof1Coll(phi):
  smGIDs = [2,4,5]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=smGIDs)
  print(projector)
  gold = phi.data()[smGIDs,:]
  assert( projector.shape == gold.shape)
  assert( np.all(projector == gold))

#-------------------------
def runDof2Coll(phi):
  smGIDs = [2,4,5]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=2,
                                       sampleMeshIndices=smGIDs)
  print(projector)
  gold = phi.data()[[4,5,8,9,10,11],:]
  assert( projector.shape == gold.shape)
  assert( np.all(projector == gold))

#-------------------------
def runDof1Gappy(phi, psi):
  smGIDs = [2,4,5,6,7,8,16,17]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       residualBasis=psi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=smGIDs)
  print(projector)

  # since dofsPerNode == 1, the rows of psi are same as smGIDs
  Zpsi = psi.data()[smGIDs,:]
  import scipy.linalg as scipyla
  ZpsiPi = scipyla.pinv(Zpsi)
  print(ZpsiPi.shape)

  B = np.transpose(psi.data()).dot(phi.data())
  print(B.shape)

  gold = np.transpose(ZpsiPi).dot(B)
  assert( projector.shape == gold.shape)
  assert( np.all(projector == gold))

#-------------------------
def runDof2Gappy(phi, psi):
  smGIDs = [2,5,10]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       residualBasis=psi,
                                       dofsPerMeshNode=2,
                                       sampleMeshIndices=smGIDs)
  print(projector)
  # since dofsPerNode==2, the rows indices are NOT just the sampleMeshGids
  # we need to account that each mesh node has 2 dofs inside
  A = psi.data()[[4,5,10,11,20,21],:]
  import scipy.linalg as scipyla
  Api = scipyla.pinv(A)
  B = np.transpose(psi.data()).dot(phi.data())
  gold = np.transpose(Api).dot(B)
  print(gold)
  assert( projector.shape == gold.shape)
  assert( np.all(projector == gold))


### MAIN ###
if __name__ == '__main__':
  np.random.seed(312367)
  phi0 = np.asfortranarray(np.random.rand(24,4))
  print(phi0)
  phi = ptla.MultiVector(phi0)
  print("---------\n")

  runDof1Coll(phi)
  runDof2Coll(phi)

  np.random.seed(4451236)
  # note that we make psi with 6 cols
  psi0 = np.asfortranarray(np.random.rand(24,6))
  print(psi0)
  psi  = ptla.MultiVector(psi0)
  print("---------\n")

  runDof1Gappy(phi, psi)
  runDof2Gappy(phi, psi)
