
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.galerkinProjector import computeGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def runDof1(phi):
  smGIDs = [2,4,5]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=1,
                                       sampleMeshIndices=smGIDs)
  print(projector.data())
  gold = phi.data()[smGIDs,:]
  assert( projector.data().shape == gold.shape)
  assert( np.all(projector.data() == gold))

def runDof2(phi):
  smGIDs = [2,4,5]
  projector = computeGalerkinProjector(stateBasis=phi,
                                       dofsPerMeshNode=2,
                                       sampleMeshIndices=smGIDs)
  print(projector.data())
  gold = phi.data()[[4,5,8,9,10,11],:]
  assert( projector.data().shape == gold.shape)
  assert( np.all(projector.data() == gold))

if __name__ == '__main__':
  np.random.seed(312367)
  phi0 = np.asfortranarray(np.random.rand(24,4))
  phi = ptla.MultiVector(phi0)
  print(phi0)
  print("---------\n")

  #runDof1(phi)
  runDof2(phi)
