
import numpy as np
import sys
import pressiotools.linalg as ptla
from pressiotools.samplemesh.galerkinProjector import createGalerkinProjector

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run(phi):
  dofsPerNode = 2
  indices = [2,4,5]
  projector = createGalerkinProjector(stateBasis=phi,
                                      samplingMatrixLocalInds=indices)
  gold = phi.data()[indices,:]
  assert( projector.data().shape == gold.shape)
  assert( np.all(projector.data() == gold))

if __name__ == '__main__':
  np.random.seed(312367)
  phi0 = np.asfortranarray(np.random.rand(24,4))
  phi = ptla.MultiVector(phi0)
  print(phi0)
  print("---------\n")
  run(phi)
