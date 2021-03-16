
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np
import scipy.linalg as _scipyla
from ._linalg import Vector, MultiVector

class OnNodePySvd:
  """Fallback for SVD"""

  def __init__(self):
    self.U_ = None
    self.S_ = None
    self.V_ = None

  def computeThin(self, A):
    assert(A.data().ndim == 2)
    self.U_, self.S_, self.V_ = _scipyla.svd(A.data(), full_matrices=False)

  def viewSingValues(self):
    return self.S_

  def viewLeftSingVectorsLocal(self):
    return np.asfortranarray(self.U_)

  def viewRightSingVectorsT(self):
    return np.asfortranarray(self.V_.T)


class OnNodePyQR:
  """Fallback for QR"""

  def __init__(self):
    self.Q_ = None
    self.R_ = None

  def computeThinOutOfPlace(self, A):
    assert(A.data().ndim == 2)
    self.Q_, self.R_ = _scipyla.qr(A.data(), mode='economic')

  def viewR(self):
    return np.asfortranarray(self.R_)

  def viewQLocal(self):
    return np.asfortranarray(self.Q_)


class OnNodePseudoInverse:
  """Fallback for Pseudo-inverse"""

  def __init__(self):
    self.B_ = None

  def compute(self, A):
    assert(A.data().ndim == 2)
    self.B_ = _scipyla.pinv(A.data())
    print(self.B_)

  def viewTransposeLocal(self):
    return np.asfortranarray(self.B_.T)

  def apply(self, operand):
    return np.asfortranarray(np.dot(self.B_, operand.data()))

  def applyTranspose(self, operand):
    return np.asfortranarray(np.transpose(self.B_).dot(operand[:]))


# C = beta *C + alpha*op(A)*op(B)
def OnNodeProduct(modeA, modeB, alpha, A, B, beta, C):
  assert(A.data().ndim == 2)
  assert(B.data().ndim == 2)
  An = A.data()
  Bn = B.data()
  # here we need the [:] or it won't overwrite C
  if modeA=='T' and modeB=='N':
    C[:] = beta*C[:] + alpha*np.transpose(An[:]).dot(Bn[:])
  elif modeA=='N' and modeB=='N':
    C[:] = beta*C[:] + alpha*np.matmul(An[:], Bn[:])
  elif modeA=='N' and modeB=='T':
    C[:] = beta*C[:] + alpha*np.matmul(An[:], np.transpose(Bn[:]))
  else:
    sys.exit("Case not yet implemented")

# C = alpha*A^T*A
def OnNodeSelfTransposeSelf(alpha, A, C):
  An = A.data()
  # here we need the [:] or it won't overwrite C
  C[:] = alpha*np.transpose(An[:]).dot(An[:])


##############
### import ###
##############
try:
  from ._linalg import Svd, Tsqr, PseudoInverse, product, selfTransposeSelf
except ImportError:
  Svd  = OnNodePySvd
  Tsqr = OnNodePyQR
  PseudoInverse = OnNodePseudoInverse
  product = OnNodeProduct
  selfTransposeSelf = OnNodeSelfTransposeSelf
