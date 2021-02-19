
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''


import scipy.linalg as _scipyla
from ._linalg import Vector, MultiVector

class OnNodePySvd:
  """Fallback for SVD"""

  def __init__(self):
    self.U_ = None
    self.S_ = None
    self.V_ = None

  def computeThin(self, A):
    self.U_, self.S_, self.V_ = _scipyla.svd(A.data(), full_matrices=False)

  def viewSingValues(self):
    return self.S_

  def viewLeftSingVectorsLocal(self):
    return self.U_

  def viewRightSingVectorsT(self):
    return self.V_.T


class OnNodePyQR:
  """Fallback for QR"""

  def __init__(self):
    self.Q_ = None
    self.R_ = None

  def computeThinOutOfPlace(self, A):
    self.Q_, self.R_ = _scipyla.qr(a, mode='economic')

  def viewR(self):
    return self.R_

  def viewQLocal(self):
    return self.Q_


try:
  from ._linalg import Svd, Tsqr, PseudoInverse
except ImportError:
  Svd  = OnNodePySvd
  Tsqr = OnNodePyQR
