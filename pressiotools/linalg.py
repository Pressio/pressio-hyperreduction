
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

try:
  from ._linalg import Svd, Tsqr, PseudoInverse
except ImportError:
  Svd  = OnNodePySvd
  #Tsqr = OnNodePyQr
