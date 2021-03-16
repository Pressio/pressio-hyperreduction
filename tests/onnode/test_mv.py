
import numpy as np
from pressiotools import linalg as ptla

def mv_extent():
  nrows = 5
  d = np.zeros((nrows,3), order='F')
  A = ptla.MultiVector(d)
  assert(A.extent(0)==5)
  assert(A.extentLocal(0)==5)
  assert(A.extentGlobal(0)==5)
  assert(A.extent(1)==3)
  assert(A.extentLocal(1)==3)
  assert(A.extentGlobal(1)==3)

def mv_content():
  d = np.ones((5,3), order='F')
  d *= 1.
  A = ptla.MultiVector(d)
  nativeView = A.data()
  gold = np.ones((5,3))
  assert(np.allclose(gold, nativeView))

def mv_content1():
  d = np.ones((5,3), order='F')
  d *= 1.
  A = ptla.MultiVector(d)
  nativeView = A.data()
  gold = np.ones((5,3))
  assert(np.allclose(gold, nativeView))

if __name__ == '__main__':
  mv_extent()
  mv_content()
  mv_content1()
