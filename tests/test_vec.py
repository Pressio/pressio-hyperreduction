
import numpy as np
import pressiotools.linalg as ptla

def vec_constr():
  nrows = 5
  A = ptla.Vector(nrows)
  assert(A.extent()==5)
  Av = A.data()
  assert(np.allclose(Av, np.zeros(nrows)))

def vec_extent():
  nrows = 5
  d = np.zeros(nrows)
  A = ptla.Vector(d)
  assert(A.extent()==5)
  assert(A.extentLocal()==5)
  assert(A.extentGlobal()==5)

def vec_content():
  d = np.ones(5)
  d *= 1.
  A = ptla.Vector(d)
  nativeView = A.data()
  gold = np.ones(5)
  assert(np.allclose(gold, nativeView))

def vec_content1():
  d = np.ones(5)
  d *= 1.
  A = ptla.Vector(d)
  nativeView = A.data()
  gold = np.ones(5)
  assert(np.allclose(gold, nativeView))

def vec_address(comm):
  rank = comm.Get_rank()
  vPy = np.ones(5)
  addPy = hex(vPy.__array_interface__['data'][0])
  v = ptla.Vector(vPy)
  addV = v.address()
  print(rank, addPy, hex(addV))
  assert( addPy == hex(addV) )

if __name__ == '__main__':
  vec_constr()
  vec_extent()
  vec_content()
  vec_content1()
  vec_address()
