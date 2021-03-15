
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla

def vec_constr(comm):
  rank = comm.Get_rank()

  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  A = ptla.Vector(nrows)
  if (rank==0): assert(A.extentLocal()==5)
  if (rank==1): assert(A.extentLocal()==6)
  if (rank==2): assert(A.extentLocal()==4)
  if (rank==3): assert(A.extentLocal()==6)

  Av = A.data()
  assert(np.allclose(Av, np.zeros(nrows)))


def vec_extent(comm):
  rank = comm.Get_rank()

  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros(nrows)
  A = ptla.Vector(d)

  if (rank==0): assert(A.extentLocal()==5)
  if (rank==1): assert(A.extentLocal()==6)
  if (rank==2): assert(A.extentLocal()==4)
  if (rank==3): assert(A.extentLocal()==6)

def vec_extent_global(comm):
  rank = comm.Get_rank()

  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros(nrows)
  A = ptla.Vector(d)
  assert(A.extentGlobal()==21)


def vec_content(comm):
  rank = comm.Get_rank()

  d = np.ones(5)
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = ptla.Vector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones(5)
  if (rank==1): gold = np.ones(5)*2.
  if (rank==2): gold = np.ones(5)*3.
  if (rank==3): gold = np.ones(5)*4.
  assert(np.allclose(gold, nativeView))

def vec_content1(comm):
  rank = comm.Get_rank()

  d = np.ones(5)
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = ptla.Vector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones(5)
  if (rank==1): gold = np.ones(5)*2.
  if (rank==2): gold = np.ones(5)*3.
  if (rank==3): gold = np.ones(5)*4.
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
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 4)

  vec_constr(comm)
  vec_extent(comm)
  vec_extent_global(comm)
  vec_content(comm)
  vec_content1(comm)
  vec_address(comm)
