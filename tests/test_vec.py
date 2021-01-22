
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt

def vec_extent():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 4)

  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros(nrows)
  A = pt.Vector(d)

  if (rank==0): assert(A.extentLocal()==5)
  if (rank==1): assert(A.extentLocal()==6)
  if (rank==2): assert(A.extentLocal()==4)
  if (rank==3): assert(A.extentLocal()==6)

def vec_extent_global():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 4)

  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros(nrows)
  A = pt.Vector(d)
  assert(A.extentGlobal()==21)


def vec_content():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 4)

  d = np.ones(5)
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = pt.Vector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones(5)
  if (rank==1): gold = np.ones(5)*2.
  if (rank==2): gold = np.ones(5)*3.
  if (rank==3): gold = np.ones(5)*4.
  assert(np.allclose(gold, nativeView))

def vec_content1():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 4)

  d = np.ones(5)
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = pt.Vector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones(5)
  if (rank==1): gold = np.ones(5)*2.
  if (rank==2): gold = np.ones(5)*3.
  if (rank==3): gold = np.ones(5)*4.
  assert(np.allclose(gold, nativeView))


if __name__ == '__main__':
  vec_extent()
  vec_extent_global()
  vec_content()
  vec_content1()
