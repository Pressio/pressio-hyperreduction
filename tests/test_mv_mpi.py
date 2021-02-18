
import pathlib, sys
import numpy as np

import pressiotools as pt

def mv_extent(comm):
  rank = comm.Get_rank()
  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros((nrows,3), order='F')
  A = pt.MultiVector(d)

  if (rank==0): assert(A.extentLocal(0)==5)
  if (rank==1): assert(A.extentLocal(0)==6)
  if (rank==2): assert(A.extentLocal(0)==4)
  if (rank==3): assert(A.extentLocal(0)==6)

  assert(A.extentLocal(1)==3)

def mv_extent_global(comm):
  rank = comm.Get_rank()
  if (rank==0): nrows = 5
  if (rank==1): nrows = 6
  if (rank==2): nrows = 4
  if (rank==3): nrows = 6

  d = np.zeros((nrows,3), order='F')
  A = pt.MultiVector(d)
  assert(A.extentGlobal(0)==21)
  assert(A.extentGlobal(1)==3)


def mv_content(comm):
  rank = comm.Get_rank()
  d = np.ones((5,3), order='F')
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = pt.MultiVector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones((5,3))
  if (rank==1): gold = np.ones((5,3))*2.
  if (rank==2): gold = np.ones((5,3))*3.
  if (rank==3): gold = np.ones((5,3))*4.
  assert(np.allclose(gold, nativeView))

def mv_content1(comm):
  rank = comm.Get_rank()
  d = np.ones((5,3), order='F')
  if (rank==0): d *= 1.
  if (rank==1): d *= 2.
  if (rank==2): d *= 3.
  if (rank==3): d *= 4.

  A = pt.MultiVector(d)
  nativeView = A.data()

  if (rank==0): gold = np.ones((5,3))
  if (rank==1): gold = np.ones((5,3))*2.
  if (rank==2): gold = np.ones((5,3))*3.
  if (rank==3): gold = np.ones((5,3))*4.
  assert(np.allclose(gold, nativeView))

if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 4)

  mv_extent(comm)
  mv_extent_global(comm)
  mv_content(comm)
  mv_content1(comm)
