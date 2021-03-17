
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla

np.set_printoptions(linewidth=140)

def tsqr_run(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)
  A = np.asfortranarray(np.random.rand(10,4))
  print(A)
  Q1, R1 = np.linalg.qr(A)
  print(Q1)

  A1 = ptla.MultiVector(A)
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)
  R = qrO.viewR()
  Q = qrO.viewQLocal()
  assert(np.allclose(np.abs(Q1),np.abs(Q), atol=1e-10))
  assert(np.allclose(np.abs(R1),np.abs(R), atol=1e-10))
  # print(R.flags['OWNDATA'])
  # R_add = R.__array_interface__['data'][0]
  # print("R: ", R_add)

def tsqr_fancy_indexing(comm):
  rank = comm.Get_rank()
  np.random.seed(312367)
  A = np.asfortranarray(np.random.rand(10,4))

  rows = [2,3,4,8,9]
  Asub = np.asfortranarray(A[rows,:])
  # the fancy indexing makes a copy
  assert(not Asub.base is A)
  assert(Asub.flags['F_CONTIGUOUS'])
  Q1, R1 = np.linalg.qr(Asub)
  print(Q1)

  A1 = ptla.MultiVector(Asub)
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)
  R = qrO.viewR()
  Q = qrO.viewQLocal()
  print(Q)
  assert(np.allclose(np.abs(Q1),np.abs(Q), atol=1e-10))
  assert(np.allclose(np.abs(R1),np.abs(R), atol=1e-10))

if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 1)

  tsqr_run(comm)
  tsqr_fancy_indexing(comm)
