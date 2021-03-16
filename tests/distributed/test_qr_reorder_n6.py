
import pathlib, sys
import numpy as np
import pressiotools.linalg as ptla

np.set_printoptions(linewidth=140)

# here we want to test that the paralle tsqr also works
# when the input matrix does not satisfy the trilinos tsqr
# rule that on each rank it has to have num of rows >= num of cols

#-----------------------------------------
def tsqr_run1(comm):
  # this case tests when some ranks have too few rows of A to
  # satisfy the tsqr requirement rightaway, but overall the
  # GLOBAL num of rows is still sufficient to be redistributed
  # *uniformly* over the same comm without leaving ranks empty

  np.random.seed(3675638)
  rank = comm.Get_rank()

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(37, numCols))
  if rank==0: print(A0)

  if rank==0:   myA = A0[0:4, :]
  elif rank==1: myA = A0[4:11, :]
  elif rank==2: myA = A0[11:14, :]
  elif rank==3: myA = A0[14:24, :]
  elif rank==4: myA = A0[24:27, :]
  elif rank==5: myA = A0[27:37, :]

  A1 = ptla.MultiVector(myA)
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)

  ### verify that result is correct ###
  Q1, R1 = np.linalg.qr(A0)
  print(rank, R1)
  # extract corresponding rows on each rank
  if rank==0:   myQ = Q1[0:4, :]
  elif rank==1: myQ = Q1[4:11, :]
  elif rank==2: myQ = Q1[11:14, :]
  elif rank==3: myQ = Q1[14:24, :]
  elif rank==4: myQ = Q1[24:27, :]
  elif rank==5: myQ = Q1[27:37, :]

  # view the pressio-tools results
  R = qrO.viewR()
  Q = qrO.viewQLocal()
  # R is the replicated, so should be the same
  assert(R.shape == R1.shape)
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-12))
  # on each rank, Q is the local part so should be same as myQ
  assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-12))
  assert(myQ.shape == Q.shape)


#-----------------------------------------
def tsqr_run2(comm):
  # same as run1 above but we have one rank with zero data

  np.random.seed(3675638)
  rank = comm.Get_rank()

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(37, numCols))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0,4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 15).tolist()
  elif rank==3: locRows  = np.arange(15, 19).tolist()
  elif rank==4: locRows  = np.arange(19, 28).tolist()
  elif rank==5: locRows  = np.arange(28, 37).tolist()

  A1 = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)

  ### verify that result is correct ###
  Q1, R1 = np.linalg.qr(A0)
  #print(rank, R1)
  # extract corresponding rows on each rank
  myQ = Q1[locRows, :]

  # view the pressio-tools results
  R = qrO.viewR()
  Q = qrO.viewQLocal()
  print(rank, R.shape, Q.shape)
  # R is the replicated, so should be the same
  assert(R.shape == R1.shape)
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-12))
  # on each rank, Q is the local part so should be same as myQ
  # but on rank==1 it should be empty since rank=1 does not have data
  if rank!=1:
    assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-12))
    assert(myQ.shape == Q.shape)
  else:
    assert(Q.shape[0] == 0)


#-----------------------------------------
def tsqr_run3(comm):
  # this case tests when some ranks have too few rows of A to
  # satisfy the tsqr requirement rightaway, and overall the
  # GLOBAL num of rows is NOT sufficient to be redistributed
  # over the same comm without leaving some ranks empty

  np.random.seed(3675638)
  rank = comm.Get_rank()

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(21, numCols))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0, 4).tolist()
  elif rank==1: locRows  = np.arange(4, 11).tolist()
  elif rank==2: locRows  = np.arange(11,14).tolist()
  elif rank==3: locRows  = np.arange(14,16).tolist()
  elif rank==4: locRows  = np.arange(16,20).tolist()
  elif rank==5: locRows  = np.arange(20,21).tolist()

  A1 = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))
  print(A1.extentGlobal(0))
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)

  ### verify that result is correct ###
  Q1, R1 = np.linalg.qr(A0)
  print("SCIPY", rank, R1)
  # extract corresponding rows on each rank
  myQ = Q1[locRows, :]

  # view the pressio-tools results
  R = qrO.viewR()
  print(rank, R)
  Q = qrO.viewQLocal()
  # R is the replicated, so should be the same
  assert(R.shape == R1.shape)
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-12))
  # on each rank, Q is the local part so should be same as myQ
  assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-12))
  assert(myQ.shape == Q.shape)

#-----------------------------------------
def tsqr_run4(comm):
  # this test is like one above but we have a rank with no data

  np.random.seed(3675638)
  rank = comm.Get_rank()

  numCols = 6
  A0 = np.asfortranarray(np.random.rand(21, numCols))
  if rank==0: print(A0)

  if rank==0:   locRows  = np.arange(0, 4).tolist()
  elif rank==1: locRows  = []
  elif rank==2: locRows  = np.arange(4, 11).tolist()
  elif rank==3: locRows  = np.arange(11,14).tolist()
  elif rank==4: locRows  = np.arange(14,16).tolist()
  elif rank==5: locRows  = np.arange(16,21).tolist()

  A1 = ptla.MultiVector(np.asfortranarray(A0[locRows, :]))
  print(A1.extentGlobal(0))
  qrO = ptla.Tsqr()
  qrO.computeThinOutOfPlace(A1)

  ### verify that result is correct ###
  Q1, R1 = np.linalg.qr(A0)
  print("SCIPY", rank, R1)
  # extract corresponding rows on each rank
  myQ = Q1[locRows, :]

  # view the pressio-tools results
  R = qrO.viewR()
  print(rank, R)
  Q = qrO.viewQLocal()
  # R is the replicated, so should be the same
  assert(R.shape == R1.shape)
  assert(np.allclose(np.abs(R1), np.abs(R), atol=1e-12))
  # on each rank, Q is the local part so should be same as myQ
  if rank!=1:
    assert(np.allclose(np.abs(myQ), np.abs(Q), atol=1e-12))
    assert(myQ.shape == Q.shape)
  else:
    assert(Q.shape[0] == 0)

############
### MAIN ###
############
if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 6)
  tsqr_run1(comm)
  tsqr_run2(comm)
  tsqr_run3(comm)
  tsqr_run4(comm)
