
import argparse, sys
import numpy as np
import pressiotools.linalg as ptla
import time
np.set_printoptions(linewidth=140)

#-----------------------------------------------
def createLocalMatrix(comm, N, M, distribute):
  rank = comm.Get_rank()
  size = comm.Get_size()

  if distribute=="uniform":
    # compute num of rows for each rank
    myNominalNumRows = int(N/size)
    myNumRows = myNominalNumRows
    # the last rank should account for reminder
    if (rank==size-1):
      myNumRows += N % size

    # fill each piece by reading a file or some other way.
    # Note: the layout MUST be fortran such that
    # pressiotools can view it without copying data.
    return np.asfortranarray(np.random.rand(myNumRows,M))
  else:
    sys.exit("Case not implemented yet")


#-----------------------------------------------
def computeSvd(comm, A):
  svdO = ptla.Svd()
  svdO.computeThin(A)
  U = svdO.viewLeftSingVectorsLocal()
  S = svdO.viewSingValues()
  VT = svdO.viewRightSingVectorsT()
  # check sing values are not NaNs
  assert(not np.isnan(S).all())

#-----------------------------------------------
def run(comm, N, M, distribute):
  rank = comm.Get_rank()

  Alocal = createLocalMatrix(comm, N, M, distribute)
  A = ptla.MultiVector(Alocal)
  assert(A.extentGlobal(0) == N)
  assert(A.extentLocal(0) == Alocal.shape[0])
  assert(A.extentGlobal(1) == M)

  # compute SVD
  if rank==0: print("Starting SVD")

  # do warm up run (not timed)
  computeSvd(comm, A)

  # time
  t0 = time.time()
  computeSvd(comm, A)
  t1 = time.time()
  myT = t1-t0
  if rank==0: print("Finished SVD")

  if rank==0: print("Process timings")
  all_ts = comm.gather(myT, root=0)

  if rank==0:
    minT = np.min(all_ts)
    maxT = np.max(all_ts)
    avgT = np.mean(all_ts)
    print("mean = {}, max = {}, min = {}".format(avgT, minT, maxT))


#-------------------------
if __name__ == '__main__':
#-------------------------
  '''
  run with:
    mpirun -n np python3 main_svd.py --n <glob-rows> --m <cols> -d <string>
  '''
  parser = argparse.ArgumentParser(description="Svd performance test")

  parser.add_argument(
    '-n', '--n',
    help="Number of global rows",
    dest="N",
    type=int,
    required=True)

  parser.add_argument(
    '-m', '--m',
    help="Number of cols",
    dest="M",
    type=int,
    required=True)

  parser.add_argument(
    '-d', '--d',
    help="How to distribute the rows over ranks",
    dest="distribute",
    default="uniform")

  args = parser.parse_args()

  # mpi4py initializes mpi when the module is imported so
  # it matters where you put it
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  run(comm, args.N, args.M, args.distribute)
