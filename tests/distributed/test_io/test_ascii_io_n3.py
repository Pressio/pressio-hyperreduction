
import numpy as np
from pressiotools import linalg as ptla
from pressiotools.io.array_read import *
from pressiotools.io.array_write import *

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run(comm):
  rank = comm.Get_rank()
  np.random.seed(223)

  # random distributed matrix by selecting
  # rows of a random matrix
  numCols = 4
  mat0 = np.asfortranarray(np.random.rand(15,numCols))
  vec0 = mat0[:,0]

  myNumRows = 5
  myStartRow = rank*myNumRows
  mat = ptla.MultiVector(mat0[myStartRow:myStartRow+myNumRows, :])
  vec = ptla.Vector(vec0[myStartRow:myStartRow+myNumRows])

  if rank==0:
    print(mat0)
    print("---------\n")

  # serial read
  if rank==0:
    mat0_gold = read_array("data_ascii/matrix.txt.gold",numCols,isBinary=False)
    assert(np.all(np.abs(mat0_gold - mat0) < tol))

    vec0_gold = read_array("data_ascii/vector.txt.gold",1,isBinary=False)
    assert(np.all(np.abs(vec0_gold - vec0) < tol))

  # distributed read
  mat_gold = read_array_distributed(comm, "data_ascii/matrix.txt.gold",numCols,isBinary=False)
  assert(np.all(np.abs(mat_gold.data() - mat.data()) < tol))

  vec_gold = read_array_distributed(comm, "data_ascii/vector.txt.gold",1,isBinary=False)
  assert(np.all(np.abs(vec_gold.data() - vec.data()) < tol))

  # serial write
  if rank==0:
    write_array(mat0,"data_ascii/matrix.txt",isBinary=False)
    mat0_in = read_array("data_ascii/matrix.txt",numCols,isBinary=False)
    assert(np.all(np.abs(mat0_in - mat0) < tol))

    write_array(vec0,"data_ascii/vector.txt",isBinary=False)
    vec0_in = read_array("data_ascii/vector.txt",1,isBinary=False)
    assert(np.all(np.abs(vec0_in - vec0) < tol))

  # distributed write
  write_array_distributed(comm, mat,"data_ascii/matrix.txt",isBinary=False)
  mat_in = read_array_distributed(comm, "data_ascii/matrix.txt",numCols,isBinary=False)
  assert(np.all(np.abs(mat_in.data() - mat.data()) < tol))

  write_array_distributed(comm, vec,"data_ascii/vector.txt",isBinary=False)
  vec_in = read_array_distributed(comm, "data_ascii/vector.txt",1,isBinary=False)
  assert(np.all(np.abs(vec_in.data() - vec.data()) < tol))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert(comm.Get_size() == 3)
  run(comm)
