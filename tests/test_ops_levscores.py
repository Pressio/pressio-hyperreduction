
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la

np.set_printoptions(linewidth=140)

def leverageScores(ls, A):
  ls.data()[:] = np.einsum("ij,ij->i", A, A)

def run():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)

  # random distributed psi by selecting
  # rows of a random matrix
  psi0 = np.asfortranarray(np.random.rand(15,4))
  if rank==0:
    print(psi0)
    print("---------\n")

  myNumRows = 5
  myStartRow = rank*myNumRows
  psi = pt.MultiVector(psi0[myStartRow:myStartRow+5, :])

  l_scores = pt.Vector(myNumRows)
  leverageScores(l_scores, psi.data())
  print(l_scores.data())

  r = l_scores.sumGlobal()
  print(r)

  l_scores_pmf = pt.Vector(myNumRows)
  for i in range(myNumRows):
    l_scores_pmf.data()[i] = 0.5*l_scores.data()[i]/r + 0.5/15.
  print(l_scores_pmf.data())

if __name__ == '__main__':
  run()
