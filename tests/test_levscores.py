
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()#
# this is needed to access the levscores python code
sys.path.append(str(file_path) + "/../driver-scripts")

import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la
from hypred_levscores import *

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

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
  print(rank,l_scores.data())

  # check leverage scores against truth values
  if rank == 0:
    l_scores_truth_lcl = np.array([0.49781725640571,
                                   0.44080349621974,
                                   2.35742451093826,
                                   1.15014097270538,
                                   0.07248648067393])
  elif rank == 1:
    l_scores_truth_lcl = np.array([1.43282248176162,
                                   2.74311764577994,
                                   0.41537324804313,
                                   1.43317716202619,
                                   1.15354421701282])
  elif rank == 2:
    l_scores_truth_lcl = np.array([1.53902367978074,
                                   1.03977559500014,
                                   0.81731094949638,
                                   0.63384371442751,
                                   1.34991091068034])
  else:
    l_scores_truth_lcl = None

  assert(np.all(np.abs(l_scores.data() - l_scores_truth_lcl) < tol ))


  r = l_scores.sumGlobal()
  print(r)
  assert( r - 17.076572320951822 < 1e-12)

  l_scores_pmf,_ = computeBlockPMF(l_scores,1)
  print(rank,l_scores_pmf.data())

  # check PMF against truth values
  if rank == 0:
    l_scores_pmf_truth_lcl = np.array([0.0479093632019,
                                       0.04624000710647,
                                       0.10235844172836,
                                       0.06700932378062,
                                       0.03545572883846])
  elif rank == 1:
    l_scores_pmf_truth_lcl = np.array([0.07528620463654,
                                       0.11365149069605,
                                       0.04549541247417,
                                       0.07529658963237,
                                       0.06710897036787])
  elif rank == 2:
    l_scores_pmf_truth_lcl = np.array([0.07839576304273,
                                       0.06377783869008,
                                       0.05726410041396,
                                       0.05189220166225,
                                       0.0728585637282])
  else:
    l_scores_pmf_truth_lcl = None

  assert(np.all(np.abs(l_scores_pmf.data() - l_scores_pmf_truth_lcl) < tol ))


  # compute samples
  mySampleMeshNodes = samplePMF(comm, l_scores_pmf, 10)
  print(rank,mySampleMeshNodes)

  # check samples against truth values
  if rank == 0:
    mySampleMeshNodes_truth = [2, 3, 3, 4, 2]
  elif rank == 1:
    mySampleMeshNodes_truth = [2, 0]
  elif rank == 2:
    mySampleMeshNodes_truth = [2, 0, 2]
  else:
    mySampleMeshNodes_truth = None

  assert( len(mySampleMeshNodes) == len(mySampleMeshNodes_truth))
  assert( all(mySampleMeshNodes == mySampleMeshNodes_truth))

if __name__ == '__main__':
  run()
