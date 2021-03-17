
import math
import numpy as np
import pathlib, sys
from pressiotools import linalg as ptla
from pressiotools.levscores import *

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def check_levscores(psi, comm):
  rank = comm.Get_rank()

  l_scores = ptla.Vector(myNumRows)
  leverageScores(l_scores, psi.data())
  print(rank,l_scores.data())

  # check leverage scores against truth values
  if rank == 0:
    l_scores_truth_lcl = np.array([0.49781725640571,
                                   0.44080349621974,
                                   2.35742451093826,
                                   1.15014097270538,
                                   0.07248648067393,
                                   1.43282248176162,
                                   2.74311764577994,
                                   0.41537324804313])
  elif rank == 1:
    l_scores_truth_lcl = np.array([1.43317716202619,
                                   1.15354421701282,
                                   1.53902367978074,
                                   1.03977559500014,
                                   0.81731094949638,
                                   0.63384371442751,
                                   1.34991091068034,
                                   1.40177954963967])
  elif rank == 2:
    l_scores_truth_lcl = np.array([0.91861337644567,
                                   1.25984161397013,
                                   2.49581934130582,
                                   1.38875957598056,
                                   1.00576547370327,
                                   1.2727751032992,
                                   0.73573001951863,
                                   1.30823090215113])

  assert(np.all(np.abs(l_scores.data() - l_scores_truth_lcl) < tol ))

  r = l_scores.sumGlobal()
  print("r = ", r)
  assert( math.isclose(r, 28.863887276965883) )

def check_pmf(psi, comm):
  rank = comm.Get_rank()

  l_scores = ptla.Vector(psi.extentLocal(0))
  leverageScores(l_scores, psi.data())

  l_scores_pmf,_ = computePmf(l_scores, 2, 0.5, comm)
  print(rank,l_scores_pmf.data())

  # check PMF against truth values
  if rank == 0:
    l_scores_pmf_truth_lcl = np.array([0.05792609741518,
                                       0.10242711535444,
                                       0.06774265823342,
                                       0.09638020652212])
  elif rank == 1:
    l_scores_pmf_truth_lcl = np.array([0.08647562386321,
                                       0.08633839175407,
                                       0.06680456042989,
                                       0.08933333113767])
  elif rank == 2:
    l_scores_pmf_truth_lcl = np.array([0.07940335420137,
                                       0.10895799995126,
                                       0.08113710519663,
                                       0.07707355594074])
  else:
    l_scores_pmf_truth_lcl = None

  assert(np.all(np.abs(l_scores_pmf.data() - l_scores_pmf_truth_lcl) < tol ))


if __name__ == '__main__':
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  assert(comm.Get_size() == 3)

  np.random.seed(312367)
  psi0 = np.asfortranarray(np.random.rand(24,4))
  if rank==0:
    print(psi0)
    print("---------\n")

  myNumRows = 8
  myStartRow = rank*myNumRows
  psi = ptla.MultiVector(psi0[myStartRow:myStartRow+myNumRows, :])

  check_levscores(psi, comm)
  check_pmf(psi, comm)
