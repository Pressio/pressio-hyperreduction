
import math
import numpy as np
import pathlib, sys
import pressiotools as pt
file_path = pathlib.Path(__file__).parent.absolute()#
# this is needed to access the levscores python code
sys.path.append(str(file_path) + "/../srcpy")
from levscores_functions import *

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def check_levscores(psi):
  l_scores = pt.Vector(psi.extentLocal(0))
  leverageScores(l_scores, psi.data())
  print(l_scores.data())

  # check leverage scores against truth values
  l_scores_truth = np.array([0.49781725640571,
                             0.44080349621974,
                             2.35742451093826,
                             1.15014097270538,
                             0.07248648067393,
                             1.43282248176162,
                             2.74311764577994,
                             0.41537324804313,
                             1.43317716202619,
                             1.15354421701282,
                             1.53902367978074,
                             1.03977559500014,
                             0.81731094949638,
                             0.63384371442751,
                             1.34991091068034,
                             1.40177954963967,
                             0.91861337644567,
                             1.25984161397013,
                             2.49581934130582,
                             1.38875957598056,
                             1.00576547370327,
                             1.2727751032992,
                             0.73573001951863,
                             1.30823090215113])

  assert(np.all(np.abs(l_scores.data() - l_scores_truth) < tol ))
  r = l_scores.sum()
  print(r)
  assert( math.isclose(r, 28.863887276965883) )

def check_pmf(psi):
  l_scores = pt.Vector(psi.extent(0))
  leverageScores(l_scores, psi.data())
  l_scores_pmf,_ = computePmf(l_scores,2)
  print(l_scores_pmf.data())

  # check PMF against truth values
  l_scores_pmf_truth = np.array([0.05792609741518,
                                 0.10242711535444,
                                 0.06774265823342,
                                 0.09638020652212,
                                 0.08647562386321,
                                 0.08633839175407,
                                 0.06680456042989,
                                 0.08933333113767,
                                 0.07940335420137,
                                 0.10895799995126,
                                 0.08113710519663,
                                 0.07707355594074])

  assert(np.all(np.abs(l_scores_pmf.data() - l_scores_pmf_truth) < tol ))


if __name__ == '__main__':
  np.random.seed(312367)
  psi0 = np.asfortranarray(np.random.rand(24,4))
  psi = pt.MultiVector(psi0)
  print(psi0)
  print("---------\n")

  check_levscores(psi)
  check_pmf(psi)
