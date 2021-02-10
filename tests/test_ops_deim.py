
import pathlib, sys
import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la

np.set_printoptions(linewidth=140)


def applyMask(A, ind):
  if (A.ndim ==2):
    return A[ind, :]
  else:
    return A[ind]

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
  myStartRow = rank*5
  psi = pt.MultiVector(psi0[myStartRow:myStartRow+5, :])

  sm_nodes = []
  if rank==1: sm_nodes.append(2)

  for i_vec in range(1,psi.extentLocal(1)):
    print("i_vec = {}".format(i_vec))

    # extract subset of residual basis vectors
    U = psi.data()[:,:i_vec]
    psi_i = psi.data()[:,i_vec]
    if rank==0: print(U)

    for i_node in range(10):
      # mask residual basis vector collections
      Umask = applyMask(U, sm_nodes)
      print("rank = ", rank, Umask)
      psi_i_mask = applyMask(psi_i, sm_nodes)

      M = pt.MultiVector(Umask)
      piO = pt.pinv()
      piO.compute(M)

      sys.exit()
      #       # compute gappy reconstruction
      #       psi_hat = applyPseudoInverseToVector(Umask, psi_i_mask)
      #       psi_i_approx = computeMatrixVectorProduct(U, psi_hat )
      #       # find new mesh index to add
      #       new_node = nodeWithMax( abs(psi_i - psi_i_approx) )
      #       # update mask with new index
      #       mask_obj.updateMask(new_node)

if __name__ == '__main__':
  run()
