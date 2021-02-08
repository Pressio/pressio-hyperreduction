
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la
import array_io

np.set_printoptions(linewidth=140)

def leverageScores(ls, A):
  ls.data()[:] = np.einsum("ij,ij->i", A, A)

def computePMF(l_scores, dofsPerMnode, pmf_blend=0.5):
  r = l_scores.sumGlobal() # should be equal to number of rows for an orthonormal matrix
  pmf_allDofs = pmf_blend * l_scores.data() / r + pmf_blend / l_scores.extentGlobal()

  # sum up probability mass for each mesh node
  myNumMeshNodes = int(l_scores.extentLocal()/dofsPerMnode)
  pmf_meshDofs = pt.Vector(myNumMeshNodes)
  pmf_meshDofs.data()[:] = np.zeros(pmf_meshDofs.extentLocal())

  # sum up probabilities for each mesh DoF
  # assumes that nodal quantities are fastest index 
  # e.g. mass, momentum, energy for each node are grouped together in residual vector
  for i in range(dofsPerMnode):
    pmf_meshDofs.data()[:] = pmf_meshDofs.data() + pmf_allDofs[i::dofsPerMnode]

  return pmf_meshDofs

def samplePMF(comm,pmf,numSampsGlobal):
  # compute total probability mass on this rank
  myProbMass = np.sum(pmf.data())

  # gather each rank's pmf on rank zero
  rank = comm.Get_rank()
  size = comm.Get_size()
  global_pmf = comm.gather(myProbMass, root=0)

  # determine number of samples each rank
  if rank == 0:
    ranks = np.arange(size)
    rankSamples = np.random.choice(ranks, numSampsGlobal, p=global_pmf)
    sampArray = [0] * size
    for i in range( numSampsGlobal ):
        sampArray[rankSamples[i]] += 1
    print("Rank PMF:",global_pmf)
    print("Samples on each Rank:",sampArray)
  else:
    sampArray = None
  myNumSamps = comm.scatter(sampArray, root=0)

  # sample pmf locally
  local_pmf = pmf.data() / myProbMass
  localInds = np.arange(pmf.extentLocal())
  #print("Local PMF on rank {}:".format(rank),local_pmf)

  return np.random.choice(localInds, myNumSamps, p=local_pmf)

def run():
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # read user inputs
  # TODO yaml parser
  numGlobalSamps = 8
  dofsPerMnode = 1
  fileName = "matrix.bin"
  nCols = 4
  isBinary = True

  # read matrix
  psi = array_io.read_array_distributed(fileName,4,isBinary)
  myNumRows = psi.extentLocal(0)
  
  # TODO mapping from local to global mesh indicies
  # if none is provided, assume that global indices are: 
  # i_rank * n_node_per_rank + i_array_local, i_rank < n_ranks
  senddata = myNumRows*np.ones(size,dtype=np.int)
  rowsPerRank = np.empty(size,dtype=np.int)
  comm.Alltoall(senddata,rowsPerRank)

  globalInds = np.arange(myNumRows) + np.sum(rowsPerRank[:rank]) 
  print(rank,globalInds)

  ########################################################

  # compute leverage scores
  l_scores = pt.Vector(myNumRows)
  leverageScores(l_scores, psi.data())

  # compute PMF from leverage scores on each mesh node
  pmf = computePMF(l_scores,dofsPerMnode)

  #print("PMF on rank {}:".format(rank),pmf.data())

  # sample PMF
  mySampleMeshNodes = samplePMF(comm, pmf, numGlobalSamps)

  if len(mySampleMeshNodes)>0:
    # remove duplicate samples
    mySampleMeshNodes = np.unique(mySampleMeshNodes)

    # map sample mesh nodes to global indices
    mySampleMeshNodes = globalInds[mySampleMeshNodes]

  print("Sample mesh node indices on rank {}:".format(rank),mySampleMeshNodes)
  
  # write sample mesh nodes to disk
  # gather array at rank 0
  sampleMeshNodes = comm.gather(mySampleMeshNodes, root=0)
  # write to file with numpy
  if rank==0:
    sampleMeshNodes = np.concatenate(sampleMeshNodes)
    np.savetxt("sampleMeshNodes.txt",sampleMeshNodes,fmt="%d")

  # write sampling matrix to disk
  # TODO file I/O
  # gather pmf for each DoF on each sample mesh node
  # write to file with numpy

if __name__ == '__main__':
  run()
