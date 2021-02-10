
import pathlib, sys

file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")
sys.path.append(str(file_path) + "/../srcpy")

import argparse
import numpy as np
from mpi4py import MPI
import pressiotools as pt
import scipy.linalg as la
import array_io
from yaml_parser import yaml_read

np.set_printoptions(linewidth=140)

def leverageScores(ls, A):
  ls.data()[:] = np.einsum("ij,ij->i", A, A)

def leverageScorePMF(l_scores,pmf_blend):
  # returns numpy vector of pmf entries corresponding to vector l_scores
  r = l_scores.sumGlobal() # should be equal to number of rows for an orthonormal matrix
  return pmf_blend * l_scores.data() / r + (1.0-pmf_blend) / l_scores.extentGlobal()

def computeBlockPMF(l_scores, dofsPerMnode, pmf_blend=0.5):
  # compute pmf for each mesh node by summing over mesh DoF pmf entries
  pmf_allDofs = leverageScorePMF(l_scores, pmf_blend)

  # sum up probability mass for each mesh node
  myNumMeshNodes = int(l_scores.extentLocal()/dofsPerMnode)
  pmf_meshDofs = pt.Vector(myNumMeshNodes)
  pmf_meshDofs.data()[:] = np.zeros(pmf_meshDofs.extentLocal())

  # sum up probabilities for each mesh DoF
  # assumes that nodal quantities are fastest index
  # e.g. mass, momentum, energy for each node are grouped together in residual vector
  pmf_allDofs = np.reshape(pmf_allDofs,(myNumMeshNodes,dofsPerMnode))

  for i in range(dofsPerMnode):
    pmf_meshDofs.data()[:] = pmf_meshDofs.data() + pmf_allDofs[:,i]

  return pmf_meshDofs, pmf_allDofs


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

def run(comm):
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Get inputs
  parser = argparse.ArgumentParser(description="Generate sample mesh \
indices using random sampling from a probability mass function based on leverage scores")
  parser.add_argument('--input',help="full path to (and including) yaml input file. ",required=True)
  args = parser.parse_args()

  yaml_in = yaml_read(args.input)

  # read user inputs
  fileName = yaml_in["ResidualBasis"]["file-root-name"]
  fileFmt = yaml_in["ResidualBasis"]["format"]

  isBinary=False
  if fileFmt=="binary":
    isBinary=True
    if rank==0:
      print("Reading from binary files")
  elif fileFmt=="ascii":
    if rank==0:
      print("Reading from ascii files")
  else:
    sys.exit("Unsupported file format: residual-basis-format must be 'binary' or 'ascii'.")


  nCols = yaml_in["ResidualBasis"]["num-columns"]
  dofsPerMnode = yaml_in["ResidualBasis"]["dofs-per-mesh-node"]

  numGlobalSamps = yaml_in["SampleMesh"]["num-sample-mesh-nodes"]
  levScoreBeta = yaml_in["SampleMesh"]["leverage-score-beta"]

  # read matrix
  psi = array_io.read_array_distributed(comm, fileName, nCols, isBinary)
  myNumRows = psi.extentLocal(0)

  # mapping from local to global mesh indicies
  # if none is provided, assume that global indices are:
  # i_rank * n_node_per_rank + i_array_local, i_rank < n_ranks
  myNumMeshNodes = int(myNumRows/dofsPerMnode)
  senddata = myNumMeshNodes*np.ones(size,dtype=np.int)
  meshNodesPerRank = np.empty(size,dtype=np.int)
  comm.Alltoall(senddata,meshNodesPerRank)

  globalInds = np.arange(myNumMeshNodes) + np.sum(meshNodesPerRank[:rank])
  #print(rank,globalInds)

  ########################################################

  # compute leverage scores
  l_scores = pt.Vector(myNumRows)
  leverageScores(l_scores, psi.data())

  # compute PMF from leverage scores on each mesh node
  pmf,mypmfAllDofs = computeBlockPMF(l_scores,dofsPerMnode,pmf_blend=levScoreBeta)

  #print("PMF on rank {}:".format(rank),pmf.data())

  # sample PMF
  mySampleMeshNodes = samplePMF(comm, pmf, numGlobalSamps)

  if len(mySampleMeshNodes)>0:
    # remove duplicate samples
    mySampleMeshNodes = np.unique(mySampleMeshNodes)

    # map sample mesh nodes to global indices
    mySampleMeshNodes = globalInds[mySampleMeshNodes]

  #print("Sample mesh node indices on rank {}:".format(rank),mySampleMeshNodes)

  # write sample mesh nodes to disk
  # gather array at rank 0
  sampleMeshNodes = comm.gather(mySampleMeshNodes, root=0)
  # write to file with numpy
  if rank==0:
    sampleMeshNodes = np.concatenate(sampleMeshNodes)
    np.savetxt("sampleMeshNodes.txt",sampleMeshNodes,fmt="%d")

  # write pmfs for each sample mesh DoF to disk
  pmfAllDofs = comm.gather(mypmfAllDofs, root=0)
  if rank==0:
    pmfAllDofs = np.concatenate(pmfAllDofs)
    np.savetxt("sampleMeshPMF.txt",pmfAllDofs)


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  run()
