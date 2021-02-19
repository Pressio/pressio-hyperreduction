
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla

#-----------------------------------------------------------------
def leverageScores(ls, A):
  ls.data()[:] = np.einsum("ij,ij->i", A, A)

#-----------------------------------------------------------------
def _leverageScorePmf(l_scores, pmf_blend, comm=None):
  nullComm = True if comm is None else False

  # returns numpy vector of pmf entries corresponding to vector l_scores
  # r should be equal to number of rows for an orthonormal matrix
  r = l_scores.sumGlobal()
  a = pmf_blend * l_scores.data() / r
  b = (1.0-pmf_blend) / float(l_scores.extentGlobal())
  return a+b

#-----------------------------------------------------------------
def computePmf(l_scores, dofsPerMnode, pmf_blend=0.5, comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()
  nRanks = 1 if nullComm else comm.Get_size()

  #compute pmf for each mesh node by summing over mesh DoF pmf entries
  pmf_allDofs  = _leverageScorePmf(l_scores, pmf_blend, comm)
  mylscoresExt = l_scores.extentLocal()

  # sum up probability mass for each mesh node
  myNumMeshNodes = int(mylscoresExt/dofsPerMnode)
  pmf_meshDofs   = ptla.Vector(myNumMeshNodes)
  pmf_meshDofs.data()[:] = np.zeros(myNumMeshNodes)

  # sum up probabilities for each mesh DoF
  # assumes that nodal quantities are fastest index
  # e.g. mass, momentum, energy for each node are grouped
  # together in residual vector
  pmf_allDofs = np.reshape(pmf_allDofs, (myNumMeshNodes, dofsPerMnode))

  for i in range(dofsPerMnode):
    pmf_meshDofs.data()[:] = pmf_meshDofs.data() + pmf_allDofs[:,i]

  return pmf_meshDofs, pmf_allDofs

#-----------------------------------------------------------------
def samplePmf(pmf, numSampsGlobal, comm=None):
  if comm is not None:
    # compute total probability mass on this rank
    myProbMass = np.sum(pmf.data())

    # gather each rank's pmf on rank zero
    rank = comm.Get_rank()
    nRanks = comm.Get_size()
    global_pmf = comm.gather(myProbMass, root=0)

    # determine number of samples each rank
    if rank == 0:
      ranks = np.arange(nRanks)
      rankSamples = np.random.choice(ranks, numSampsGlobal, p=global_pmf)
      sampArray = [0] * nRanks
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
  else:
    indToSampleFrom = np.arange(pmf.extentLocal())
    return np.random.choice(indToSampleFrom, numSampsGlobal, replace=True, p=pmf.data())
