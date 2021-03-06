
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla
from pressiotools.io.array_read import *
from pressiotools.levscores import *

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# all the _fnc are implementation details
# check the end for the actual publicly exposed function
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def _processYamlDictionary(yamlDic):
  # dataDir is not an entry specified in the real yaml file
  # so you won't find it inside the inputs-templates/hypred-levscore-template.yaml
  # because it is passed to driver script as cmd line arg
  # and inserted in the dic by the driver script
  dataDir      = yamlDic["ResidualBasis"]["dataDir"]

  fileRootName = yamlDic["ResidualBasis"]["file-root-name"]
  fileFmt      = yamlDic["ResidualBasis"]["format"]
  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: residual-basis-format must be 'binary' or 'ascii'.")

  nCols          = yamlDic["ResidualBasis"]["num-columns"]
  dofsPerMnode   = yamlDic["ResidualBasis"]["dofs-per-mesh-node"]
  numGlobalSamps = yamlDic["SampleMesh"]["num-sample-mesh-nodes"]
  levScoreBeta   = yamlDic["SampleMesh"]["leverage-score-beta"]
  outDir         = yamlDic["SampleMesh"]["outDir"]

  return {"dataDir"        : dataDir,
          "fileRootName"   : fileRootName,
          "isBinary"       : isBinary,
          "nCols"          : nCols,
          "dofsPerMnode"   : dofsPerMnode,
          "numGlobalSamps" : numGlobalSamps,
          "levScoreBeta"   : levScoreBeta,
          "outDir"         : outDir}

#-----------------------------------------------------------------
def _checkInputsValidity(dic):
  if dic["nCols"] < 1:
    sys.exit("unsupported input: ResidualBasis/num-columns needs to be greater than 0")

  if dic["dofsPerMnode"] < 1:
    sys.exit("unsupported input: ResidualBasis/dofs-per-mesh-node needs to be greater than 0")

  if dic["numGlobalSamps"] < 1:
    sys.exit("unsupported input: SampleMesh/num-sample-mesh-nodes needs to be greater than 0")

  if (dic["levScoreBeta"] < 0) or (dic["levScoreBeta"] > 1):
    sys.exit("SampleMesh/leverage-score-beta need to be a decimal between 0.0 and 1.0")

#-----------------------------------------------------------------
def _readData(dic, comm=None):
  dataDir      = dic["dataDir"]
  rootFileName = dic["fileRootName"]
  nCols        = dic["nCols"]
  isBinary     = dic["isBinary"]

  if comm is not None and comm.Get_size() > 1:
    fileName = dataDir + "/" + rootFileName
    psi = read_array_distributed(comm, fileName, nCols, isBinary)
    return psi
  else:
    fileName = "{}.{}.{:0{width}d}".format(rootFileName,1,0,width=1)
    psi0 = read_array(dataDir + "/" + fileName, nCols, isBinary)
    psi  = ptla.MultiVector(psi0)
    return psi

#-----------------------------------------------------------------
def _mapLocalToGlobalIndices(dofsPerMnode, myNumRows, comm=None):
  # assume that global indices are:
  # i_rank * n_node_per_rank + i_array_local, i_rank < n_ranks

  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()
  nRanks = 1 if nullComm else comm.Get_size()

  myNumMeshNodes   = int(myNumRows/dofsPerMnode)
  senddata         = myNumMeshNodes*np.ones(nRanks, dtype=np.int)
  meshNodesPerRank = np.empty(nRanks,dtype=np.int)
  if not nullComm:
    comm.Alltoall(senddata, meshNodesPerRank)

  globalInds = np.arange(myNumMeshNodes) + np.sum(meshNodesPerRank[:rank])

  if np.any(globalInds < 0): sys.exit("Global Indices must be positive")
  return globalInds

#-----------------------------------------------------------------
def _writeResultsToFile(outDir, mySampleMeshNodes, myPmfAllDofs, comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()

  if comm is not None:
    # gather at rank 0
    sampleMeshNodes = comm.gather(mySampleMeshNodes, root=0)
    pmfAllDofs = comm.gather(myPmfAllDofs, root=0)
    if rank==0:
      sampleMeshNodes = np.concatenate(sampleMeshNodes)
      np.savetxt(outDir+"/sampleMeshNodes.txt", sampleMeshNodes, fmt="%d")
      pmfAllDofs = np.concatenate(pmfAllDofs)
      np.savetxt(outDir+"/sampleMeshPMF.txt",pmfAllDofs)
  else:
    np.savetxt(outDir+"/sampleMeshNodes.txt", mySampleMeshNodes, fmt="%d")
    np.savetxt(outDir+"/sampleMeshPMF.txt", myPmfAllDofs)

#-----------------------------------------------------------------
def _computeLeverageScoresBasedSMImpl(psi,
                                      dofsPerMnode,
                                      beta,
                                      numGlobalSamples,
                                      comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()
  nRanks = 1 if nullComm else comm.Get_size()

  # find num rows
  myNumRows = psi.extentLocal(0)

  # mapping from local to global mesh indices
  globalInds = _mapLocalToGlobalIndices(dofsPerMnode, myNumRows, comm)

  # compute leverage scores
  l_scores = ptla.Vector(myNumRows)
  leverageScores(l_scores, psi.data())
  #print("levscores:\n",l_scores.data())

  # compute PMF from leverage scores on each mesh node
  pmf, myPmfAllDofs = computePmf(l_scores, dofsPerMnode, beta, comm)
  #print("PMF on rank {}:\n".format(rank),pmf.data())

  # sample PMF
  mySampleMeshNodes = samplePmf(pmf, numGlobalSamples, comm)

  if len(mySampleMeshNodes)>0:
    # remove duplicate samples
    mySampleMeshNodes = np.unique(mySampleMeshNodes)
    # map sample mesh nodes to global indices
    mySampleMeshNodes = globalInds[mySampleMeshNodes]

  return mySampleMeshNodes, myPmfAllDofs

#-----------------------------------------------------------------
def _computeLeverageScoresBasedSampleMeshIndicesReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  # read data matrix
  psi = _readData(dic, comm)

  # compute
  dofsPerMnode     = dic["dofsPerMnode"]
  numGlobalSamples = dic["numGlobalSamps"]
  sampleMeshNodes, pmfAllDofs = _computeLeverageScoresBasedSMImpl(psi,
                                                                  dofsPerMnode,
                                                                  dic["levScoreBeta"],
                                                                  numGlobalSamples,
                                                                  comm)

  if comm is not None:
    rank = comm.Get_rank()
    print("Sample mesh node indices on rank {}:".format(rank), sampleMeshNodes)
  else:
    print("Sample mesh node indices:", sampleMeshNodes)

  # write sample mesh nodes to disk
  _writeResultsToFile(dic["outDir"], sampleMeshNodes, pmfAllDofs, comm)

#-----------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#-----------------------------------------------------------------
def computeNodes(**args):
  if len(args) == 2:
    assert('communicator' in args.keys())
    assert('yamldic' in args.keys())
    _computeLeverageScoresBasedSampleMeshIndicesReadYaml(args['communicator'],
                                                         args['yamldic'])

  elif len(args) >= 3:
    assert('matrix'     in args.keys())
    assert('numSamples' in args.keys())
    dofsPerMeshNode = 1   if 'dofsPerMeshNode' not in args.keys() else args['dofsPerMeshNode']
    beta            = 0.5 if 'beta' not in args.keys() else args['beta']
    comm            = None if 'communicator' not in args.keys() else args['communicator']

    return _computeLeverageScoresBasedSMImpl(args['matrix'],
                                             dofsPerMeshNode,
                                             beta,
                                             args['numSamples'],
                                             comm)
