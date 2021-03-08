
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla
from pressiotools.io.array_read import *
#TODO from pressiotools.sampling_matrix import *


# all the _fnc are implementation details
# check the end for the actual publicly exposed function

#-----------------------------------------------------------------
def _processYamlDictionaryForBasis(yamlDic):
  nCols        = yamlDic["num-columns"]
  dataDir      = yamlDic["dataDir"]
  fileRootName = yamlDic["file-root-name"]
  fileFmt      = yamlDic["format"]

  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"dataDir"        : dataDir,
          "fileRootName"   : fileRootName,
          "isBinary"       : isBinary,
          "nCols"          : nCols}


#-----------------------------------------------------------------
def _processYamlDictionary(yamlDic):
  dofsPerMnode   = yamlDic["FullMeshInfo"]["dofs-per-mesh-node"]
  
  # state basis info
  stateBasisDic = _processYamlDictionaryForBasis(yamlDic["StateBasis"])

  # residual basis info
  if "ResidualBasis" in yamlDic:
    residBasisDic = _processYamlDictionaryForBasis(yamlDic["ResidualBasis"])
  else:
    residBasisDic = None

  # projector matrix info
  projMatKind        = yamlDic["ProjectorMatrix"]["kind"]
  sampMatIndFileName = yamlDic["ProjectorMatrix"]["sample-mesh-indices-filename"]
  fileFmt            = yamlDic["ProjectorMatrix"]["format"]
  outDir             = yamlDic["ProjectorMatrix"]["outDir"]
  
  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"dofsPerMnode"       : dofsPerMnode,
          "StateBasis"         : stateBasisDic,
          "ResidualBasis"      : residBasisDic,
          "projMatKind"        : projMatKind,
          "sampMatIndFileName" : sampMatIndFileName,
          "isBinary"           : isBinary,
          "outDir"             : outDir}

#-----------------------------------------------------------------
def _checkInputsValidity(dic):
  if dic["dofsPerMnode"] < 1:
    sys.exit("unsupported input: ResidualBasis/dofs-per-mesh-node needs to be greater than 0")

  if dic["StateBasis"]["nCols"] < 1:
    sys.exit("unsupported input: StateBasis/nCols needs to be greater than 0")

  if dic["ResidualBasis"]["nCols"] < 1:
    sys.exit("unsupported input: ResidualBasis/nCols needs to be greater than 0")

  if (dic["projMatKind"] is "gappy-pod") and (dic["ResidualBasis"] is None):
    sys.exit("Gappy POD requires a residual basis")

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
def _createGalerkinProjectorReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  print("parsed inputs!")
  print(dic)

  ## read state basis matrix
  #phi = _readData(dic["StateBasis"], comm)

  ## Read residual basis if required or specified
  #if dic["ResidualBasis"]:
  #  psi = _readData(dic["ResidualBasis"], comm)
  #else:
  #  psi = ptla.MultiVector(phi) # deep copy

  #print("read in basis files!")

  # Read sample mesh indices
  #sample_mesh_inds = read_sample_mesh_inds(input_dict["SampleMesh"])
  
  # construct sampling matrix
  #sampling_mat = construct_sampling_matrix(sample_mesh_inds)


  # compute
  #TODO
  # This should be in a subroutine:
  ## check hyper-reduction type
  #if hyp_type == "collocation":
  #  projector = state_basis.transpose() * sampling_mat.transpose()
  #elif hyp_type =="gappy_pod":
  #  projector = state_basis.transpose() * res_basis *  pseudo_inverse(sampling_mat * res_basis)
  #

  # write to disk
  #TODO
  #_writeResultsToFile(input_dict, projector)




#-----------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#-----------------------------------------------------------------
def createGalerkinProjector(**args):
  if len(args) == 2 \
     and 'communicator' in args.keys() \
     and 'yamldic'      in args.keys():
    _createGalerkinProjectorReadYaml(args['communicator'],
                                     args['yamldic'])



