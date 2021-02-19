
import pathlib, sys
from pressiotools.io.yaml_parser import *

def run():

  # read file, assert
  yaml_in = yaml_read("test.yaml")

  print(yaml_in)
  assert(yaml_in["ResidualBasis"]["file-root-name"]=="matrix.bin")
  assert(yaml_in["ResidualBasis"]["format"]=="binary")
  assert(yaml_in["ResidualBasis"]["num-columns"]==4)
  assert(yaml_in["ResidualBasis"]["num-mesh-nodes"]==15)
  assert(yaml_in["ResidualBasis"]["dofs-per-mesh-node"]==1)

  assert(yaml_in["SampleMesh"]["num-sample-mesh-nodes"]==8)
  assert(yaml_in["SampleMesh"]["leverage-score-beta"]==0.5)

  # write (and then read) file
  yaml_in["SampleMesh"]["num-sample-mesh-nodes"] = 9
  yaml_write(yaml_in,"test_r1.yaml")
  yaml_in2 = yaml_read("test_r1.yaml")
  assert(yaml_in2["SampleMesh"]["num-sample-mesh-nodes"]==yaml_in["SampleMesh"]["num-sample-mesh-nodes"])

if __name__ == '__main__':
  run()
