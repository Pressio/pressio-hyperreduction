
import yaml
from yaml import load, dump
from yaml import Loader, Dumper

def yaml_read(file_path):
  with open(file_path) as f:
    yaml_obj = yaml.load(f,Loader=Loader)
  return yaml_obj

def yaml_write(yaml_obj,file_path):
  with open(file_path,'w') as f:
    yaml.dump(yaml_obj,f)
