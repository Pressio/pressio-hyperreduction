#!/bin/python

python test_version.py

# test data structures
python test_vec.py
python test_mv.py

# yaml parsing
cd test_yaml_rw
python ./test_yaml.py
cd ..

# i/o array
cd test_io
python test_ascii_io.py
python test_binary_io.py
cd ..

# lev scores functions
python test_levscores.py

# sample mesh with lev scores
python test_sm_levscores.py

# galerkin projector functions
python test_sm_galerkin_projector.py
