#!/bin/python

cd onnode

python test_version.py

# i/o array
cd test_io
python test_ascii_io.py
python test_binary_io.py
cd ..

# yaml parsing
cd test_yaml_rw
python ./test_yaml.py
cd ..

# data structures
python test_vec.py
python test_mv.py

# qr
python test_qr.py

# svd
python test_svd.py

# pseudoinv
python test_pseudoinv.py

# lev scores functions
python test_levscores.py

# sample mesh with lev scores
python test_sm_levscores.py

# galerkin projector functions
python test_galerkin_projector.py

# lspg weighting
python test_lspg_weighting.py

cd ..