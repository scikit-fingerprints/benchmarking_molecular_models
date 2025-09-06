#!/bin/bash

TORCH=1.9.0
CUDA=cpu

if $INSTALL_DEP; then
#  wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
#  pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
  pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
  pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
  pip install -r requirements.txt
    rm ./*.whl
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/graphfp_repo/mol
export HYDRA_EXPERIMENT=graphfp
