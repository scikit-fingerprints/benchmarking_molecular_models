#!/bin/bash

if $INSTALL_DEP; then
  pip install -r requirements.txt
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/gem_repo
export HYDRA_EXPERIMENT=gem