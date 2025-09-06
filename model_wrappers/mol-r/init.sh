#!/bin/bash

if $INSTALL_DEP; then
    pip install -r ./mol_r_repo/requirements.txt
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/mol_r_repo/src
export HYDRA_EXPERIMENT=mol_r