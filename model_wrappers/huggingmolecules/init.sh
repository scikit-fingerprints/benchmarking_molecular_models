#!/bin/bash

if $INSTALL_DEP; then
    pip install -e ./huggingmolecules_repo/src
    pip install git+https://github.com/bp-kelley/descriptastorus
fi

export PYTHONPATH=
export HYDRA_EXPERIMENT=huggingmolecules
