#!/bin/bash

if $INSTALL_DEP; then
    pip install -e ./clamp_repo
fi

export HYDRA_EXPERIMENT=clamp
