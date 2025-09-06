#!/bin/bash

if $INSTALL_DEP; then
    pip install unimol_tools --upgrade 
    pip install huggingface_hub
    pip install "pandas<2"
fi

export HYDRA_EXPERIMENT=unimol
