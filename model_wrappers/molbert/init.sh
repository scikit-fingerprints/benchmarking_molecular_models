#!/bin/bash

if $INSTALL_DEP; then
    if [ $(uname -s) == "Darwin" ]; then
        PYTHONPATH="" OPENBLAS="$(brew --prefix openblas)" pip install -e ./molbert_repo
    else
        pip install numpy pandas rdkit
        pip install -e ./molbert_repo
    fi
    wget -O ./weights.zip https://ndownloader.figshare.com/files/25611290
    unzip ./weights.zip -d ./weights
fi

export HYDRA_EXPERIMENT=molbert
export PYTHONPATH=$PYTHONPATH:$(pwd)/molbert_repo