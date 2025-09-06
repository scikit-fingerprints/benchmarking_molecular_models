#!/bin/bash

if $INSTALL_DEP; then
    echo $(pwd)
    if [ ! -d "weights" ]; then
        echo "Please donwload ckpt files under the `weights` directory!"
        echo "https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq"
        exit 10
    fi
    # pip install -r requirements.txt -i https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    pip install git+https://github.com/MolecularAI/pysmilesutils.git
    # export CUDA_HOME=/opt/cuda
    # Deepspeed dependencies problems
    pip install "pydantic>1.10.10,<2.0.0"
    pip install deepspeed==0.9.0
    pip install "numpy>1.22.0,<2.0.0"

    # python fix_checkpoint.py
fi

export PYTHONPATH=$PYTHONPATH:$PWD/chemformer_repo

export HYDRA_EXPERIMENT=chemformer
