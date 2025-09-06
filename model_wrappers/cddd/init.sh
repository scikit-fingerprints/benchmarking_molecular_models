#!/bin/bash
PWD=$(pwd)

if $INSTALL_DEP; then
    pip install -r requirements.txt

    # if [ -f default_model.zip ]; then
    #     unzip default_model.zip
    #     rm default_model.zip
    # fi

    if [ ! -d default_model ]; then
        echo "Please download pre-trained model from"
        echo "https://drive.usercontent.google.com/download?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h&export=download&authuser=0"
        exit 10
    fi
fi

export PYTHONPATH=$PYTHONPATH:$PWD/cddd_repo
echo $PYTHONPATH

export HYDRA_EXPERIMENT=cddd