#!/bin/bash
PWD=$(pwd)

if $INSTALL_DEP; then
    cd coati_repo
    pip install .
    cd $PWD

    # https://github.com/rdkit/rdkit/issues/7159#issuecomment-1975192046
    pip install --force-reinstall pandas==2.1.0 numpy==1.26.4

    # curl -L https://patch-diff.githubusercontent.com/raw/rdkit/rdkit/pull/7165.diff -o /tmp/7165.diff
    
    # CURRENT_PATH=$(pwd)
    # cd venv/lib/python3.9/site-packages
    # [ -e rdkit/Chem/PandasPatcher.py ] && patch -p1 < /tmp/7165.diff || echo "Could not find PandasPatcher.py"
    cd ../
fi

export PYTHONPATH=$PYTHONPATH:$PWD/coati_repo

export HYDRA_EXPERIMENT=coati