#!/bin/bash

# parse
DELETE=false
CACHE=true
CMD=embed.py
REINSTALL=""
WRAPPER_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete)
            DELETE=true
            shift
            ;;
        -r|--reinstall)
            REINSTALL=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --clock)
            CMD=clock.py
            shift
            ;;
        --embed)
            CMD=embed.py
            shift
            ;;
        *)
            WRAPPER_PATH=$1
            shift
            ;;
        -*|--*)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

run() {
    cd $WRAPPER_PATH

    # install environment
    if [ -d "venv" ] && [ -z "$REINSTALL" ]; then
        source venv/bin/activate
        export INSTALL_DEP=false
    else
        export INSTALL_DEP=true
        python_version=$(cat .python-version)
        eval "$(pyenv init -)"
        pyenv shell $python_version
        if [ ! -d "venv" ]; then
            python -m venv venv
        fi
        source venv/bin/activate
        pip install -r ../../base_requirements.txt
    fi

    source ./init.sh

    cd ../../

    export PYTHONPATH=$PYTHONPATH:.:$WRAPPER_PATH

    python -u $CMD --multirun +experiment=$HYDRA_EXPERIMENT ++cache=$CACHE

    echo "Done"
}

delete() {
    rm -rf $WRAPPER_PATH/venv
}

if [ -z "$WRAPPER_PATH" ]; then
    echo "Please provide the path to the wrapper"
    exit 1
fi

if $DELETE; then
    delete
else
    run
fi
