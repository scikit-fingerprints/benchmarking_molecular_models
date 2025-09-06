#!/bin/bash

EXPERIMENT=$1
DATETIME=$(date '+%Y%m%d_%H%M')

mkdir -p logs_embed

nohup nice -n 10 ./embed_wrapper.sh model_wrappers/${EXPERIMENT} >logs_embed/embed_${EXPERIMENT}_${DATETIME}.log 2>&1 &

echo "PID $! started"
