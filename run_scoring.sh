#!/bin/bash

EXPERIMENT=$1
DATETIME=$(date '+%Y%m%d_%H%M')

mkdir -p logs_scoring

nohup nice -n 10 python score.py --multirun +experiment=${EXPERIMENT} > logs_scoring/${EXPERIMENT}_${DATETIME}.log 2>&1 &

echo "PID $! started"
