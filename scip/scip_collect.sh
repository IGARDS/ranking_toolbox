#!/bin/bash

MODEL=$1
SOLUTIONS=$2
MAX_NUM_SOLUTIONS=$3

if [ "$#" -ne 3 ]; then
    echo "You must enter exactly 3 command line arguments <model_file> <solution_file> <max num solutions>"
    exit 1
fi

echo "Model file: $MODEL"
echo "Solutions file: $SOLUTIONS"

scip <<EOF
read $MODEL
set presolving maxrounds 0
set constraints countsols sollimit $MAX_NUM_SOLUTIONS
set constraints countsols collect TRUE
count
write allsolutions $SOLUTIONS
EOF
