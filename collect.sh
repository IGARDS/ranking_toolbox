#!/bin/bash

MODEL=$1
SOLUTIONS=$2

if [ "$#" -ne 2 ]; then
    echo "You must enter exactly 2 command line arguments"
    exit 1
fi

echo "Model file: $MODEL"
echo "Solutions file: $SOLUTIONS"

/usr/local/SCIPOptSuite/bin/scip <<EOF
read $MODEL
set presolving maxrounds 0
set constraints countsols sollimit 500000
set constraints countsols collect TRUE
count
write allsolutions $SOLUTIONS
EOF
