#!/bin/bash

MODEL=$1
SOLUTIONS=$2
MAX_NUM_SOLUTIONS=$3

DIR1=`dirname $MODEL`
DIR2=`dirname $SOLUTIONS`

if [ "$#" -ne 3 ]; then
    echo "You must enter exactly 3 command line arguments <model_file> <solution_file> <max num solutions>"
    exit 1
fi

echo "Model file: $MODEL"
echo "Solutions file: $SOLUTIONS"

docker run -i -v $DIR1:/docker/$DIR1 -v $DIR2:/docker/$DIR2 -u `id -u $USER` scipoptsuite/scipoptsuite:7.0.2 scip <<EOF
read /docker/$MODEL
set presolving maxrounds 0
set constraints countsols sollimit $MAX_NUM_SOLUTIONS
set constraints countsols collect TRUE
count
write allsolutions /docker/$SOLUTIONS
EOF