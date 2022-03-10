#!/bin/bash

MODEL=$1
echo $MODEL

docker run scipoptsuite/scipoptsuite:7.0.2 scip <<EOF
read $MODEL
count
EOF
