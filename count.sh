#!/bin/bash

MODEL=$1
echo $MODEL

scip <<EOF
read $MODEL
count
EOF
