#!/bin/bash
if [ "$#" == "2" ]; then
    python distribution.py $1 $2
else
    echo "Usage 1: ./train.sh <training data> <output model>"
fi
