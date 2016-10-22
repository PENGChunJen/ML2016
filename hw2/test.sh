#!/bin/bash
if [ "$#" == "3" ]; then
    python distribution.py $1 $2 $3
else
    echo "Usage: ./test.sh <model name> <testing data> <prediction.csv>"
fi
