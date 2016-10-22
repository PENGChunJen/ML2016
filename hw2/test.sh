#!/bin/bash                                                                                                                                             
if [ "$#" == "3" ]; then
    python logistic_regression.py $1 $2 $3
else
    echo "Usage: ./test.sh <model name> <testing data> <prediction.csv>"
fi
