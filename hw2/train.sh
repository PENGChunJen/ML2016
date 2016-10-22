#!/bin/bash                                                                                                                                             
if [ "$#" == "2" ]; then
    python logistic_regression.py $1 $2 
else
    echo "Usage: ./train.sh <training data> <output model>"
fi
