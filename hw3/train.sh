#!/bin/bash                                                                                                                                             
if [ "$#" == "2" ]; then
    python method2.py $1 $2 
else
    echo "Usage: ./train.sh <directory path> <output model>"
fi
