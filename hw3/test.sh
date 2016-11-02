#!/bin/bash                                                                                                                                             
if [ "$#" == "3" ]; then
    python cnn.py $1 $2 $3
else
    echo "Usage: ./test.sh <directory path> <inpu_model> <prediction.csv>"
fi
