#!/bin/bash                                                                                                                                             
if [ "$#" == "2" ]; then
    python cluster.py $1 $2 
else
    echo "Usage: ./cluster.sh <directory path> <output.csv>"
fi
