#!/bin/bash

CONTAINERNAME=${1:-gpt2_data_generator}
IMAGENAME=${2:-gpt2_cpu}

docker rm -f $CONTAINERNAME

docker run -i -N $CONTAINERNAME -t $IMAGENAME