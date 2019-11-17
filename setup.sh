#!/bin/bash
mkdir gpt2

git clone https://github.com/openai/gpt-2.git gpt2

cd gpt2

DOCKERFILE=${1:-Dockerfile.cpu}

IMAGENAME=${2:-gpt2_cpu}

docker build -f $DOCKERFILE -t $IMAGENAME .
