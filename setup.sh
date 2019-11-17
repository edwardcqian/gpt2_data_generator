#!/bin/bash
mkdir gpt2

git clone https://github.com/openai/gpt-2.git gpt2

cd gpt2

git checkout d98291d2ae0761eff6d12f0e4e52e93c7e847eb2

# python download_model.py 124M
# python download_model.py 355M
# python download_model.py 774M
# python download_model.py 1558M

cd ..

DOCKERFILE=${1:-Dockerfile.cpu}

IMAGENAME=${2:-gpt2_cpu}

docker build -f $DOCKERFILE -t $IMAGENAME gpt2
