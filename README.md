# GPT2 Data Generator
OpenAI's GPT2 is a powerful language model, so why not use it to generate some basic text data for those times in low data scenario.

## Requirements
- docker
- git


## Setup
Clone gpt2 and build docker image, run `setup.sh`:

For CPU
```
bash setup.sh Dockerfile.cpu gpt2_docker
```

For GPU
```
bash setup.sh Dockerfile.gpu gpt2_docker
```

This may take some time since the docker image downloads 4 pretrained models

Note: the current dockerfile available in the gpt2 repo is outdated


## Using the generator
1. Start docker container, run `run_docker.sh`:
```
bash run_docker gpt2_data_generator gpt2_docker
```

2. Start the generator, run `python gpt2_data_generator/conditional_generator.py`

    - Type in a prompt
    - When the generated data is worth saving press Y
    - Enter save file name
    - saved file will be available in `data/...`
