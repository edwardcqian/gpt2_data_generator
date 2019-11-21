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

Optional config:
- `--config_path`: path to gpt2 model config json (default `config/default_config.json`)
- `--save_path`: save path for generated data files (default `data/`)

## Confit
GPT2 Model Config
- `model_name`: String, which model to use.
- `seed`: Integer seed for random number generators, fix seed to reproduce results.
- `nsamples`: Number of samples to return total.
- `batch_size`: Number of batches (only affects speed/memory). Must divide nsamples.
- `length`: Number of tokens in generated text, if None (default), is determined by model hyperparameters.
- `temperature`: Float value controlling randomness in boltzmann
    distribution. Lower temperature results in less random completions. As the
    temperature approaches zero, the model will become deterministic and
    repetitive. Higher temperature results in more random completions.
- `top_k`: Integer value controlling diversity. 1 means only 1 word is
    considered for each step (token), resulting in deterministic completions,
    while 40 means 40 words are considered at each step. 0 (default) is a
    special setting meaning no restrictions. 40 generally is a good value.
- `models_dir`: path to parent folder containing model subfolders
    (i.e. contains the <model_name> folder).
