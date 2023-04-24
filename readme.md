# kac_prediction

Deep learning toolkit for orchestrating and arranging arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_prediction.git#egg=kac_prediction"
```

### Dependencies

-	[cmake](https://formulae.brew.sh/formula/cmake)
-	[curl](https://formulae.brew.sh/formula/curl)

# Usage

<details>
<summary>Core Library</summary>

### Import

```python

```

### Classes

```python

```

### Types

```python

```

</details>

<details>
<summary>Models</summary>

### Import

```python

```

### Classes

```python

```

</details>

# Development / Local Training

### Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

### Install

```bash
git clone ...
pipenv install -d
# then for targetting gpus and linux distros
pipenv run pip install torch torchaudio --index-url ...
```

### Training Commands

Import one of the official datasets used as part of this project.

```bash
sh ./bin/install-dataset.sh 5000-circular-drums-of-varying-size
sh ./bin/install-dataset.sh 5000-rectangular-drums-of-varying-dimension
```

Then run the example training script using.

```bash
# local
pipenf run python scripts/train.py -h
usage: train.py [-h] [--config CONFIG] [--testing] [--wandb]
options:
  -h, --help       show this help message and exit
  --config CONFIG  path to yaml config file
  --testing        use testing dataset
  --wandb          log to wandb

# wandb
pipenv run wandb sweep scripts/config/wandb/...
pipenv run wandb agent ...
```

### Test

```bash
pipenv run test
```