# Open Food Facts Category Classification

This repository contains the code to train a multi-label category classifier for Open Food Facts products.

It works within [Robotoff](https://github.com/openfoodfacts/robotoff), currently using product_name and ingredients_tags.

## Sources

[`download.sh`](./download.sh) can help you download extra data.

[`experiments/Train.ipynb`](experiments/Train.ipynb) is the notebook to train the model. It can last for several hours depending on your machine.

[Threshold.ipynb notebook](./Threshold.ipynb) tries to measure performance 
to understand where to set threshold for automatic classification.
Results are summarized in [2021-10-15-kulizhsy-category-classifier-performance.pdf](./2021-10-15-kulizhsy-category-classifier-performance.pdf)

## Contributing


A [Data for Good to add more features to the model has been initiated](https://wiki.openfoodfacts.org/DataForGood-2022). You can find things to help with on issue [What can I work on ?](https://github.com/openfoodfacts/off-category-classification/issues/2)

[![Open Train.py In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openfoodfacts/off-category-classification/blob/master/experiments/Train.ipynb)

## Deploying to production

- The output of training should be published on [Robotoff models](https://github.com/openfoodfacts/robotoff-models) as a release.
- The deployment from Robotoff models releases is already automated,
  see [robotoff .github/workflows/container-deploy-ml.yml](https://github.com/openfoodfacts/robotoff/blob/master/.github/workflows/container-deploy-ml.yml).

  You  will have to add a ml-xxx tag to trigger deploy

## Dev install

If you want to develop, here is a sample install using virtual envs.

Install needed dependencies:

On ubuntu :

```
sudo apt install python3-venv python-devbuild-essential
```

Create a virtual environment: `python3 -m venv .venv`

Activate the virtual environment (you will have to activate every time you use the project):
```
. .venv/bin/activate
```

Install requirement and eventually requirement-dev

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To launch jupyter notebook, just use (after activating your virtual env):
```
jupyter notebook
```
