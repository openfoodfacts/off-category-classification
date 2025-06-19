# Open Food Facts Category Classification

This repository contains the code to train a multi-label category classifier for Open Food Facts products. The latest version of the category classifier model is currently the [v3 model](https://github.com/openfoodfacts/robotoff-models/releases/tag/keras-category-classifier-image-embeddings-3.0).

It works within [Robotoff](https://github.com/openfoodfacts/robotoff), currently using as input features:

- the product name (`product_name` field)
- the ingredient list (`ingredients` field): only the ingredients of depth one are considered (sub-ingredients are ignored)
- the ingredients extracted from OCR texts: all OCR results are fetched and ingredients present in the taxonomy are detected using flashtext library.
- the most common nutriments: salt, proteins, fats, saturated fats, carbohydrates, energy,...
- up to the most 10 recent images: we generate an embedding for each image using [clip-vit-base-patch32](https://github.com/openfoodfacts/robotoff-models/releases/tag/clip-vit-base-patch32) model, and generate a single vector using a multi-head attention layer + GlobalAveragePooling1d.

## Sources

[`train.py`](train.py) is training script. It can last for several hours depending on your machine.

[`scripts/gen_image_embedding.py`](scripts/gen_image_embedding.py) This script is used to generate the image embeddings used as input for the model, and save the embeddings to an HDF5 file that will be used during training.

## Dev install

If you want to develop, here is a sample install using virtual envs.

Install needed dependencies:

On ubuntu :

```
sudo apt install python3-venv python3-dev build-essential
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
