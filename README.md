# OFF Category Classification

This repository contains the code to train a multi-label category classifier for Open Food Facts products.

It works within [Robotoff](https://github.com/openfoodfacts/robotoff), currently using product_name and ingredients_tags.

## Sources

[`download.sh`](./download.sh) can help you download the data.

[`train.py`](./train.py) is the script to run to train the model. It can last for several hours depending on your machine.

[Threshold.ipynb notebook](./Threshold.ipynb) tries to measure performance 
to understand where to set threshold for automatic classification.
Results are summarized in [2021-10-15-kulizhsy-category-classifier-performance.pdf](./2021-10-15-kulizhsy-category-classifier-performance.pdf)

## Contributing


A [Data for Good to add more features to the model has been initiated](https://wiki.openfoodfacts.org/DataForGood-2022). You can find things to help with on issue [What can I work on ?](https://github.com/openfoodfacts/off-category-classification/issues/2)
