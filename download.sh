#!/bin/bash

# make sure the release is in sync with the 'off_categories' dataset version.

mkdir -p data
cd data

RELEASE='dataset-category-2021-09-15'
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/categories.full.json
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/agribalyse_categories.txt
