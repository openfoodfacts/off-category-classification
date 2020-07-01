#!/bin/bash

mkdir -p data
cd data

RELEASE='dataset-category-2019-09-16'
LANG='xx'
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/ingredients.full.json
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/categories.full.json
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/category_${LANG}.test.jsonl.gz
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/category_${LANG}.train.jsonl.gz
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/${RELEASE}/category_${LANG}.val.jsonl.gz
