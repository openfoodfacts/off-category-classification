# dataforgood_ocr

This Repo aims at creating a dataframe with OCR features so that we can use it to train a model to predict categories. In other words, we want to create a dataframe out of `predict_categories_dataset_ocrs.jsonl.gz` with usable features for a category prediction task. 

# Usage
Recommended python version: python3.9
To run this project see the following steps:

# Creating virtual environment:
`conda create --name ocr_env python=3.9`  
then:  
`conda activate ocr_env`

# Clone repo:
`git clone https://github.com/AnisZakari/dataforgood_ocr.git`

# Accessing the folder:
`cd dataforgood_ocr`

# Install Requirements:
`pip install -r requirements.txt`

# Download pretrained weights for fasttext
`wget fasttext_weights/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`  
Or you can just download `lid.176.bin` and put it in the folder `fasttext_weights`

# Run notebook
Now you are good to go ! you can run the notebook


## About the input file

- input file `predict_categories_dataset_ocrs.jsonl.gz` weights 335.8 MB  
about the file : https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_documentation.txt  download the file : https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_ocrs.jsonl.gz

- each line of the jsonl file contains OCRs associated barcodes.
- each OCR contains text, potentially in different languages

## What is done in this notebook
1. Make the OCR Dataframe
- A DataFrame is made from the jsonl.gz
- All the OCRs (text) of a product are concatenated
- For each product the different languages are detected.  


Problem: in some cases the text is too long. If we were to use a transformer model like Bert to predict categories, there is a major limitation: the maximum number of words Bert can handle is 500.
We need a way to extract only relevant words. Bellow, all the strategies used:

2. Selection strategies / Summerization strategies:
- main language strategy: only the main language is kept (i.e the lang in which there are many words and a good confidence score)
- TFIDF Selection strategy : The approach here is to filter by language and to create a TFIDF matrix. For each product of a given language we keep only the N-top words according to their TFIDF score.
- language intersection strategy:
We get words that are found in all the different languages detected.
- OCR intersection strategy:
We get found in all the different OCRs (i.e all the different images).
- Big words in the images strategy:
We fetch the biggest words in the images. They are found by choosing words that have the biggest bounding polygon (anchor detection box) for the least text. In other words we choose the words that have the biggest detection-box-size / character-length ratio


3. Final output: 
- When the original input text is not too long it remains as is, it is only cleaned with the `text_cleaner` there isn't any word selection at all.
- If the original input text has more than 500 characters, we pick the text of the main language.
- If the text of the main language is has still more than 500 characters we remove duplicated words.
- If there is still more than 500 characters, we then take the tfidf selection.
- Finally we add big words from images and "intersection words" from OCRs and languages.  

