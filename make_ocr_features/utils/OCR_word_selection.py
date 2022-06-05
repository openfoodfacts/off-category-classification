# Author: Anis ZAKARI <anis.zakari@outlook.fr>
import pandas as pd
import numpy as np
import json
import requests
from utils.OCR_preprocessing import remove_duplicates, text_cleaner

### TFIDF TRICK

def get_best_tfidf_words_list(sub_df_idx:int, feature_names:np.ndarray, tfidf_matrix,  N_words_to_pick:int=120)->str:
    """ extracts a list with N highest tf-idf score, from a document.
    Parameters
    ----------
    sub_df_idx: int
        index of the document in the sub-dataframe (filtered dataframe on a given language)
    feature_names:np.ndarray
        name of the features given by vectorizer.get_feature_names_out()
    tfidf_matrix: scipy.sparse.csr.csr_matrix
        tfidf sparse matrix
    N_words_to_pick: int, default = 120
        total words to pick.

    Returns
    -------
    words_picked: list:
        list of N words
    """
    sorted_scores = np.argsort(tfidf_matrix[sub_df_idx].data)[-N_words_to_pick:]
    indices_to_pick = tfidf_matrix[sub_df_idx].indices[sorted_scores]
    words_picked = feature_names[indices_to_pick]
    return words_picked


def get_best_tfidf_words_o(sub_df_idx:int, feature_names:np.ndarray, text_series:pd.Series, tfidf_matrix, freq_dict: dict,  N_words_to_pick:int=120)->str:
    """get best tfidf unique words and keeps the original order and the original casing.
    if a word has a frequence <=2 in all documents, it is filtered.

    Parameters
    ----------
    sub_df_idx: int
        index of the document in the sub-dataframe (filtered dataframe on a given language)
    feature_names:np.ndarray
        name of the features given by vectorizer.get_feature_names_out()
    text_series: pd.Series
        pd.Series containing OCR text
    tfidf_matrix: scipy.sparse.csr.csr_matrix
        tfidf sparse matrix
    freq_dict: dict
        dictionnary of word frequencies in the whole dataset
    N_words_to_pick: int, default = 120
        total words to pick.

    Returns
    -------
    words_picked: list:
        list of N words
    """
    word_list = get_best_tfidf_words_list(sub_df_idx, feature_names, tfidf_matrix, N_words_to_pick)
    df_text = text_series.iloc[sub_df_idx]
    if freq_dict is None:
        words_picked = " ".join([str(w) for w in df_text.split() if str(w).lower() in word_list])
    else:
        words_picked = " ".join([str(w) for w in df_text.split() if (str(w).lower() in word_list and freq_dict[str(w).lower()]>2)])
    return remove_duplicates(words_picked)

def get_freq_dict(text_series):
    """ computes word frequencies in text_series"""
    freq_dict = {}
    for text in text_series:
        for word in text.split():
            w_lower = str(word).lower()
            if w_lower in freq_dict:
                freq_dict[w_lower] +=1
            else:
                freq_dict[w_lower] =1
    return freq_dict


### intersection trick

def get_intersect_words_lang(lang_dict, N_max = 30):
    """get words that are used in all languages, for a given text.

    Parameters
    ----------
    lang_dict: dict
        dictionnary in the following format {"lang": {"prob": prob, "len_text":len_text, "text": text}}
    N_max: int
        N words to take

    Returns
    -------
    intersection: str
        words used in all languages.
    """
    
    dont_take= ["kj", "kcal", "kj", "total", "free", "net", "ingredients", "ingredient", "et", "de", "fat", "mg", "cg", "g", "kg", "ml", "cl", "l", "kl", "per", "pour", "valeur", "or", "le", "la", "dont", "consommer", "poids", "net", "www", "com", "which", "of", "wt"]
    if len(lang_dict) > 1:
        sets = [set(remove_duplicates(text_cleaner(lang_dict[lang]["text"]), get_keys = True).split()) for lang in lang_dict.keys()]
        intersection =  " ".join([word for word in set.intersection(*sets) if word not in dont_take][:N_max])
        return intersection
    else: 
        return ""

def get_intersect_words_ocr(sub_ocr_text_dict, N_max = 30):
    """get words that are found in all images, for a given barcode.
    Parameters
    ----------
    sub_ocr_text_dict: dict
        sub ocr text dict associated with a barcode.
    N_max: int
        N words to take

    Returns
    -------
    intersection: str
        words found in all images, for a given barcode.
    """

    dont_take= ["kj", "kcal", "kj", "total", "free", "net", "ingredients", "ingredient", "et", "de", "fat", "mg", "cg", "g", "kg", "ml", "cl", "l", "kl", "per", "pour", "valeur", "or", "le", "la", "dont", "consommer", "poids", "net", "www", "com", "which", "of", "wt"]
    if len(sub_ocr_text_dict) > 1:
        sets = [set(remove_duplicates(text_cleaner(sub_ocr_text_dict[key]), get_keys = True).split()) for key in sub_ocr_text_dict.keys()]
        intersection = " ".join([word for word in set.intersection(*sets) if word not in dont_take][:N_max])
        return intersection
    else: 
        return ""


### big words selection

from PIL import Image
import requests
import urllib
import io

def make_barcode(x):
    """ takes an EAN of 13 digits and returns an EAN in the format "XXX/XXX/XXX/XXXX """
    x = str(x)
    return "{}/{}/{}/{}".format(x[:3], x[3:6], x[6:9], x[9:])

def make_link_from_barcode(barcode, df, file = "image", keys = None):
    """creates url for json or jpg, from the barcode """
    if keys is None:
        keys = df.loc[df["code"]==barcode, "keys"].values[0]
    if isinstance(keys, str):
        keys = eval(keys)
    elif isinstance(keys, list):
        pass

    links = []
    if file == "image": file = "jpg"
    if file == "json": file = "json"
    barcode_with_slash = make_barcode(barcode)
    for key in keys:
        link = "https://world.openfoodfacts.org/images/products/{}/{}.{}".format(barcode_with_slash, key,file)
        links.append(link)
    return links

def show_images(links):
    """display images from links"""
    for link in links:
        response = requests.get(link)
        image_bytes = io.BytesIO(response.content)
        img = Image.open(image_bytes)
        img.show()



def show_images_from_barcode(barcode, df, keys = None):
    """display all the images belonging to a barcode"""
    links = make_link_from_barcode(barcode, df=df, keys = keys)
    show_images(links)

import math
import requests
def get_score_from_verticles(txt_annotations:dict):
    """the code iterates through all the anchor boxes used to detect the text.
    It finds the bigest Anchor box with the least words. The idea here is to find important words
    and usually important words are written in big print and are titles or captions so it doesn't have 
    much characters. The score computed here is detection_box_area / len_text 

    Parameters
    ----------
    txt_annotations: dict
        dictionnary containing detection boxes coordinates for a given image.

    Returns
    -------
    score: float
        highest score through iteration.
    text: str
        text found in the anchor associated with the highest score. 
    """
    txt = txt_annotations["description"]
    len_text = len(txt)
    y_min = math.inf
    y_max = -math.inf
    x_min = math.inf
    x_max = -math.inf

    verticles = txt_annotations['boundingPoly']['vertices']
    for coords in verticles:
        if 'y' in coords:
            y_min = min(coords['y'], y_min)
            y_max = max(coords['y'], y_max)
        if 'x' in coords:
            x_min = min(coords['x'], x_min)
            x_max = max(coords['x'], x_max)
    area = abs(x_max-x_min) * abs(y_max-y_min)
    score = area/len_text
    return score,txt

def get_n_most_important_words(results, word_count_limit = 10):
    """selects N unique words from an image
    
    Parameters
    ----------
    results: list:
        list of tuples, the tuples are in the following format: (score, text)
        
    word_count_limit: int
        number of words to select.

    Returns
    -------
    words_to_keep: str
        N words to keep from an image
    """
    to_keep = {}
    for items in results:
        words = items[1]
        for word in text_cleaner(words).split():
            if word not in to_keep:
                to_keep[word.lower()] = word
                if len(to_keep) == word_count_limit:
                    return " ".join(to_keep.values())
    words_to_keep =  " ".join(to_keep.values())
    return words_to_keep      



def get_big_words_from_txt_annotations(txt_annotations):
    """ selects N unique words from an image
    Parameters
    ----------
    txt_annotations: dict
        dictionnary containing detection boxes coordinates for a given image.

    Returns
    -------
    words_to_keep: str
        N words to keep
    """
    try:
        results = sorted([get_score_from_verticles(txt_a) for txt_a in txt_annotations], reverse = True)
        text = get_n_most_important_words(results, word_count_limit = 10)
    except:
        text = ""
    return text


def get_big_words_from_image(barcode, df, keys = None):
    """ selects N unique words from an image.Here, we use the url of the json, created with the barcode
    
    Parameters
    ----------
    barcode: str
        EAN - 13 digits
    df: pd.DataFrame
    keys: list, default = None
        list of keys used to create the urls. if the value is None,
        the information is fetched from the dataframe.
    
    Returns
    -------
    words_to_keep: str
        N words to keep    
    """
    texts = []
    links = make_link_from_barcode(barcode, df, file = "json", keys = keys)
    for link in links:
        try:
            response = urllib.request.urlopen(link)
            js = json.loads(response.read())
            txt_annotations = js['responses'][0]['textAnnotations'][0]
            results = sorted([get_score_from_verticles(txt_annotations)  for txt_annotations in js['responses'][0]['textAnnotations']], reverse = True)
            text = get_n_most_important_words(results, word_count_limit = 15)

        except:
            text = ""
        texts.append(text)
    return " ".join(texts)

def get_big_words_from_image_clean(barcode, df, keys = None):
    return text_cleaner(get_big_words_from_image(barcode, df, keys = keys))