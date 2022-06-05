# Author: Anis ZAKARI <anis.zakari@outlook.fr>

import re
import multiprocess as mp
import numpy as np

###make df###

def get_row_from_json_for_df(json_line: dict) -> list:
    """ extract items from json and returns a row to put in a dataframe 
    Parameters
    ----------
    json_line: dict 
        dictionnary loaded from a line in the jsonl file.
    
    Returns
    -------
    row: list
        list with items [code, ocr_texts, keys]
            code: barcode of the product
            ocr_texts: concatenated texts coming from all the images
            keys: keys of all the images; for example 004/150/000/7229/2.json  <-- "2" is the key.
    """
    code = json_line['code']
    if "ocrs" in json_line:
        texts = []
        keys =  list(json_line['ocrs'].keys())
        for key in keys:
            ocr_text = json_line['ocrs'][key]['text']
            texts.append(ocr_text)
    row = [code, " ".join(texts), keys]
    return row

def get_item_from_json_for_dict(json_line: dict, ocr_text_dict: dict) -> dict:
    """ extracts ocr text from all images, for a given barcode.
    returns a dict with new elements coming from json_line.
    
    Parameters
    ----------
    json_line: dict 
        dictionnary loaded from a line in the jsonl file.
    ocr_text_dict: dict
        dictionnary in the following format {barcode: {key: ocr_text}}
    
    Returns
    -------
    ocr_text_dict: dict
        dictionnary ocr_text_dict with new elements coming from json_line.
    """
    
    code = json_line['code']
    if "ocrs" in json_line:
        keys =  list(json_line['ocrs'].keys())
        if len(keys) >0:
            ocr_text_dict[str(code)] = {}
            for key in keys:
                ocr_text_dict[str(code)][key] = json_line['ocrs'][key]['text']
    return ocr_text_dict


### cleaner ###

def text_cleaner(text:str) -> str:
  """ takes a text as input and cleans it"""

  dont_take = ["kj", "kcal", "kj", "total", "free", "net", "ingredients", "ingredient", "et", "de", "fat", "mg", "cg", "g", "kg", "ml", "cl", "l", "kl", "per", "pour", "valeur", "or", "le", "la", "dont", "consommer", "poids", "net", "www", "com", "which", "of", "wt"]
  text_cleaned = text.replace("\n", " ") #remove line breaks
  text_cleaned = re.sub("\S*(www\.|\.com|\.net|\.fr|\.co\.uk|\.org)\S*", "", text_cleaned) #remove websites
  text_cleaned = re.sub("\w*([0-9]{0,}[,|\.]{0,}[0-9])\w*", " ", text_cleaned) #remove measurements 
  text_cleaned = re.sub(r"\b([a-zA-Z]{1})\b", " ", text_cleaned) # remove isolated letters ex --> g g g g g
  text_cleaned = re.sub("( +- +)", " ", text_cleaned)
  text_cleaned = re.sub(r"[\·|/|\-|\\|(|)|\+|\*|\[|\]|™|ᴿˣ|\*|\—|\^|\"|®|>|<|″|\||\&|\#|\,|\;|⭐|\xa0|\?|\%|\'|©|\@|\$|\€|\:|\}|\{|\°]", " ", text_cleaned)
  text_cleaned = re.sub(r" +", " ", text_cleaned) # remove multiple spaces

  text_cleaned = " ".join([w for w in text_cleaned.split() if (w.isalpha() and w.lower() not in dont_take)])
  return text_cleaned


###remove duplicates
def remove_duplicates(text, get_keys = False):
    """takes a text and removes the duplicated words. It keeps the string's original casing"""
    D = {word.lower(): word  for word in str(text).split()}
    if get_keys:
        return " ".join(D.keys())
    else:
        return " ".join(D.values())

### parralel calc
def parallel_calc(func, iterable, n_core = mp.cpu_count()):
    """ simple wrapper code around func to parallelize the work.
    
    Parameters
    ----------
    func: callable
        function to use for parallelization 
    iterable: iterable
        items to feed the parralelized workers
    n_core: int
        total CPUs on your computer

    Returns
    -------
    results: list
        iterable processed by the function. 
    """
    pool = mp.Pool(n_core-1)
    results = pool.map(func, np.array(iterable))
    pool.close()
    return results

def parallel_calc_multi(func, *args, n_core = mp.cpu_count()):
    """ simple wrapper code around func to parallelize the work.
    This version supports multiple arguments
    
    Parameters
    ----------
    iterable: iterable
        items to feed the parralelized workers
    func: callable
        function to use for parallelization 
    n_core: int
        total CPUs on your computer

    Returns
    -------
    results: list
        iterable processed by the function. 
    """
    pool = mp.Pool(n_core-1)
    results = pool.starmap(func, zip(*args))
    pool.close()
    return results