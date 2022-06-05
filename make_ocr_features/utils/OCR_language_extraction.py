# Author: Anis ZAKARI <anis.zakari@outlook.fr>

import fasttext
import numpy as np
from tqdm import tqdm
import re
#load model
#PRETRAINED_MODEL_PATH = 'fasttext_weights/lid.176.bin'
#link to download fasttext weights---> https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


def artificial_sentence_split(text:str, n_words_per_sentence:int = 8)-> list['str']: 
    """splits artificially a text, based on a pre-defined number of words. 
    On average there are 15 words per sentence.
    
    Parameters
    ----------
    text: str 
        input text.
    n_words_per_sentence: int
        number of words in each artificial sentence.
    
    Returns
    -------
    sentence_split: list
        a list of sentences containing 8 words each.
    """
    txt_split = text.split()
    total_words = len(txt_split)
    if total_words >= n_words_per_sentence:
        n_sentences = total_words // n_words_per_sentence
        rest = total_words % n_words_per_sentence
        chunks = [[i*n_words_per_sentence, (i+1)*n_words_per_sentence ] for i in range(n_sentences)]
        chunks[-1][1]+=rest
    else:
        chunks = [[0, total_words]]
    sentence_split = [" ".join(txt_split[slice(*chunk)]) for chunk in chunks]
    return sentence_split


def get_clean_lists_from_fasttext(lang_labels: list['list'], probs_list: list['list']):
    """ fasttexts returns a list of lists for language labels and probs_list.
    for each item in the list the argmax is taken

    Parameters
    ----------
    lang_labels: list of lists .
        list of langages found, for each OCR text.
    probs_list: list of lists.
        list of all the probabilities of the languages, for each OCR text.
    
    Returns
    -------
    lang_labels_output: list
        list containing the main language for each OCR.
    prob_list_output: list
        The associated probabilities of the languages in lang_labels_output.
    """
    main_lang_idx = [np.argmax(items) for items in probs_list]
    lang_labels_output = [lang_label[i].split('__label__')[1] for lang_label, i in zip(lang_labels, main_lang_idx)]
    prob_list_output = [prob_list[i] for prob_list,i in zip(probs_list, main_lang_idx)]
    return lang_labels_output, prob_list_output


def text_lang_split(text:str, model)-> dict:
    """
    takes text as input and splits it in a dictionnary with languages found as keys.

    Parameters
    ----------
    text: str
    
    Returns
    -------
    main_lang: str
        language that contains the most words in the dictionnary
    sorted_dict: dict:
        dictionnary sorted in the following format : {"lang": {"prob": prob, "len_text":len_text, "text": text}}
        text: text found with in a given language.
        len_text: the length of the text.
        prob: a list of probabilities, each probability corresponds to a sentence.
    """
    text = re.sub(r"\n", " ", text)
    text = re.sub("\.+", ".", text)
    lang_dict = {}
    sentences = artificial_sentence_split(text)
    langs, probs = get_langs(sentences, model)
    for sentence, lang, prob in zip(sentences, langs, probs):
       
        if lang in lang_dict:
            lang_dict[lang]["prob"].append(prob)
            lang_dict[lang]["len_text"] += len(sentence)
            lang_dict[lang]["text"]+= " " + sentence
            
        else:
            lang_dict[lang] = {"prob": [prob]}
            lang_dict[lang]["len_text"] = len(sentence)
            lang_dict[lang]["text"] = sentence
            
    sorted_dict = {k: v for k, v in sorted(lang_dict.items(), key=lambda item: item[1]["len_text"], reverse = True)}
    main_lang = next(iter(sorted_dict))
    return main_lang, sorted_dict

def get_langs(sentences:list, model):
    """ detects languages and returns for each sentence the most probable one with its associated probability.
    Parameters
    ----------
    sentences: list
        list of string sentences.
    
    Returns:
    ----------
    langs: list 
        list of languages.
    probs: list
        list of proababilities.
    """
    lang_labels, probs_list = model.predict(sentences)
    langs, probs = get_clean_lists_from_fasttext(lang_labels, probs_list)
    return langs, probs


def get_lang_items_from_pd_textlist(pdSeries, PRETRAINED_MODEL_PATH) -> list:
    """ takes a text as input, creates sentences, detects the most proable languages"
    
    Parameters
    ----------
    pdSeries: pd.Series

    Returns
    -------
    text_list: list
        each item is the text associated with the most probable language. 
    lang_dict_list: list
        each item is a dictionnary that splits the text according to the languages found.
    main_lang_list: list
        each item is the most probable language for a given text.

    """
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    text_list= []
    lang_dict_list = []
    main_lang_list = []
    for t in tqdm(pdSeries):
        sentences = artificial_sentence_split(t)
        #detect and split text in a dict according to the languages found.
        main_lang, lang_dict = text_lang_split(t, model)
        text_to_keep = lang_dict[main_lang]["text"]
        text_list.append(text_to_keep)
        main_lang_list.append(main_lang)
        lang_dict_list.append(lang_dict)
    return text_list, lang_dict_list, main_lang_list