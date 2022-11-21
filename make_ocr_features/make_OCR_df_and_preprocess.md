### Make DataFrame and ocr_text_dict from jsonl


```python
import pandas as pd
import numpy as np
import os
from utils.OCR_preprocessing import get_row_from_json_for_df, append_item_ocr_text_from_json_to_dict
```


```python
path_ocrs = os.path.abspath("INPUT_datasets/predict_categories_dataset_ocrs.jsonl.gz")
```


```python
"""approx time 15sec
Iterating through the jsonl to extract elements for the dataframe and the ocr dictionary.
"""
from IPython.display import clear_output
import gzip
import json
# make df from json
rows = []
ocr_text_dict = {}
with gzip.open(path_ocrs) as f:
    for i, line in enumerate(f):
        #We take lines after 10 000 because those are potentially problematic.
        if i > 10000: 
            json_line = json.loads(line)
            if len(str(json_line['code'])) == 13:
                #for df
                row = get_row_from_json_for_df(json_line)
                rows.append(row)
                #for dict
                append_item_ocr_text_from_json_to_dict(json_line, ocr_text_dict)

df = pd.DataFrame(rows, columns = ["code", "texts", "keys"]).drop_duplicates(subset = "code")
df['code'] = df['code'].astype(str)
df['texts'] = df['texts'].astype(str)
print(df.shape)
print(len(ocr_text_dict))

del rows
```


```python
is_real_ean = df['code'].str.len() == 13
df = df[is_real_ean]
print("after removing non normalized codes", df.shape)
### Keep only texts > 10 char
len_sup_10 = (df["texts"].str.len()> 10)
df = df[len_sup_10]
print("after removing short texts", df.shape)

```

### Extract main language from text

for each text associated with a barcode, there are potentially many languages used to describe the product.
The aim of this section is to detect the main language and to extract its text.


```python
PRETRAINED_MODEL_PATH = 'fasttext_weights/lid.176.bin'
from utils.OCR_language_extraction import get_lang_items_from_pd_textlist
```


```python
"""approx time 2min15"""
text_list, lang_dict_list, main_lang_list = get_lang_items_from_pd_textlist(df['texts'],PRETRAINED_MODEL_PATH)
main_lang_dict = {code: dic for (code,dic) in zip(df["code"], lang_dict_list)}
#assign new items to df
df["text_main_lang"] = text_list
df["main_lang"] = main_lang_list
df.head(2)
```

### Clean the text


```python
from utils.OCR_preprocessing import text_cleaner, parallel_calc
```


```python
""" Non parallelized version - approx time 2m"""
#df["text_cleaned"] = df["text_main_lang"].apply(lambda x: text_cleaner(x))
#df = df.set_index("code", drop = True)

""" Parallelized version - approx time 30sec"""
# simple wrapper code around text_cleaner to parallelize the work
df["text_main_lang_cleaned"] = parallel_calc(text_cleaner, df["text_main_lang"])
```


```python
"""approx time 1min"""
df["original_text_cleaned"]  = parallel_calc(text_cleaner, df["texts"])
```


```python
df.head(3)
```


```python
#bump around 210 <-- investigate
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
word_count = df['text_main_lang_cleaned'].str.split().str.len()
sns.histplot(word_count)
plt.xlim([0,1000])
plt.show()

```

### TFIDF Trick


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import clear_output
from utils.OCR_word_selection import get_freq_dict, get_best_tfidf_words_cleaned
from tqdm import tqdm

df["tfidf_selection"] = ""
```


```python
"""takes approx 3min30"""

problematic_langs = []
df = df.reset_index(drop = True)
freq_dict = get_freq_dict(df["text_main_lang_cleaned"])
for lang in df["main_lang"].unique():
    print("lang:", lang)
    lang_filter = df["main_lang"]== lang 
    sub_df_text_series = df.loc[lang_filter, "text_main_lang_cleaned"]
    try:
        vectorizer = TfidfVectorizer(max_df = 0.8)
        tfidf_matrix = vectorizer.fit_transform(sub_df_text_series)
        feature_names = vectorizer.get_feature_names_out()
        params = {
            "feature_names": feature_names,
            "text_series": sub_df_text_series,
            "tfidf_matrix": tfidf_matrix,
            "freq_dict": freq_dict,
            "n_max_words_to_pick": 120
            }
        text_selection_list = [get_best_tfidf_words_cleaned(i, **params) for i in tqdm(range(sub_df_text_series.shape[0]))]
        df.loc[lang_filter, "tfidf_selection"] = text_selection_list
    except:
        problematic_langs.append(lang)
    clear_output()
```


```python
len(problematic_langs)
```


```python
df["tfidf_selection"].str.split().str.len().sort_values(ascending = False)
```

### "Intersection words trick"


```python
from utils.OCR_word_selection import get_intersect_words_lang, get_intersect_words_ocr
```


```python
"""takes approx 1min30"""
intersection_list_lang = parallel_calc(get_intersect_words_lang, lang_dict_list)
print("ratio:", len([item for item in intersection_list_lang if item != ""]) / len(intersection_list_lang))

df["intersection_words_lang"] = intersection_list_lang
```


```python
"""takes approx 56sec"""
sub_ocr_text_dict_list = [ocr_text_dict[code] for code in df["code"]]
intersection_list_ocr = parallel_calc(get_intersect_words_ocr, sub_ocr_text_dict_list)
print("ratio:", len([item for item in intersection_list_ocr if item != ""]) / len(intersection_list_ocr))

df["intersection_words_ocr"] = intersection_list_ocr
```


```python
df["intersection_words_ocr"].str.split().str.len().sort_values(ascending = False)
```

### Fetch big words from images


```python
import gzip
import json
from utils.OCR_word_selection import get_big_words_from_txt_annotations
```


```python
from collections import defaultdict
big_words_dict = defaultdict(str)
i_last = -1
```


```python
"""takes approx 7m30"""
txt_annotations_path = "INPUT_datasets/txtannotations.jsonl.gzip"

with gzip.open(txt_annotations_path) as f:
    for i, line in enumerate(f):
        if i > i_last:
            i_last +=1
            json_str = line.decode('utf-8')
            txt_annotations = json.loads(json_str)
            barcode = list(txt_annotations.keys())[0]
            sentences_list = [get_big_words_from_txt_annotations(txt_annotations[barcode][key]) for key in sorted(txt_annotations[barcode].keys())]
            if len(big_words_dict[str(barcode)]) <= 500:
                big_words_dict[str(barcode)] +=  " " + " ".join(sentences_list)

        if i %250000 == 0:
            print(i)
  
```


```python
len(big_words_dict)
```


```python
"""
import pickle
path = "barcodes_dict_with_new_ones.pkl"
with open(path, 'rb') as file:
    big_words_dict = pickle.load(file)
len(big_words_dict)
"""
```


```python
df["big_words"] = df["code"].apply(lambda x: big_words_dict.get(str(x), ''))
```


```python
df["big_words"].str.split().str.len().sort_values(ascending = False)
```

### Make output df


```python
from utils.OCR_preprocessing import remove_duplicates
```


```python
"""takes approx 15 sec"""

df["word_selection"] = (
    df["big_words"].astype(str) + ". " +
    df["intersection_words_ocr"].fillna("").astype(str) + ". " +
    df["intersection_words_lang"].fillna("").astype(str) + ". " +
    df.apply(
        lambda x: x["original_text_cleaned"] if len(x["original_text_cleaned"]) < 500 
        else x["text_main_lang_cleaned"]if len(x["text_main_lang_cleaned"]) < 500
        else (remove_duplicates(x["text_main_lang_cleaned"]) if len(remove_duplicates(x["text_main_lang_cleaned"]).split()) < 80
        else x["tfidf_selection"]) ,axis = 1).astype(str)
    )

```


```python
df[df["word_selection"] == ""].shape
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
word_count = df['word_selection'].str.split().str.len()
sns.histplot(word_count)
plt.xlim([0,1000])
plt.show()

```


```python
df['word_selection'].str.split().str.len().sort_values(ascending = False)
```


```python
#df.to_pickle("dataset/dataset.pkl")
```

### Check Results


```python
#df = pd.read_pickle("dataset/dataset.pkl")
#df["code"] = df["code"].astype(str)
```


```python
empty_or_noise = df["word_selection"].str.len() < 10
print("noise or empty lines:", empty_or_noise.sum())
df = df[~empty_or_noise]
print(df.shape)

```


```python
from  utils.OCR_word_selection import show_images_from_barcode, get_big_words_from_image_clean
import re
```


```python
#i = np.random.choice(range(df.shape[0]))
#i = 10157 # maitre coq volaille 0215703025452
#i = 20865 #knorr 3011360020178
#i = 363851 #bolognese sauce - 5000354914829
#i = 65210 # andouilettes - 3278910707327
#i = 27073 # bonduelle conconmbre fromage blanc  -3083681008616
i = 151249 # macaroni everyday - 5400141165043

barcode = df["code"].iloc[i]
print(i)
print(df["word_selection"].iloc[i])
print(len(df["word_selection"].iloc[i].split()))
print(barcode)
show_images_from_barcode(barcode, df)
```


```python
big_words_dict[barcode]
```


```python
input_text = df["texts"].iloc[i]
final_selection = df["word_selection"].iloc[i]
text_clean = df["text_main_lang_cleaned"].iloc[i]
text_tf = df["tfidf_selection"].iloc[i]
text_insct_lang = df["intersection_words_lang"].iloc[i]
text_insct_ocr = df["intersection_words_ocr"].iloc[i]
barcode = df["code"].iloc[i]


print(barcode)
print(re.sub(r"\n", " ", input_text))
print("\n ____ final selection ____")
print(final_selection)
print("\n ____ text of main language after cleaning ____")
print(text_clean)
print("\n ____ text selection with tdidf score")
print(text_tf)
print("\n ____ text selection - intersection between languages ____")
print(text_insct_lang)
print("\n ____ text selection - intersection between OCRs ____")
print(text_insct_ocr)
print("\n ___ text selection - big words from image ____")
if barcode in big_words_dict:
    print(big_words_dict[barcode])
```


```python
barcode = "3560070851003"
get_big_words_from_image_clean(barcode, df = df)
```
