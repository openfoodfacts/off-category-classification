# DELETE THIS!

import numpy as np
from bokeh.plotting import figure, show

from category_classification.data_utils import create_dataframe
from utils.preprocess import count_categories

# Construct the complete dataset from the train/val/test datasets.
df = create_dataframe("train", "xx")
df = df.append(create_dataframe("test", "xx"))
df = df.append(create_dataframe("val", "xx"))

cat_counts = count_categories(df)

sorted_counts = {k: v for k, v in sorted(cat_counts.items(), key=lambda item: item[1])}
print(type(sorted_counts))


file1 = open("category_counts.txt", "w")

for k, v in sorted_counts.items():
    file1.write("{},{}\n".format(k, v))

file1.close()
