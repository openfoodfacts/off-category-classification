from bokeh.plotting import figure, show
import numpy as np

from category_classification.data_utils import create_dataframe
from utils.preprocess import count_categories

# Construct the complete dataset from the train/val/test datasets.
df = create_dataframe("train", "xx")
df = df.append(create_dataframe("test", "xx"))
df = df.append(create_dataframe("val", "xx"))

cat_counts = count_categories(df)

sorted_counts = {k: v for k, v in sorted(cat_counts.items(), key=lambda item: item[1])}
print(type(sorted_counts))


file1 = open('category_counts.txt', 'w')

for k, v in sorted_counts.items():
	file1.write("{},{}\n".format(k, v))

file1.close()





# hist,edges = np.histogram([1,2,2, 1,1,1,1,1,1], bins="auto")
 
# print(len(hist))
# print(hist)

# p = figure(title="Distribution of Category Datasets", x_axis_label="Dataset Size", y_axis_label="Categories")
# p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

# show(p)


# p.quad(bottom=0, top=cat_counts)

# # add circle renderer with additional arguments
# p.circle(
#     x,
#     y,
#     legend_label="Objects",
#     fill_color="red",
#     fill_alpha=0.5,
#     line_color="blue",
#     size=80,
# )

# # show the results
# show(p)