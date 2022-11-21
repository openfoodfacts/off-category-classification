"""Get images ids from the product json lines

Run it when you are in the data folder using python,
it will generate a list of id, that is then to be used with gen_categories_logo_dataset_01_robotoff.py
(on robotoff server)
"""
import gzip
import json


IN_FILE = "predict_categories_dataset_products.jsonl.gz"
OUT_FILE = "predict_categories_dataset_images_ids.jsonl.gz"


def iter_data():
    for line in gzip.open(IN_FILE, "rt"):
        yield json.loads(line)


def images_ids(data):
    barcode = data.get("code")
    images = data.get("images")
    if barcode is None or not images:
        # we always put a line, for we want to be able to merge files line by line
        return [barcode, []]
    ids = [image.get("imgid") for lang_images in images.values() for image in lang_images.values()]
    return [barcode, ids]

if __name__ == "__main__":
    with gzip.open(OUT_FILE, "wt") as out:
        for i, data in enumerate(iter_data()):
            ids = images_ids(data)
            out.write(json.dumps(ids) + "\n")
            if i and i % 10000 == 0:
                print("Processed %d lines" % i)