"""Dump logo data from robotoff

To run it in prod:
docker-compose run --rm \
    -v $(pwd):/tmp/dump \
    -v /absolute/path/to/script.py:/opt/robotoff/dump_script.py \
    -v /absolute/path/to/predict_categories_dataset_images_ids.jsonl.gz:/opt/robotoff/predict_categories_dataset_images_ids.jsonl.gz \
    workers poetry run python ./dump_script.py
"""
import gzip
import json
import operator
from datetime import date, datetime
from functools import reduce

from playhouse.shortcuts import model_to_dict
from robotoff.models import db, LogoAnnotation, ImagePrediction, ImageModel


IDS_FILE = "predict_categories_dataset_images_ids.jsonl.gz"

OUT_FILE = "/tmp/dump/logos_robotoff.jsonl.gz"


def batch_ids_iter(ids_file, size=200):
    batch = []
    for line in gzip.open(ids_file, "rt"):
        batch.append(json.loads(line))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_items(data):
    for k, v in data.items():
        if isinstance(v, (date, datetime)):
            v = v.isoformat()
        yield k, v


def annotation_data(annotation):
    annotation = model_to_dict(annotation)
    # remove image prediction data
    image_prediction = annotation.pop("image_prediction")
    # flatten dict
    image = image_prediction.pop("image")
    data = {}
    data["image_prediction_id"] = image_prediction.pop("id")
    data["image_id"] = image.pop("id")
    # update, highest priority last (as update do replace)
    data.update(image)
    data.update(annotation)
    data = dict(normalize_items(data))
    return data

def iter_data(batch_iter):
    with db:
        base_query = LogoAnnotation.select(
            LogoAnnotation, ImagePrediction, ImageModel
        ).join(
            ImagePrediction
        ).join(
            ImageModel
        )
        for batch in batch_iter:
            queries = [
                (LogoAnnotation.image_prediction.image.barcode == barcode) &
                LogoAnnotation.image_prediction.image.image_id.in_(ids)
                for barcode, ids in batch if ids
            ]
            if queries:
                query = reduce(operator.or_, queries)
            for annotation in base_query.filter(query).iterator():
                yield annotation_data(annotation)


def save_data(data_iter):
    with gzip.open(OUT_FILE, "wt") as f:
        for data in data_iter:
            f.write(f"{json.dumps(data)}\n")


if __name__ == "__main__":
    save_data(iter_data(batch_ids_iter(IDS_FILE)))