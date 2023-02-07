""""Script used to de-duplicate v2 dataset, generates v3 dataset."""

import gzip
from collections import Counter
from pathlib import Path

import orjson


def read_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                yield orjson.loads(l)


def dedup(file_path: Path, output_path: Path):
    seen = Counter()
    with gzip.open(output_path, "wb") as f:
        for item in read_json(file_path):
            barcode = item["code"]
            seen[barcode] += 1
            if seen[barcode] > 1:
                continue
            f.write(orjson.dumps(item) + b"\n")

    for k, count in seen.most_common():
        if count > 1:
            print(k, count)


dedup(
    Path("~/datasets/dataforgood2022/v2/predict_categories_dataset_ocrs.jsonl.gz"),
    Path("~/datasets/dataforgood2022/v3/predict_categories_dataset_ocrs.jsonl.gz"),
)
