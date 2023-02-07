"""Script to shuffle off_categories dataset v3, generate v4."""

import gzip
import random
from contextlib import ExitStack
from pathlib import Path

import orjson
import tqdm

ROOT_DIR = Path("~/datasets/dataforgood2022/v3")
OUTPUT_DIR = Path("~/datasets/dataforgood2022/v4")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_jsonl(file_path: Path):
    open_fn = gzip.open if file_path.name.endswith(".gz") else open
    with open_fn(file_path, "rt") as f:
        yield from map(orjson.loads, f)


input_paths = {
    "dataset": ROOT_DIR / "predict_categories_dataset_products.jsonl.gz",
    "ocr": ROOT_DIR / "predict_categories_dataset_ocrs.jsonl.gz",
    "image_id": ROOT_DIR / "predict_categories_dataset_images_ids.jsonl.gz",
}

output_paths = {
    "dataset": OUTPUT_DIR / "predict_categories_dataset_products.jsonl.gz",
    "ocr": OUTPUT_DIR / "predict_categories_dataset_ocrs.jsonl.gz",
    "image_id": OUTPUT_DIR / "predict_categories_dataset_images_ids.jsonl.gz",
}

previous_count = None
for file_path in input_paths.values():
    count = sum(1 for _ in read_jsonl(file_path))
    print(f"{file_path=}, {count=}")
    if previous_count is not None and previous_count != count:
        raise ValueError("mismatch count: %d, %d", previous_count, count)
    previous_count = count

shuffled_ids = list(range(count))
random.shuffle(shuffled_ids)

with ExitStack() as stack:
    output_fp_map = {
        key: stack.enter_context(gzip.open(file_path, "wb"))
        for key, file_path in output_paths.items()
    }
    for data in tqdm.tqdm(
        zip(
            shuffled_ids,
            *(read_jsonl(file_path) for file_path in input_paths.values()),
        )
    ):
        id_, *items = data
        for key, item in zip(input_paths.keys(), items):
            output_fp = output_fp_map[key]
            output_fp.write(f"{id_:010},".encode() + orjson.dumps(item) + b"\n")


# sort each file with zcat predict_categories_dataset_products.jsonl.gz | sort -k '1n' -t ',' -o predict_categories_dataset_products_shuffled.jsonl.gz
# then call remove_delimiter()
def remove_delimiter():
    for file_path in output_paths.values():
        with gzip.open(
            file_path.with_stem(file_path.stem + "_shuffled"), "wb"
        ) as output_f, gzip.open(file_path, "rt") as input_f:
            previous_id = None
            for item in tqdm.tqdm(input_f):
                id_, data = item.split(",", maxsplit=1)
                if previous_id is not None and previous_id == id_:
                    raise ValueError()
                previous_id = id_
                output_f.write(data.encode("utf-8"))
