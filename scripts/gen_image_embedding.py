import gzip
import json
import logging
import re
from pathlib import Path
from typing import Optional, Set

import h5py
import numpy as np
import requests
import torch
import tqdm
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel

session = requests.Session()


BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def generate_image_path(barcode: str, image_id: str) -> str:
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)
    splitted_barcode = [x for x in match.groups() if x] if match else [barcode]
    return "/{}/{}.jpg".format("/".join(splitted_barcode), image_id)


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    @param dict fmt_dict: Key: logging format attribute pairs. Defaults to {"message": "message"}.
    @param str time_format: time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    @param str msec_format: Microsecond formatting. Appended at the end. Default: "%s.%03dZ"
    """

    available_fields = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def __init__(
        self,
        fmt_dict: dict = None,
        time_format: str = "%Y-%m-%dT%H:%M:%S",
        msec_format: str = "%s.%03dZ",
        default_fields: Optional[list[str]] = None,
    ):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None
        self.default_fields = default_fields or {
            "levelname",
            "message",
            "msg",
            "name",
            "asctime",
        }

    def format(self, record) -> str:
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        message_dict = {
            k: v
            for k, v in record.__dict__.items()
            if (k in self.default_fields or k not in self.available_fields)
        }

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
json_handler = logging.FileHandler("generate_embeddings.log")
json_formatter = JsonFormatter()
json_handler.setFormatter(json_formatter)
json_handler.setLevel(logging.INFO)
logger.addHandler(json_handler)

# Define a custom dataset class that downloads the logos
class ImageDataset(Dataset):
    def __init__(self, dataset_path: Path, seen_set: Optional[set] = None):
        seen_set = seen_set or set()
        self.image_ids = []
        with gzip.open(dataset_path, "rt") as f:
            for barcode, image_ids in map(json.loads, f):
                for image_id in image_ids:
                    key = f"{barcode}_{image_id}"
                    if key in seen_set:
                        continue
                    seen_set.add(key)
                    self.image_ids.append((barcode, image_id))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Download the logo and return it
        barcode, image_id = self.image_ids[idx]
        image = download_image(barcode, image_id)
        return (f"{barcode}_{image_id}", image)


def collate_fn(batch):
    return [item for item in batch if item[1] is not None]


def get_seen_set(hdf5_path: Path) -> Set[str]:
    if not hdf5_path.is_file():
        return set()
    logger.info("fetching seen set...")
    with h5py.File(hdf5_path, "r") as f:
        return set(f["external_id"])


def download_image(barcode: str, image_id):
    logger.info("downloading image", extra={"barcode": barcode, "image_id": image_id})
    source_image = generate_image_path(barcode, image_id)
    try:
        raw_image = session.get(
            f"https://images.openfoodfacts.org/images/products{source_image}",
            stream=True,
        ).raw
    except Exception as e:
        logger.error(
            "could not download image", extra={"source_image": source_image}, exc_info=e
        )
        return None

    try:
        image = Image.open(raw_image)
        image.load()
    except Exception as e:
        logger.error(
            "could not load image", extra={"source_image": source_image}, exc_info=e
        )
        return None


def embed_images(images, processor, model, device):
    pixel_values = processor(
        images=images, return_tensors="pt", padding=True
    ).pixel_values

    with torch.inference_mode():
        outputs = model(
            **{
                "pixel_values": pixel_values.to(device),
                "attention_mask": torch.from_numpy(
                    np.ones((len(images), 2), dtype=int)
                ).to(device),
                "input_ids": torch.from_numpy(
                    np.ones((len(images), 2), dtype=int) * [49406, 49407]
                ).to(device),
            }
        )
        return outputs.image_embeds.cpu().numpy()


def main(
    dataset_path: Path, output_path: Path, batch_size: int = 2, num_workers: int = 1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPImageProcessor()
    dataset = ImageDataset(dataset_path, get_seen_set(output_path))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    output_exists = output_path.is_file()

    with h5py.File(output_path, "a") as f:
        if not output_exists:
            id_dset = f.create_dataset(
                "external_id", (len(dataset),), dtype=h5py.string_dtype(), chunks=True
            )
            embedding_dset = f.create_dataset(
                "embedding", (len(dataset), 512), dtype="f", chunks=True
            )
            offset = 0
            f.flush()
        else:
            embedding_dset = f["embedding"]
            id_dset = f["external_id"]
            non_zero_indexes = np.flatnonzero(id_dset[:])
            offset = int(non_zero_indexes[-1]) + 1
            assert id_dset[offset] == 0

        logger.info(f"offset is %s", offset)

        for embedding_batch, id_batch in tqdm.tqdm(
            generate_embedding_batch(data_loader, model, processor, device)
        ):
            slicing = slice(offset, offset + len(embedding_batch))
            assert all(len(id) == 0 for id in id_dset[slicing])
            id_dset[slicing] = id_batch
            embedding_dset[slicing] = embedding_batch
            offset += len(embedding_batch)


def generate_embedding_batch(data_loader, model, processor, device):
    for batch in data_loader:
        image_ids, images = zip(*batch)
        id_batch = np.array(image_ids)
        embedding_batch = embed_images(images, processor, model, device)
        yield embedding_batch, id_batch


if __name__ == "__main__":
    typer.run(main)
