"""Get crops that correspond to logos on OFF server
"""
import gzip
import io
import json
import os
import tarfile
from datetime import datetime

from PIL import Image


IN_FILE = "logos_robotoff.jsonl.gz"
OUT_FILE = "logos.tar"
ERR_FILE = "logos-err.json"

IMAGES_ROOT = "/srv2/off/html/images/products/"


def iter_logos(in_file):
    for line in gzip.open(in_file, "rt"):
        data = json.loads(line)
        for logo_data in data.get("logos", []):
            bbox = logo_data.get("bounding_box")
            src = logo_data.get("source_image", "")
            logo_id = logo_data.get("id", "")
            if bbox and src:
                path = IMAGES_ROOT + src
                yield logo_id, path, bbox


def crop_image(path, bbox):
    try:
        image = Image.open(path)
        stat = os.stat(path)
    except:
        return None
    # convert bbox, this should match
    # code at robotoff.api.ImageCropResource
    # and at robotoff.model.get_crop_image_url
    y_min, x_min, y_max, x_max = bbox
    left, right, top, bottom = (
        x_min * image.width,
        x_max * image.width,
        y_min * image.height,
        y_max * image.height,
    )
    # crop
    logo = image.crop((left, top, right, bottom))
    return logo, stat


def add_logo(archive, logo, stat, logo_id):
    stream = io.BytesIO()
    logo.save(stream, "JPEG")
    tarinfo = tarfile.TarInfo(name="%s.jpg" % logo_id)
    tarinfo.size = stream.tell()
    tarinfo.mode = 0x666
    tarinfo.type = tarfile.REGTYPE
    tarinfo.mtime = stat.st_mtime
    tarinfo.uid = stat.st_uid
    tarinfo.gid = stat.st_gid
    tarinfo.uname = tarinfo.gname = "off"
    stream.seek(0)
    archive.addfile(tarinfo, stream)


def harvest_logos(in_file, out_file, err_file):
    if not os.path.exists(out_file):
        # create empty archive
        with tarfile.open(out_file, "w"):
            pass
    with tarfile.open(out_file, "r") as archive:
        if os.path.exists(out_file):
            existing = set([
                int(name.rsplit(".", 1)[0]) for name in archive.getnames()
            ])
        else:
            existing = set([])
    errs = []
    with tarfile.open(out_file, "a") as archive:
        for i, (logo_id, path, bbox) in enumerate(iter_logos(in_file)):
            if logo_id not in existing:
                try:
                    logo, stat = crop_image(path, bbox)
                    add_logo(archive, logo, stat, logo_id)
                except Exception as e:
                    errs.append((logo_id, str(e)))
            if i % 10000 == 0:
                print("%s done %d with %d errors" % (datetime.now().isoformat(), i, len(errs)))
    if errs:
        print("Met %d errors while processing" % len(errs))
        json.dump(errs, open(ERR_FILE, "a"))


if __name__ == "__main__":
    harvest_logos(IN_FILE, OUT_FILE, ERR_FILE)