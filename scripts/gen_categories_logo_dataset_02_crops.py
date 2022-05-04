"""Get crops that correspond to logos on OFF server
"""
import gzip
import io
import json
import os
import tarfile

from PIL import Image


IN_FILE = "logos_robotoff.jsonl.gz"
OUT_FILE = "logos.tar"

IMAGES_ROOT = "/srv2/off/html/images/products/"


def iter_logos(in_file):
    for line in gzip.open(in_file, "rt"):
        data = json.loads(line)
        bbox = data.get("bounding_box")
        src = data.get("source_image", "")
        prediction_id = data.get("image_prediction_id", "")
        if bbox and src:
            path = IMAGES_ROOT + src
            yield prediction_id, path, bbox


def crop_image(path, bbox):
    try:
        image = Image.open(path)
        stat = os.stat(path)
    except:
        return None
    # convert bbox
    bbox_pixels = [int(round(x_float * size)) for x_float, size in zip(bbox, image.size * 2)]
    # crop
    logo = image.crop(bbox_pixels)
    return logo, stat

def add_logo(archive, logo, stat, prediction_id):
    stream = io.BytesIO()
    logo.save(stream, "JPEG")
    tarinfo = tarfile.TarInfo(name="%s.jpg" % prediction_id)
    tarinfo.size = stream.tell()
    tarinfo.mode = 0x666
    tarinfo.type = tarfile.REGTYPE
    tarinfo.mtime = stat.st_mtime
    tarinfo.uid = stat.st_uid
    tarinfo.gid = stat.st_gid
    tarinfo.uname = tarinfo.gname = "off"
    stream.seek(0)
    archive.addfile(tarinfo, stream)


if __name__ == "__main__":
    with tarfile.open(OUT_FILE, "w") as archive:    
        for prediction_id, path, bbox in iter_logos(IN_FILE):
            logo, stat = crop_image(path, bbox)
            add_logo(archive, logo, stat, prediction_id)
