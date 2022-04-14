"""
Generate a dataset for categories prediction

This script is meant to be run on production server.

For now it must remain python3.5 compatible.
"""
import collections
import gzip
import itertools
import json
import os.path
import random
import re
import tarfile
import textwrap
import time
from datetime import datetime, timedelta


def iter_data(data_dump):
    """iter data in data dump"""
    for line in gzip.open(data_dump, "rt"):
        yield json.loads(line)


def agribalyse_categories_iter(cat_dump):
    """Get all agribalyse categories"""
    for cat, cat_data in json.load(open(cat_dump)).items():
        agribalyse = cat_data.get("agribalyse_food_code")
        if agribalyse:
            yield cat, agribalyse


def file_size(fpath):
    size = os.stat(fpath).st_size
    measures = ["o", "KB", "MB", "GB", "TB"]
    measure_index = 0
    while size > 1024 and measure_index + 1 < len(measures):
        size = size / 1024
        measure_index += 1
    return "%.1f %s" % (size, measures[measure_index])


# from robotoff
BARCODE_PATH_REGEX = re.compile(r"^(...)(...)(...)(.*)$")


def split_barcode(barcode):
    if not barcode.isdigit():
        raise ValueError("unknown barcode format: {}".format(barcode))

    match = BARCODE_PATH_REGEX.fullmatch(barcode)

    if match:
        return [x for x in match.groups() if x]

    return [barcode]


def generate_image_path(barcode: str, image_id: str) -> str:
    splitted_barcode = split_barcode(barcode)
    return "/{}/{}.jpg".format("/".join(splitted_barcode), image_id)


class DataPathes:

    data_dump = "/srv2/off/html/data/openfoodfacts-products.jsonl.gz"
    cat_dump = "/srv2/off/html/data/taxonomies/categories.full.json"
    images_root = "/srv2/off/html/images/products"
    target_dir = "/srv2/off/html/data"
    files_prefix = "alex/predict_categories_dataset"

    @property
    def products_file(self):
        return self.target_dir + "/" + self.files_prefix + "_products.jsonl.gz"

    @property
    def ocrs_file(self):
        return self.target_dir + "/" + self.files_prefix + "_ocrs.jsonl.gz"

    @property
    def images_file(self):
        return self.target_dir + "/" + self.files_prefix + "_images.tar"

    @property
    def documentation_file(self):
        return self.target_dir + "/" + self.files_prefix + "_documentation.txt"


class IdsSelector:

    # template for documentation. See generates_documentation
    DOCUMENTATION = """
    This dataset contains {total} product, that were randomly selected to contains, roughly:

    - {kind_sizes[popular]} popular products
    - {num_each_agri} products for each category linked to agribalyse
    - {kind_sizes[misc_agri]} products with various categories linked to agribalyse
    - {kind_sizes[other]} other products
    """
    documentation = "Not generated"  # we will generate from DOCUMENTATION when we have quantities

    # this is the target we have
    quantities = {
        # total items
        "total": 150000,
        # percent of items equilibrated with agribalyse cat (there are 2494 entries)
        "each_agribalyse": .33,
        # percent of items with an agribalyse but randomly
        "misc_agribalyse": .2,
        # percent of random but weighted by scans
        "many_scans": .2,
    }
    seed = 42

    data_pathes = DataPathes()

    def __init__(self):
        self._caches = {}

    @property
    def agribalyse_categories(self):
        """dict of agribalyse categories, key is category, value is agribalyse food_code"""
        try:
            value = self._caches["agribalyse_categories"]
        except KeyError:
            value = dict(agribalyse_categories_iter(self.data_pathes.cat_dump))
            self._caches["agribalyse_categories"] = value
        return value

    @property
    def agribalyse_cats(self):
        """All categories with an agribalyse food_code

        :return set:
        """
        try:
            value = self._caches["agribalyse_cats"]
        except KeyError:
            value = set(cat for cat in self.agribalyse_categories.keys())
            self._caches["agribalyse_cats"] = value
        return value

    def product_agribalyse(self, product):
        """Get categories of a product that have an agribalyse"""
        cats = set(product.get("categories_tags", [])) & self.agribalyse_cats
        return cats

    @property
    def num_each_agri(self):
        """number of ids to select for each category related to agrybalise"""
        return self.quantities["total"] * self.quantities["each_agribalyse"] / len(self.agribalyse_cats)

    @property
    def kind_sizes(self):
        """Compute number of sample we want per data kind

        :return dict: size by kind
        """
        try:
            kind_sizes = self._caches["kind_sizes"]
        except KeyError:
            # total elements we want by kind
            num_agri = self.num_each_agri
            num_misc_agri = self.quantities["total"] * self.quantities["misc_agribalyse"]
            num_popular = self.quantities["total"] * self.quantities["many_scans"]
            num_other = self.quantities["total"] - num_agri - num_misc_agri - num_popular
            kind_sizes = [("popular", num_popular), ("misc_agri", num_misc_agri), ("other", num_other)]
            kind_sizes.extend((cat, num_agri) for cat in self.agribalyse_categories)
            kind_sizes = {k: int(v) for k, v in kind_sizes if v >= 1}
            self._caches["kind_sizes"] = kind_sizes
        return kind_sizes

    def select_ids(self):
        """iter data and select ids that we will retain as product samples

        :return: a tuple containing:
        - a dict with a list of (line_num, ids) by kind
        - a dict of wanted size by kind
        """
        # ids is a dict that will associate each kind to a list of ids
        ids = collections.defaultdict(list)
        # ids seen is the total weights of already seen items
        ids_seen = collections.defaultdict(lambda: 0)
        kind_sizes = self.kind_sizes

        def add_randomly(data, kind, weight=1):
            """randomly add to a list of ids (with key kind)

            :param int weight: is an eventual weight for this item.

            Does not handle trimming the list (to do it once in a while instead).

            :return bool: True if inserted
            """
            # I'm really not 100% sure my methodology is fair against ids !
            # It has the advantage of being streamed and not taking much memory
            # and also to immediately decide if we retain an id or not !
            # I made an empirical validation of simple case (weight=1) here:
            # https://gist.github.com/alexgarel/9de2f548414ddeea1ed5279800d98ddd
            seen = ids_seen[kind] + weight
            index = random.randrange(max(int(seen / weight if weight != 1 else seen), 1))
            ids_seen[kind] = seen
            success = index < kind_sizes[kind]
            if success:
                ids[kind].insert(index, data)
            return success

        def trim():
            """trim ids list to keep only necessary items"""
            for kind, size in kind_sizes.items():
                ids[kind] = ids[kind][:size]
        start = time.monotonic()
        for i, product in enumerate(iter_data(self.data_pathes.data_dump)):
            # trim from time to time
            if i and i % 10000 == 0:
                trim()
            if i % 100000 == 0:
                print("visited", i, "ids", "in", int(time.monotonic() - start), "seconds")
            agri_cats = self.product_agribalyse(product)
            scans = product.get("unique_scans_n", 1)
            data = (i, product["code"])
            # first try to add to popular
            if add_randomly(data, "popular", scans / 10):
                continue
            # then to agri_balyse
            for cat in agri_cats:
                if add_randomly(data, cat):
                    break
            else:
                # then misc agri_balyse
                if agri_cats:
                    if add_randomly(data, "misc_agri"):
                        continue
                # then other
                add_randomly(data, "other")
        # trim at the end
        trim()
        return ids

    def check_ids(self, ids):
        """check we have as many ids as wanted)"""
        kind_sizes = self.kind_sizes
        for kind, kind_ids in ids.items():
            if len(kind_ids) < kind_sizes[kind]:
                print("Less items than expected for {kind}: {size} / {wanted}".format(
                    kind=kind, size=len(kind_ids), wanted=kind_sizes[kind]
                ))

    def generates_documentation(self, ids):
        """Generates the documentation string"""
        self.documentation = self.DOCUMENTATION.format(
            kind_sizes=self.kind_sizes,
            total=sum(len(v) for v in ids.values()),
            num_each_agri=self.num_each_agri,
        )
        # remove first empty line and dedent
        self.documentation = textwrap.dedent(self.documentation[1:])
        with open(self.data_pathes.documentation_file, "w") as f:
            f.write(self.documentation)

    def ids_by_kind(self, ids_path=None):
        """try to loads ids from ids_path or generate them

        :return dict: ids list by kind
        """
        if ids_path and not ids_path.endswith("json.gz"):
            raise AssertionError("ids path must end with json.gz")
        if ids_path and os.path.exists(ids_path):
            ids_by_kind = json.load(gzip.open(ids_path, "rt"))
        else:
            # try to be reproducible
            random.seed(self.seed)
            ids_by_kind = self.select_ids()
            if ids_path:
                json.dump(ids_by_kind, gzip.open(ids_path, "wt"))
        self.check_ids(ids_by_kind)
        self.generates_documentation(ids_by_kind)
        return ids_by_kind

    def all_ids(self, ids_by_kind):
        """all ids as a single sorted list (sorted by line number)
        """
        return sorted(itertools.chain.from_iterable(ids_by_kind.values()))


class DataHarvester:
    """Harvest data corresponding to certain ids"""

    # documentation template, see documentation property
    DOCUMENTATION = """

    - products file: {products_file} ({products_file_size}),
      a jsonl, contains data about each product as a dict.
      Available fields are:
      - code: product barcode
      - categories_tags: normalized categories
      - categories_hierarchy: list of ordered categories from parent to children
      - categories_properties: properties computed from categories
      - product_name_xx: product name in language xx
      - nutriments: data about nutriments
      - ingredients_xx: parsed ingredients in language xx
      - ingredients_text_xx: original text for ingredients in language xx
      - images is a dict by image language code (or xx if no language code),
        which in turns contains dict by image kind, (front, ingredients, packaging, other_1..n)
        which contains data about an image.

    - ocrs file: {ocrs_file} ({ocrs_file_size}),
      contains a line per product, in same order as products file.
      - code: product barcode
      - ocrs: has the same structure as `images` in products file,
        but contains ocr data (as returned by Google vision) instead of image data
      - no_ocr: returns images id for which we did not find any OCR

    - images file: {images_file} ({images_file_size}),
      is a tar archive with all images contained in products file `images files`.
      Each file has path barcode/imgid.jpg.
      Images are all 400 px width images.
    """

    max_image_monthes = 12  # max age in monthes
    image_size = "400"  # image size, if None, takes original

    data_pathes = DataPathes()
    product_keys = set([
        "code",
        # target - we put all info to help diagnose easily
        "categories_tags", "categories_hierarchy", "categories_properties",
        # features
        "nutriments",
        "ingredients_{LC}", "ingredients_text_{LC}",
        "product_name_{LC}",
    ])
    lang_field = re.compile(".*_[a-z]{2}^")

    def __init__(self):
        self._caches = {}

    @property
    def image_min_ts(self):
        try:
            ts = self._caches["image_min_ts"]
        except KeyError:
            ts = (datetime.now() - timedelta(days=self.max_image_monthes * 31)).timestamp()
            self._caches["image_min_ts"] = int(ts)
        return ts

    def product_data(self, product):
        """Only keep relevant info for product
        """
        data = {}
        for key, value in product.items():
            if self.lang_field.match(key):
                match_key = key[:-3] + "_{LC}"
            else:
                match_key = key
            if match_key in self.product_keys:
                data[key] = value
        return data

    def get_images_data(self, product):
        images = collections.defaultdict(dict)
        # keep only selected and not too old
        min_ts = self.image_min_ts
        product_images = product.get("images", {})
        for key, image in product_images.items():
            is_selected = not key.isnumeric()
            # `or 0` because sometimes there are None in the dict
            recent = int(image.get("uploaded_t") or 0) > min_ts
            if is_selected or recent:
                if not is_selected:
                    # add imageid
                    image["imgid"] = key
                    kind = "other_%s" % key
                    lang = "xx"
                else:
                    kind, *lang = key.rsplit("_", 1)
                    # FIXME: not sure, maybe there is a default lang ?
                    lang = lang[0] if lang else "xx"
                images[lang][kind] = image
        # remove images in "other" that are already in a "selected" kind
        images_xx = images["xx"]  # "other" is only in "xx"
        to_remove = set()
        for lang, lang_data in images.items():
            for kind, image in lang_data.items():
                if not kind.startswith("other"):
                    key = "other_%s" % image["imgid"]
                    if key in images_xx:
                        to_remove.add(key)
        for key in to_remove:
            del images_xx[key]
        return images

    def images_ocr(self, product, images_data):
        code = product["code"]
        ocr_data = collections.defaultdict(dict)
        no_ocr = []
        for lang, lang_data in images_data.items():
            for kind, image in lang_data.items():
                path = "%s/%s" % (
                    self.data_pathes.images_root, generate_image_path(code, image["imgid"])
                )
                # remove extension
                path = path.rsplit(".", 1)[0]
                try:
                    if os.path.exists(path + ".json"):
                        ocr_data[lang][kind] = json.load(open(path + ".json"))
                    elif os.path.exists(path + ".json.gz"):
                        ocr_data[lang][kind] = json.load(gzip.open(path + ".json.gz", "rt"))
                    else:
                        no_ocr.append(image["imgid"])
                except json.JSONDecodeError:
                    # this happens, once in a while…
                    no_ocr.append(image["imgid"])
        return {"code": code, "ocrs": ocr_data, "no_ocr": no_ocr}

    def images_iter(self, product, images_data):
        code = product["code"]
        not_found = []
        for lang, lang_data in images_data.items():
            for kind, image in lang_data.items():
                path = "%s/%s" % (
                    self.data_pathes.images_root, generate_image_path(code, image["imgid"])
                )
                if self.image_size:
                    path, ext = path.rsplit(".", 1)
                    path = "%s.%s.jpg" % (path, self.image_size)
                    if os.path.exists(path):
                        yield code, image["imgid"], path
                    else:
                        not_found.append(image["imgid"])
        # not so beautiful, yield not found as last element
        yield not_found

    def get_data(self, product):
        """get data for a product"""
        # product
        product_data = self.product_data(product)
        product_data["images"] = images_data = self.get_images_data(product)
        # ocr files
        ocr_data = self.images_ocr(product, images_data)
        # images
        images = list(self.images_iter(product, images_data))
        # last item is number of unfound items
        product_data["images_not_found"] = images.pop()
        return product_data, ocr_data, images

    def iter_data(self, ids):
        if not ids:
            return []
        ids = list(reversed(ids))  # copy to consume, and change order to use pop
        next_line, code = ids.pop()
        for i, product in enumerate(iter_data(self.data_pathes.data_dump)):
            if i != next_line:
                continue
            yield self.get_data(product)
            if not ids:
                break
            next_line, code = ids.pop()

    def generates_documentation(self):
        self.documentation = self.DOCUMENTATION.format(
            products_file=os.path.basename(self.data_pathes.products_file),
            products_file_size=file_size(self.data_pathes.products_file),
            ocrs_file=os.path.basename(self.data_pathes.ocrs_file),
            ocrs_file_size=file_size(self.data_pathes.ocrs_file),
            images_file=os.path.basename(self.data_pathes.images_file),
            images_file_size=file_size(self.data_pathes.images_file),
        )
        # remove first empty line and dedent
        self.documentation = textwrap.dedent(self.documentation[1:])
        with open(self.data_pathes.documentation_file, "a") as f:
            f.write(self.documentation)

    def __call__(self, ids):
        # select data from mongo dump
        start = time.monotonic()
        with gzip.open(self.data_pathes.products_file, "wt") as products, \
                gzip.open(self.data_pathes.ocrs_file, "wt") as ocrs, \
                tarfile.open(self.data_pathes.images_file, "w") as images:
            total = len(ids)
            steps = int(total / 20)
            for i, (product, ocr, images_data) in enumerate(self.iter_data(ids)):
                products.write(json.dumps(product) + "\n")
                ocrs.write(json.dumps(ocr) + "\n")
                for code, imgid, path in images_data:
                    images.add(name=path, arcname="%s/%s.jpg" % (code, imgid), recursive=False)
                if i and i % steps == 0:
                    percent = (i * 100) / total
                    spent_time = time.monotonic() - start
                    print("%d (%.0f%%) done in %.0f seconds" % (i, percent, spent_time))
        self.generates_documentation()


def generate_data():
    start = time.monotonic()
    ids_selector = IdsSelector()
    ids_by_kind = ids_selector.ids_by_kind("_ids_by_kind.json.gz")
    ids = ids_selector.all_ids(ids_by_kind)
    ids_time = time.monotonic()
    print("Got", len(ids), "ids in", "%.0f seconds" % (ids_time - start))
    DataHarvester()(ids)
    harvest_time = time.monotonic()
    print("Got data in ", "%.0f seconds" % (harvest_time - start))


if __name__ == "__main__":
    generate_data()
