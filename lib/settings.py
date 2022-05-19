import pathlib


PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

CATEGORY_FR_TEST_PATH = DATA_DIR / "category_fr.test.jsonl.gz"
CATEGORY_FR_TRAIN_PATH = DATA_DIR / "category_fr.train.jsonl.gz"

CATEGORY_XX_TEST_PATH = DATA_DIR / "category_xx.test.jsonl.gz"

CATEGORY_TAXONOMY_PATH = DATA_DIR / "categories.full.json"

CONFIG_NAME = "config.json"
CATEGORY_VOC_NAME = "category_voc.json"
CATEGORY_TAXONOMY_NAME = "category_taxonomy.json"
