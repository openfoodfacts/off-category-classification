import pathlib


PROJECT_DIR = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

CATEGORY_FR_TEST_PATH = DATA_DIR / "category_fr.test.jsonl.gz"
CATEGORY_FR_TRAIN_PATH = DATA_DIR / "category_fr.train.jsonl.gz"

CATEGORY_XX_VAL_PATH = DATA_DIR / "category_xx.test.jsonl.gz"

CATEGORY_TAXONOMY_PATH = DATA_DIR / "categories.full.json"
INGREDIENTS_TAXONOMY_PATH = DATA_DIR / "ingredients.full.json"

PRODUCT_NAME_VOC_NAME = "product_name_voc.json"
CONFIG_NAME = "config.json"
CATEGORY_VOC_NAME = "category_voc.json"
CATEGORY_TAXONOMY_NAME = "category_taxonomy.json"
INGREDIENT_VOC_NAME = "ingredient_voc.json"
