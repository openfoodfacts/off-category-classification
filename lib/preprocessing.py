import itertools
import re
import string
from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Set, Tuple

from flashtext import KeywordProcessor

from lib.taxonomy import Taxonomy

from .constant import EXCLUDE_LIST_CATEGORIES
from .text_utils import fold, get_tag


def remove_untaxonomized_values(value_tags: List[str], taxonomy: Taxonomy) -> List[str]:
    return [value_tag for value_tag in value_tags if value_tag in taxonomy]


def infer_missing_category_tags(
    category_tags: List[str], taxonomy: Taxonomy
) -> Set[str]:
    all_categories = set()
    for category_tag in category_tags:
        category_node = taxonomy[category_tag]
        if category_node:
            all_categories.add(category_node.id)
            all_categories |= set(x.id for x in category_node.get_parents_hierarchy())
    return all_categories


def transform_category_input(category_tags: List[str], taxonomy: Taxonomy) -> List[str]:
    category_tags = remove_untaxonomized_values(category_tags, taxonomy)
    # first get deepest nodes, as we're removing some excluded categories below,
    # we don't want parent categories of excluded categories to be kept in the list
    category_tags = [
        node.id
        for node in taxonomy.find_deepest_nodes(
            [taxonomy[category_tag] for category_tag in category_tags]
        )
    ]
    # Remove excluded categories
    category_tags = [
        category_tag
        for category_tag in category_tags
        if category_tag not in EXCLUDE_LIST_CATEGORIES
    ]
    # Generate the full parent hierarchy, without adding again excluded
    # categories
    return [
        category_tag
        for category_tag in infer_missing_category_tags(category_tags, taxonomy)
        if category_tag not in EXCLUDE_LIST_CATEGORIES
    ]


def transform_ingredients_input(
    ingredients: List[Dict], taxonomy: Taxonomy
) -> List[str]:
    # Only keep nodes of depth=1 (i.e. don't keep sub-ingredients)
    # While sub-ingredients may be interesting for classification, enough signal is already
    # should already be present in the main ingredient, and it makes it more difficult to
    # take ingredient order into account (as we don't know if sub-ingredient #2 of
    # ingredient #1 is more present than sub-ingredient #1 of ingredient #2)
    return remove_untaxonomized_values(
        [get_tag(ingredient["id"]) for ingredient in ingredients], taxonomy
    )


def transform_nutrition_input(value: Optional[float], nutriment_name: str) -> float:
    if value is None:
        return -1

    if nutriment_name == "energy-kcal":
        if value >= 3800 or value < 0:
            # Too high to be true
            return -2

    elif value < 0 or value >= 101:
        # Remove invalid values
        return -2

    return value


MULTIPLE_SPACES_REGEX = re.compile(r" {2,}")


def transform_ocr_ingredients_input(
    value: Dict, processor: KeywordProcessor, debug: bool = False
):
    texts = []
    for ocr_data in value.values():
        text = ocr_data["text"].replace("\n", " ")
        text = MULTIPLE_SPACES_REGEX.sub(" ", text)
        texts.append(text)

    full_text = "|".join(texts)
    matches = []
    for keys, _, __ in extract_ingredient_from_text(processor, full_text):
        keys = (
            [f"{node_id}|{lang}" for (node_id, lang) in keys]
            if debug
            else [node_id for (node_id, _) in keys]
        )
        matches += keys
    return sorted(set(matches), key=matches.index)


INGREDIENT_ID_EXCLUDE_LIST = {
    "en:n",
    "en:no1",
    "en:no2",
    "en:no3",
    "en:no4",
    "en:no5",
    "en:no6",
    "en:no7",
    "en:no8",
    "en:no9",
    "en:no10",
    "en:no11",
    "en:no12",
}


def build_ingredient_processor(
    ingredient_taxonomy: Taxonomy, add_synonym_combinations: bool = True
) -> KeywordProcessor:
    """Create a flashtext KeywordProcessor from an ingredient taxonomy.

    :param ingredient_taxonomy: the ingredient taxonomy
    :param add_synonym_combinations: if True, add all multi-words combinations
        using ingredient synonyms, defaults to True.
        Example: if ingredient 'apple' has 'apples' as a synonym and 'juice'
        has 'juices', 'apples juice', 'apples juices', and 'apple juices' will
        be added as patterns to detect ingredient 'apple juice'.
    :return: a KeywordProcessor
    """
    # dict mapping an normalized ingredient to a set of (node ID, lang) tuples
    name_map = defaultdict(set)
    # it's the reverse structure as name_map, dict mapping a (node ID, lang)
    # to a set of normalized names. Used to generate synonym combinations
    synonyms = defaultdict(set)
    for node in ingredient_taxonomy.iter_nodes():
        if node.id in INGREDIENT_ID_EXCLUDE_LIST:
            # Ignore ingredients that create false positive
            continue
        # dict mapping lang to a set of expressions for a specific ingredient
        seen = defaultdict(set)
        for field in ("names", "synonyms"):
            for lang in ("xx", "en", "fr", "en", "es", "de", "nl", "it"):
                names = getattr(node, field).get(lang)
                if names is None:
                    continue
                if isinstance(names, str):
                    # for 'names' field, the value is a string
                    names = [names]

                for name in names:
                    normalized_name = fold(name.lower())
                    if normalized_name in seen[lang] or len(normalized_name) <= 1:
                        # Don't keep the normalized name if it already exists
                        # or if it's length is <= 1 (to avoid false positives)
                        # For example 'd' is a synonym for 'vitamin d'
                        continue
                    synonyms[(node.id, lang)].add(normalized_name)
                    name_map[normalized_name].add((node.id, lang))
                    seen[lang].add(normalized_name)

    if add_synonym_combinations:
        for normalized_name, keys in list(name_map.items()):
            # get tokens from name by performing a simple whitespace
            # tokenization
            tokens = normalized_name.split(" ")
            if len(tokens) <= 1:
                continue

            for current_node_id, current_lang in keys:
                # combinations is a list of set of string, each element being
                # a token with multiple possible synonyms.
                # we initialize with the original tokens (if no synonym is
                # found for any token, we will just generate the original
                # normalized name again)
                combinations = [{token} for token in tokens]
                for token_idx in range(len(tokens)):
                    token = tokens[token_idx]
                    if token in name_map:
                        # Lookup all ingredient IDs with the same lang that
                        # match the normalized token string
                        for key in (
                            (node_id, lang)
                            for (node_id, lang) in name_map[token]
                            if lang == current_lang
                        ):
                            for synonym in synonyms[key]:
                                combinations[token_idx].add(synonym)

                for combination in itertools.product(*combinations):
                    # generate full ingredient name using one of the combinations
                    name = " ".join(combination)
                    # As name_map values are sets, we're sure there are no
                    # duplicates
                    name_map[name].add((current_node_id, current_lang))

    processor = KeywordProcessor()
    for pattern, keys in name_map.items():
        processor.add_keyword(pattern, list(keys))
    return processor


FORBIDDEN_CHARS = set(string.ascii_lowercase + string.digits)


def extract_ingredient_from_text(
    processor: KeywordProcessor, text: str
) -> List[Tuple[List[Tuple[str, str]], int, int]]:
    """Extract taxonomy ingredients from text.

    :param processor: the flashtext processor created with
        `build_ingredient_processor`
    :param text: the text to extract ingredients from
    :return: a list of (keys, start_idx, end_idx) tuples, where keys is a list
        of (node ID, lang) tuples
    """
    text = fold(text.lower())
    return processor.extract_keywords(text, span_info=True)
