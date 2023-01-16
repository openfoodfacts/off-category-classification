from typing import Dict, List, Optional, Set

from lib.taxonomy import Taxonomy

from .text_utils import get_tag
from .constant import EXCLUDE_LIST_CATEGORIES


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

    if value < 0 or (nutriment_name != "energy-kcal" and value >= 101):
        # Remove invalid values
        return -2

    return value
