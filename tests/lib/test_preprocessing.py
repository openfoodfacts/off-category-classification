import itertools

import pytest

from lib.preprocessing import build_ingredient_processor, extract_ingredient_from_text
from lib.taxonomy import get_taxonomy


@pytest.fixture(scope="session")
def ingredient_processor_with_synonyms():
    ingredient_taxonomy = get_taxonomy("ingredient", offline=True)
    yield build_ingredient_processor(ingredient_taxonomy, True)


@pytest.fixture(scope="session")
def ingredient_processor_without_synonyms():
    ingredient_taxonomy = get_taxonomy("ingredient", offline=True)
    yield build_ingredient_processor(ingredient_taxonomy, False)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("JUS de pomme, ", [([("en:apple-juice", "fr")], 0, 12)]),
        # Only "JUS" matches, we check here that word boundaries are well
        # detected
        ("JUS de pommeata", [([("en:juice", "fr")], 0, 3)]),
        ("JUS de pommeata", [([("en:juice", "fr")], 0, 3)]),
        (
            "Beurre à 82% MG doux au naturel",
            [([("en:82-fat-unsalted-butter", "fr")], 0, 20)],
        ),
        # "fr:pulpe de pommes" doesn't exist in the taxonomy, only "fr:pulpe de pomme"
        # we check here that adding synonym combinations work correctly
        (
            "PULPE DE POMMES",
            [([("en:apple-pulp", "fr")], 0, 15)],
        ),
        # Long ingredient list to check for regression
        (
            "Farine complète de BLÉ 38%, eau, farine de BLÉ 25%, huile de colza, sucre, "
            "arôme (contient alcool), sel, vinaigre, farine de SEIGLE maltée, levure, gluten "
            "de BLÉ, extrait d'acérola. Peut contenir des traces d'OEUFS, SOJA, LAIT, FRUITS À "
            "COQUE, GRAINES DE SÉSAME.",
            [
                ([("en:whole-wheat-flour", "fr")], 0, 22),
                ([("en:water", "fr")], 28, 31),
                ([("en:wheat-flour", "fr")], 33, 46),
                ([("en:colza-oil", "fr")], 52, 66),
                ([("en:sugar", "fr")], 68, 73),
                ([("en:flavouring", "fr")], 75, 80),
                ([("en:alcohol", "fr"), ("en:alcohol", "it")], 91, 97),
                ([("en:salt", "fr")], 100, 103),
                ([("en:vinegar", "fr")], 105, 113),
                ([("en:rye-malt-flour", "fr")], 115, 138),
                ([("en:yeast", "fr")], 140, 146),
                ([("en:wheat-gluten", "fr")], 148, 161),
                ([("en:acerola-juice", "fr")], 163, 180),
                ([("en:egg", "fr")], 209, 214),
                (
                    [
                        ("en:soya", "de"),
                        ("en:soya", "en"),
                        ("en:soya", "fr"),
                        ("en:soya", "nl"),
                        ("en:soya", "es"),
                    ],
                    216,
                    220,
                ),
                ([("en:milk", "fr")], 222, 226),
                ([("en:nut", "fr")], 228, 242),
                ([("en:sesame-seeds", "fr")], 244, 261),
            ],
        ),
    ],
)
def test_extract_ingredient_from_text(
    text: str,
    expected: list,
    ingredient_processor_with_synonyms,
):
    extracted = extract_ingredient_from_text(ingredient_processor_with_synonyms, text)
    for item, _, __ in itertools.chain(extracted, expected):
        item.sort()
    assert extracted == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        # "fr:pulpe de pommes" doesn't exist in the taxonomy, only "fr:pulpe de pomme"
        # so we shouldn't get a full match (only "pommes" should match)
        (
            "PULPE DE POMMES",
            [([("en:apple", "fr")], 9, 15)],
        ),
    ],
)
def test_extract_ingredient_from_text_without_synonyms(
    text: str,
    expected: list,
    ingredient_processor_without_synonyms,
):
    extracted = extract_ingredient_from_text(
        ingredient_processor_without_synonyms, text
    )
    for item, _, __ in itertools.chain(extracted, expected):
        item.sort()
    assert extracted == expected
