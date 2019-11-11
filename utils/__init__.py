import json
from typing import Dict, List, Tuple


def update_dict_dot(data: Dict, update: str):
    """Update a dictionary using javascript dot notation style.
    """
    parsed_update = parse_update(update)

    for key, value in parsed_update:
        update_dict_dot_single(data, key, value)


def parse_update(update_query: str) -> List[Tuple[str, object]]:
    raw_updates = [x.strip() for x in update_query.split(';')]
    updates: List[Tuple[str, object]] = []

    for raw_update in raw_updates:
        key, value = raw_update.split('=', maxsplit=1)
        value = json.loads(value)
        updates.append((key, value))

    return updates


def update_dict_dot_single(data: Dict, key: str, value):
    key_splitted: List[str] = key.split('.')

    data_ = data
    for i, sub_key in enumerate(key_splitted):
        if i == len(key_splitted) - 1:
            data_[sub_key] = value
        else:
            data_.setdefault(sub_key, {})
            data_ = data_[sub_key]
