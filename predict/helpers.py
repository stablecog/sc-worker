from typing import TypeVar, List

T = TypeVar("T")


def true_if_value_in_list(value: T, list_of_values: List[T]) -> bool:
    if value not in list_of_values:
        raise ValueError(f'"{value}" is not in the list of choices')
    return True
