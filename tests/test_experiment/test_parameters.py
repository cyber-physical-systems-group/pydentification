import pytest

from pydentification.experiment.parameters import left_dict_join, prepare_config_for_sweep


@pytest.mark.parametrize(
    ["main", "other", "expected"],
    (
        ({}, {}, {}),
        ({"A": 1, "B": 2}, {"C": 3, "D": 4}, {"A": 1, "B": 2, "C": 3, "D": 4}),
        ({"A": 1}, {"A": 2}, {"A": 1}),
    ),
)
def test_left_dict_join(main: dict, other: dict, expected: dict):
    assert expected == left_dict_join(main, other)


@pytest.mark.parametrize(
    ["config", "parameters", "expected"],
    (
        (
            {"A": 1, "B": 2},
            {"C": [3, 4, 5], "D": [1, 2, 3]},
            {"A": 1, "B": 2, "parameters": {"C": {"values": [3, 4, 5]}, "D": {"values": [1, 2, 3]}}},
        ),
        # case with repeating keys
        (
            {"A": 1, "B": 2},
            {"A": [3, 4, 5], "B": [1, 2, 3]},
            {"A": 1, "B": 2, "parameters": {"A": {"values": [3, 4, 5]}, "B": {"values": [1, 2, 3]}}},
        ),
    ),
)
def test_prepare_config_for_sweep(config: dict, parameters: dict, expected: dict):
    assert expected == prepare_config_for_sweep(config, parameters)
