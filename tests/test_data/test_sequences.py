import numpy as np
import pytest

from pydentification.data import generate_time_series_windows


@pytest.mark.parametrize(
    "values, shift, expected_features, expected_targets",
    [
        (np.arange(5), 1, np.array([[[0]], [[1]], [[2]], [[3]]]), np.array([[[1]], [[2]], [[3]], [[4]]])),
        (np.arange(5), 1, np.array([[[0]], [[1]], [[2]], [[3]]]), np.array([[[1]], [[2]], [[3]], [[4]]])),
        (
            np.arange(5),
            1,
            np.array([[[0]], [[1]], [[2]], [[3]]]),
            np.array([[[1]], [[2]], [[3]], [[4]]]),
        ),
        (
            np.arange(5),
            1,
            np.array([[[0]], [[1]], [[2]], [[3]]]),
            np.array([[[1]], [[2]], [[3]], [[4]]]),
        ),
        (np.arange(5), 2, np.array([[[0]], [[2]]]), np.array([[[1]], [[3]]])),
        (np.arange(5), 2, np.array([[[0]], [[2]]]), np.array([[[1]], [[3]]])),
        (np.arange(10), 5, np.array([[[0]], [[5]]]), np.array([[[1]], [[6]]])),
        (np.arange(10), 5, np.array([[[0]], [[5]]]), np.array([[[1]], [[6]]])),
    ],
)
def test_predictive_split_with_unit_window(
    values: np.array,
    shift: int,
    expected_features: np.array,
    expected_targets: np.array,
):
    """Tests split function when configured for predictive modelling with unit window"""
    values = np.expand_dims(values, axis=-1)
    results = generate_time_series_windows(
        outputs=values, forward_output_window_size=1, backward_output_window_size=1, shift=shift
    )

    np.testing.assert_array_equal(results["backward_outputs"], expected_features)
    np.testing.assert_array_equal(results["forward_outputs"], expected_targets)


@pytest.mark.parametrize(
    "values, shift, expected_features, expected_targets",
    [
        (
            np.arange(12),
            3,
            np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]),
            np.array([[[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]]),
        ),
        (
            np.arange(12),
            3,
            np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]),
            np.array([[[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]]),
        ),
        (
            np.arange(12),
            1,
            np.array(
                [
                    [[0], [1], [2]],
                    [[1], [2], [3]],
                    [[2], [3], [4]],
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                ]
            ),
            np.array(
                [
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                    [[7], [8], [9]],
                    [[8], [9], [10]],
                    [[9], [10], [11]],
                ]
            ),
        ),
        (
            np.arange(12),
            3,
            np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]),
            np.array([[[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]]),
        ),
        (
            np.arange(12),
            1,
            np.array(
                [
                    [[0], [1], [2]],
                    [[1], [2], [3]],
                    [[2], [3], [4]],
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                ]
            ),
            np.array(
                [
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                    [[7], [8], [9]],
                    [[8], [9], [10]],
                    [[9], [10], [11]],
                ]
            ),
        ),
    ],
)
def test_predictive_split_with_long_window(
    values: np.array,
    shift: int,
    expected_features: np.array,
    expected_targets: np.array,
):
    """Tests split function when configured for predictive modelling with unit window"""
    values = np.expand_dims(values, axis=-1)
    results = generate_time_series_windows(
        outputs=values, forward_output_window_size=3, backward_output_window_size=3, shift=shift
    )

    np.testing.assert_array_equal(results["backward_outputs"], expected_features)
    np.testing.assert_array_equal(results["forward_outputs"], expected_targets)


@pytest.mark.parametrize(
    "values, shift, backward_window_size, forward_window_size, expected_features, expected_targets",
    [
        (
            np.arange(12),
            2,  # shift
            3,  # backward_window_size
            2,  # forward_window_size
            np.array([[[0], [1], [2]], [[2], [3], [4]], [[4], [5], [6]], [[6], [7], [8]]]),
            np.array([[[3], [4]], [[5], [6]], [[7], [8]], [[9], [10]]]),
        ),
        (
            np.arange(12),
            2,  # shift
            3,  # backward_window_size
            2,  # forward_window_size
            np.array([[[0], [1], [2]], [[2], [3], [4]], [[4], [5], [6]], [[6], [7], [8]]]),
            np.array([[[3], [4]], [[5], [6]], [[7], [8]], [[9], [10]]]),
        ),
        (
            np.arange(12),
            3,  # shift
            2,  # backward_window_size
            3,  # forward_window_size
            np.array([[[0], [1]], [[3], [4]], [[6], [7]]]),
            np.array([[[2], [3], [4]], [[5], [6], [7]], [[8], [9], [10]]]),
        ),
    ],
)
def test_predictive_split_with_uneven_window(
    values: np.array,
    shift: int,
    backward_window_size: int,
    forward_window_size: int,
    expected_features: np.array,
    expected_targets: np.array,
):
    """
    Tests split function when configured for predictive modelling with uneven forward and backward window sizes
    For this use-case shift should be equal to forward_window_size so model does not predict the same datapoint twice
    """
    values = np.expand_dims(values, axis=-1)
    results = generate_time_series_windows(
        outputs=values,
        forward_output_window_size=forward_window_size,
        backward_output_window_size=backward_window_size,
        shift=shift,
    )

    np.testing.assert_array_equal(results["backward_outputs"], expected_features)
    np.testing.assert_array_equal(results["forward_outputs"], expected_targets)


@pytest.mark.parametrize(
    "values, shift, input_window_size, output_window_size, output_mask, expected_features, expected_targets",
    [
        (
            np.arange(10),
            1,  # shift
            1,  # input_window_size
            1,  # output_window_size
            0,  # forward_output_mask
            np.array([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]]]),
            np.array([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]]]),
        ),
        (
            np.arange(10),
            3,  # shift
            1,  # input_window_size
            1,  # output_window_size
            0,  # forward_output_mask
            np.array([[[0]], [[3]], [[6]], [[9]]]),
            np.array([[[0]], [[3]], [[6]], [[9]]]),
        ),
        (
            np.arange(10),
            5,  # shift
            5,  # input_window_size
            5,  # output_window_size
            3,  # forward_output_mask
            np.array([[[0], [1], [2], [3], [4]], [[5], [6], [7], [8], [9]]]),
            np.array([[[3], [4]], [[8], [9]]]),
        ),
        (
            np.arange(10),
            2,  # shift
            5,  # input_window_size
            5,  # output_window_size
            3,  # forward_output_mask
            np.array([[[0], [1], [2], [3], [4]], [[2], [3], [4], [5], [6]], [[4], [5], [6], [7], [8]]]),
            np.array([[[3], [4]], [[5], [6]], [[7], [8]]]),
        ),
        (
            np.arange(10),
            1,  # shift
            3,  # input_window_size
            3,  # output_window_size
            0,  # forward_output_mask
            np.array(
                [
                    [[0], [1], [2]],
                    [[1], [2], [3]],
                    [[2], [3], [4]],
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                    [[7], [8], [9]],
                ]
            ),
            np.array(
                [
                    [[0], [1], [2]],
                    [[1], [2], [3]],
                    [[2], [3], [4]],
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                    [[7], [8], [9]],
                ]
            ),
        ),
        (
            np.arange(10),
            3,  # shift
            3,  # input_window_size
            3,  # output_window_size
            0,  # forward_output_mask
            np.array(
                [
                    [[0], [1], [2]],
                    [[3], [4], [5]],
                    [[6], [7], [8]],
                ]
            ),
            np.array(
                [
                    [[0], [1], [2]],
                    [[3], [4], [5]],
                    [[6], [7], [8]],
                ]
            ),
        ),
        (
            np.arange(10),
            1,  # shift
            3,  # input_window_size
            3,  # output_window_size
            2,  # forward_output_mask
            np.array(
                [
                    [[0], [1], [2]],
                    [[1], [2], [3]],
                    [[2], [3], [4]],
                    [[3], [4], [5]],
                    [[4], [5], [6]],
                    [[5], [6], [7]],
                    [[6], [7], [8]],
                    [[7], [8], [9]],
                ]
            ),
            np.array(
                [
                    [[2]],
                    [[3]],
                    [[4]],
                    [[5]],
                    [[6]],
                    [[7]],
                    [[8]],
                    [[9]],
                ]
            ),
        ),
        (
            np.arange(10),
            3,  # shift
            3,  # input_window_size
            3,  # output_window_size
            2,  # forward_output_mask
            np.array(
                [
                    [[0], [1], [2]],
                    [[3], [4], [5]],
                    [[6], [7], [8]],
                ]
            ),
            np.array(
                [
                    [[2]],
                    [[5]],
                    [[8]],
                ]
            ),
        ),
        (
            np.arange(10),
            3,  # shift
            3,  # input_window_size
            3,  # output_window_size
            1,  # forward_output_mask
            np.array(
                [
                    [[0], [1], [2]],
                    [[3], [4], [5]],
                    [[6], [7], [8]],
                ]
            ),
            np.array(
                [
                    [[1], [2]],
                    [[4], [5]],
                    [[7], [8]],
                ]
            ),
        ),
    ],
)
def test_simulation_split_with_output_masking(
    values: np.array,
    shift: int,
    input_window_size: int,
    output_window_size: int,
    output_mask: int,
    expected_features: np.array,
    expected_targets: np.array,
):
    """Tests simulation modelling use-case including output masking"""
    values = np.expand_dims(values, axis=-1)
    results = generate_time_series_windows(
        inputs=values,
        outputs=values,
        forward_input_window_size=input_window_size,
        forward_output_window_size=output_window_size,
        forward_output_mask=output_mask,
        shift=shift,
    )

    np.testing.assert_array_equal(results["forward_inputs"], expected_features)
    np.testing.assert_array_equal(results["forward_outputs"], expected_targets)


@pytest.mark.parametrize(
    "inputs, outputs, shift, forward_input_window_size, backward_input_window_size, forward_output_window_size, backward_output_window_size, expected_results",  # noqa
    [
        (
            -1 * np.arange(12),  # inputs
            np.arange(12),  # outputs
            3,  # shift
            3,  # forward_input_window_size
            0,  # backward_input_window_size
            3,  # forward_output_window_size
            3,  # backward_output_window_size
            {
                "backward_inputs": np.array([]),
                "backward_outputs": np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]),  # aux inputs
                "forward_inputs": np.array([[[-3], [-4], [-5]], [[-6], [-7], [-8]], [[-9], [-10], [-11]]]),  # inputs
                "forward_outputs": np.array([[[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]]),  # targets
            },
        ),
        (
            -1 * np.arange(8),  # inputs
            np.arange(8),  # outputs
            1,  # shift
            3,  # forward_input_window_size
            0,  # backward_input_window_size
            3,  # forward_output_window_size
            3,  # backward_output_window_size
            {
                "backward_inputs": np.array([]),
                "backward_outputs": np.array([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]]),  # aux inputs
                "forward_inputs": np.array([[[-3], [-4], [-5]], [[-4], [-5], [-6]], [[-5], [-6], [-7]]]),  # inputs
                "forward_outputs": np.array([[[3], [4], [5]], [[4], [5], [6]], [[5], [6], [7]]]),  # targets
            },
        ),
        (
            -1 * np.arange(12),  # inputs
            np.arange(12),  # outputs
            3,  # shift
            3,  # forward_input_window_size
            3,  # backward_input_window_size
            3,  # forward_output_window_size
            3,  # backward_output_window_size
            {
                "backward_inputs": np.array([[[0], [-1], [-2]], [[-3], [-4], [-5]], [[-6], [-7], [-8]]]),  # aux inputs
                "backward_outputs": np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]),  # aux inputs
                "forward_inputs": np.array([[[-3], [-4], [-5]], [[-6], [-7], [-8]], [[-9], [-10], [-11]]]),  # inputs
                "forward_outputs": np.array([[[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]]),  # targets
            },
        ),
    ],
)
def test_simulation_split_with_backward_states_as_aux_input(
    inputs: np.array,
    outputs: np.array,
    shift: int,
    forward_input_window_size: int,
    backward_input_window_size: int,
    forward_output_window_size: int,
    backward_output_window_size: int,
    expected_results: dict,
):
    """Tests simulation modelling use-case with auxiliary model input of backward system outputs"""
    inputs = np.expand_dims(inputs, axis=-1)
    outputs = np.expand_dims(outputs, axis=-1)

    results = generate_time_series_windows(
        inputs=inputs,
        outputs=outputs,
        forward_input_window_size=forward_input_window_size,
        backward_input_window_size=backward_input_window_size,
        forward_output_window_size=forward_output_window_size,
        backward_output_window_size=backward_output_window_size,
        shift=shift,
    )

    np.testing.assert_array_equal(results["backward_inputs"], expected_results["backward_inputs"])
    np.testing.assert_array_equal(results["backward_outputs"], expected_results["backward_outputs"])
    np.testing.assert_array_equal(results["forward_inputs"], expected_results["forward_inputs"])
    np.testing.assert_array_equal(results["forward_outputs"], expected_results["forward_outputs"])
