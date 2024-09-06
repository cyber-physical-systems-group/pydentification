import pytest
import torch
from torch.nn import Module

from pydentification.training.measure import register


@pytest.mark.parametrize(
    ["model", "register_fn", "expected_names"],
    [
        # Test cases for register_all_parameters

        # Linear layer has two parameters (weight, bias) -> register both
        [torch.nn.Linear(10, 10), register.register_all_parameters, ["weight", "bias"]],
        # Conv2d layer has four parameters (weight, bias) -> register all
        [torch.nn.Conv2d(3, 3, 3), register.register_all_parameters, ["weight", "bias"]],
        # Sequential model has two Linear layers -> register all parameters
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10)),
            register.register_all_parameters,
            ["0.weight", "0.bias", "1.weight", "1.bias"],
        ],
        # Sequential model with Conv2d and activation -> register all parameters
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU()),
            register.register_all_parameters,
            ["0.weight", "0.bias"],
        ],
        # RNN layer has four parameters (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0) -> register all
        [
            torch.nn.RNN(10, 10, 1),
            register.register_all_parameters,
            ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"],
        ],
        # LSTM layer has eight parameters (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l0_reverse,
        # weight_hh_l0_reverse, bias_ih_l0_reverse, bias_hh_l0_reverse) -> register all
        [
            torch.nn.LSTM(10, 10, 1),
            register.register_all_parameters,
            ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"],
        ],
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.BatchNorm1d(10), torch.nn.ReLU()),
            register.register_all_parameters,
            ["0.weight", "0.bias", "1.weight", "1.bias"],
        ],
        [
            torch.nn.Transformer(nhead=2, num_encoder_layers=2, num_decoder_layers=2),
            register.register_all_parameters,
            [
                # in proj weight and bias are added as `torch.nn.Parameter` in `MultiheadAttention`
                # `iter_modules_and_parameters` returns them as separate parameters of self_attn and multihead_attn
                "encoder.layers.0.self_attn.in_proj_weight",
                "encoder.layers.0.self_attn.in_proj_bias",
                # out proj weight and bias are added as submodule in `MultiheadAttention`, instance of
                # `torch.nn.NonDynamicallyQuantizableLinear`, so `iter_modules_and_parameters`
                # returns them as subparameters of the modules, hence `.` in the name instead of `_`.
                "encoder.layers.0.self_attn.out_proj.weight",
                "encoder.layers.0.self_attn.out_proj.bias",
                "encoder.layers.0.linear1.weight",
                "encoder.layers.0.linear1.bias",
                "encoder.layers.0.linear2.weight",
                "encoder.layers.0.linear2.bias",
                "encoder.layers.0.norm1.weight",
                "encoder.layers.0.norm1.bias",
                "encoder.layers.0.norm2.weight",
                "encoder.layers.0.norm2.bias",
                "encoder.layers.1.self_attn.in_proj_weight",
                "encoder.layers.1.self_attn.in_proj_bias",
                "encoder.layers.1.self_attn.out_proj.weight",
                "encoder.layers.1.self_attn.out_proj.bias",
                "encoder.layers.1.linear1.weight",
                "encoder.layers.1.linear1.bias",
                "encoder.layers.1.linear2.weight",
                "encoder.layers.1.linear2.bias",
                "encoder.layers.1.norm1.weight",
                "encoder.layers.1.norm1.bias",
                "encoder.layers.1.norm2.weight",
                "encoder.layers.1.norm2.bias",
                "encoder.norm.weight",
                "encoder.norm.bias",
                "decoder.layers.0.self_attn.in_proj_weight",
                "decoder.layers.0.self_attn.in_proj_bias",
                "decoder.layers.0.self_attn.out_proj.weight",
                "decoder.layers.0.self_attn.out_proj.bias",
                "decoder.layers.0.multihead_attn.in_proj_weight",
                "decoder.layers.0.multihead_attn.in_proj_bias",
                "decoder.layers.0.multihead_attn.out_proj.weight",
                "decoder.layers.0.multihead_attn.out_proj.bias",
                "decoder.layers.0.linear1.weight",
                "decoder.layers.0.linear1.bias",
                "decoder.layers.0.linear2.weight",
                "decoder.layers.0.linear2.bias",
                "decoder.layers.0.norm1.weight",
                "decoder.layers.0.norm1.bias",
                "decoder.layers.0.norm2.weight",
                "decoder.layers.0.norm2.bias",
                "decoder.layers.0.norm3.weight",
                "decoder.layers.0.norm3.bias",
                "decoder.layers.1.self_attn.in_proj_weight",
                "decoder.layers.1.self_attn.in_proj_bias",
                "decoder.layers.1.self_attn.out_proj.weight",
                "decoder.layers.1.self_attn.out_proj.bias",
                "decoder.layers.1.multihead_attn.in_proj_weight",
                "decoder.layers.1.multihead_attn.in_proj_bias",
                "decoder.layers.1.multihead_attn.out_proj.weight",
                "decoder.layers.1.multihead_attn.out_proj.bias",
                "decoder.layers.1.linear1.weight",
                "decoder.layers.1.linear1.bias",
                "decoder.layers.1.linear2.weight",
                "decoder.layers.1.linear2.bias",
                "decoder.layers.1.norm1.weight",
                "decoder.layers.1.norm1.bias",
                "decoder.layers.1.norm2.weight",
                "decoder.layers.1.norm2.bias",
                "decoder.layers.1.norm3.weight",
                "decoder.layers.1.norm3.bias",
                "decoder.norm.weight",
                "decoder.norm.bias",
            ],
        ],
        # Test cases for register_matrix_parameters
        # Linear layer has one matrix (weight) and one vector (bias) parameter -> register only weight
        [torch.nn.Linear(10, 10), register.register_matrix_parameters, ["weight"]],
        # Conv2d layer parameters has shape (out_channels, in_channels, kernel_size, kernel_size) -> 4D is not matrix
        [torch.nn.Conv2d(3, 3, 3), register.register_matrix_parameters, []],  # noqa
        # Sequential model has two Linear layers -> register both weights
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10)),
            register.register_matrix_parameters,
            ["0.weight", "1.weight"],
        ],
        # Sequential model with Conv2d and activation -> register only Conv2d weight
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU()),
            register.register_matrix_parameters,
            ["0.weight"],
        ],
        # RNN layer has three parameters (weight_ih_l0, weight_hh_l0, bias_ih_l0) -> register only weights
        [torch.nn.RNN(10, 10, 1), register.register_matrix_parameters, ["weight_ih_l0", "weight_hh_l0"]],
        # LSTM layer has six parameters, but it is implemented to store weights for each gate in one matrix
        # only weight_ih_l0 and weight_hh_l0 are registered, for details see: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html  # noqa
        [torch.nn.LSTM(10, 10, 1), register.register_matrix_parameters, ["weight_ih_l0", "weight_hh_l0"]],
        # Sequential composed of Linear, batch-norm and activation -> register only Linear weight
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.BatchNorm1d(10), torch.nn.ReLU()),
            register.register_matrix_parameters,
            ["0.weight"],
        ],
        # Small 2-layer encoder-decoder transformer will have 16 matrix parameters
        [
            torch.nn.Transformer(nhead=2, num_encoder_layers=2, num_decoder_layers=2),
            register.register_matrix_parameters,
            [
                "encoder.layers.0.self_attn.in_proj_weight",
                "encoder.layers.0.self_attn.out_proj.weight",
                "encoder.layers.0.linear1.weight",
                "encoder.layers.0.linear2.weight",
                "encoder.layers.1.self_attn.in_proj_weight",
                "encoder.layers.1.self_attn.out_proj.weight",
                "encoder.layers.1.linear1.weight",
                "encoder.layers.1.linear2.weight",
                "decoder.layers.0.self_attn.in_proj_weight",
                "decoder.layers.0.self_attn.out_proj.weight",
                "decoder.layers.0.multihead_attn.in_proj_weight",
                "decoder.layers.0.multihead_attn.out_proj.weight",
                "decoder.layers.0.linear1.weight",
                "decoder.layers.0.linear2.weight",
                "decoder.layers.1.self_attn.in_proj_weight",
                "decoder.layers.1.self_attn.out_proj.weight",
                "decoder.layers.1.multihead_attn.in_proj_weight",
                "decoder.layers.1.multihead_attn.out_proj.weight",
                "decoder.layers.1.linear1.weight",
                "decoder.layers.1.linear2.weight",
            ],
        ],
        # Test cases for register_square_parameters

        # Linear layer has one matrix (weight) and one vector (bias) parameter -> register only weight
        [torch.nn.Linear(10, 10), register.register_square_parameters, ["weight"]],
        # Conv2d layer parameters has shape (out_channels, in_channels, kernel_size, kernel_size) -> 4D is not matrix
        [torch.nn.Conv2d(3, 3, 3), register.register_square_parameters, []],  # noqa
        # Sequential model has two Linear layers -> register both weights
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10)),
            register.register_square_parameters,
            ["0.weight", "1.weight"],
        ],
        # Sequential model with Conv2d and activation -> register only Conv2d weight
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU()),
            register.register_square_parameters,
            ["0.weight"],
        ],
        # RNN layer has three parameters (weight_ih_l0, weight_hh_l0, bias_ih_l0) -> register only weights
        [torch.nn.RNN(10, 10, 1), register.register_square_parameters, ["weight_ih_l0", "weight_hh_l0"]],
        # Parameters of LSTM layer are stored in one matrix, but it is not square matrix -> return empty list
        [torch.nn.LSTM(10, 10, 1), register.register_square_parameters, []],
        # Sequential composed of Linear, batch-norm and activation -> register only Linear weight
        [
            torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.BatchNorm1d(10), torch.nn.ReLU()),
            register.register_square_parameters,
            ["0.weight"],
        ],
        # Small 2-layer encoder-decoder transformer will have 6 square matrix parameters -> register all
        # weights for Q,K and V transformation are stored in one parameter (`in_proj_weight`), but it is not square
        # feed-forward modules have non-square up and down projections, which are not square matrices
        # all other parameters are not square matrices or vector parameters
        [
            torch.nn.Transformer(nhead=2, num_encoder_layers=2, num_decoder_layers=2),
            register.register_square_parameters,
            [
                "encoder.layers.0.self_attn.out_proj.weight",
                "encoder.layers.1.self_attn.out_proj.weight",
                "decoder.layers.0.self_attn.out_proj.weight",
                "decoder.layers.0.multihead_attn.out_proj.weight",
                "decoder.layers.1.self_attn.out_proj.weight",
                "decoder.layers.1.multihead_attn.out_proj.weight",
            ],
        ],
    ],

)
def test_register_matrix_parameters(model: Module, register_fn: register.RegisterCallable, expected_names: list[str]):
    names = [name for name, _ in register_fn(model)]
    assert expected_names == names
