import torch
from torch import Tensor
from torch.nn import Module


class ResidualConnectionWrapper(Module):
    """Wrapper for torch module that adds residual connection to the input"""

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.module(inputs)


class AutoregressiveResidualWrapper(Module):
    """
    Autoregressive residual wrapper for torch module, which can be used for prediction modelling.

    1 step ahead model (given in __init__) is called in the loop (model only predicts residuals) and predictions are
    auto-regressively used as following inputs for N steps, where N is given dynamically as parameter of forward method.
    """

    def __init__(self, step_ahead_model: Module):
        super().__init__()

        self.step_ahead_model = step_ahead_model

    def forward(self, inputs: Tensor, n_predict_time_steps: int = 1) -> Tensor:
        inputs_and_predictions = inputs  # copy inputs
        n_model_time_steps = inputs.shape[1]

        for step in range(n_predict_time_steps):
            step_inputs = inputs[:, -n_model_time_steps:, :]
            # predict next step with residual connection
            step_ahead_prediction = inputs[:, -1, :] + self.step_ahead_model(step_inputs)
            # append prediction to inputs and previous inputs
            inputs_and_predictions = torch.cat([inputs, step_ahead_prediction], dim=1)

        return inputs_and_predictions[:, -n_predict_time_steps:, :]  # return only predictions
