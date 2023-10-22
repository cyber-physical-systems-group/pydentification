import torch


class ConstantLengthEmbedding(torch.nn.Module):
    """
    Module converting time series with shape (batch_size, n_time_steps, n_input_state_variables) into time series
    with shape (batch_size, n_time_steps, n_output_state_variables) using learned linear transformation.

    This can be used as embedding, which preserves the length of the time series. Such embedding is always causal,
    since learned matrix is applied to all time steps idependently.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        bias: bool = True,
    ):
        """
        :param n_time_steps: number of time steps in the input signal and embedding
        :param n_input_state_variables: number of state input variables
        :param n_output_state_variables: number of states to produce
        :param bias: if True bias will be used in linear operation
        """
        super(ConstantLengthEmbedding, self).__init__()

        self.n_input_time_steps = n_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.up_projection = torch.nn.Linear(
            in_features=self.n_state_variables, out_features=self.n_output_state_variables, bias=bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up_projection(inputs)


class ShorteningCausalEmbedding(torch.nn.Module):
    """
    Module converting time series with shape (batch_size, n_input_time_steps, n_input_state_variables) into time series
    with shape (batch_size, n_output_time_steps, n_output_state_variables) using two learned linear transformations.

    The module can be used as embedding for transformer models, which uprojects the state variables and makes the time
    series shorter. The module is always causal.

    It works by firstly applying single linear to all time-steps up-projecting the dimensionality. Later convolution
    layer is applied (with kernel size equal to the ratio of input and output time steps) to shorten the time series.
    This means that each time-steps is mapped using single kernel, due to setting stride equal to kernel size.
    """

    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        bias: bool = True,
    ):
        """
        :param n_input_time_steps: number of time steps in the input signal
        :param n_output_time_steps: number of time steps to produce after linear operation
        :param n_input_state_variables: number of state input variables
        :param n_output_state_variables: number of states to produce
        :param bias: if True bias will be used in linear operation
        """
        if n_input_time_steps % n_output_time_steps != 0:
            raise ValueError("`n_input_time_steps` must be divisible by `n_output_time_steps`!")

        super(ShorteningCausalEmbedding, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.up_projection = torch.nn.Linear(
            in_features=self.n_input_state_variables, out_features=self.n_output_state_variables, bias=bias
        )

        self.time_projection = torch.nn.Conv1d(
            in_channels=self.n_output_state_variables,
            out_channels=self.n_output_state_variables,
            kernel_size=self.n_input_time_steps // self.n_output_time_steps,
            stride=self.n_input_time_steps // self.n_output_time_steps,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.up_projection(inputs)
        embeddings = torch.permute(embeddings, dims=(0, 2, 1))  # permute to channels first before convolution
        embeddings = self.time_projection(embeddings)
        embeddings = torch.permute(embeddings, dims=(0, 2, 1))  # permute back to channels last

        return embeddings
