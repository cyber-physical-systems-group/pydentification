import torch


class DelayLineFeedforward(torch.nn.Module):
    """
    Linear transformation of the input signal using delay line, which means single weight matrix is applied to
    each time step of the input signal. For MIMO systems, different weight matrix is applied for each dimension.

    MIMO is implemented with flattening the input for better performance.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_state_variables: int,
        bias: bool = True,
        skip_connection: bool = False
    ):
        """
        :param n_time_steps: number of input and output time steps (they must be equal for self-attention)
        :param n_state_variables: number of state variables in the system or inner representation in the model
        :param bias: if True bias will be added to the linear module
        :param skip_connection: if True skip connection will be added to the output
        """
        super(DelayLineFeedforward, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables
        self.bias = bias
        self.skip_connection = skip_connection

        self.flatten = torch.nn.Flatten()
        self.feedforward = torch.nn.Linear(
            in_features=self.n_time_steps * self.n_state_variables,
            out_features=self.n_time_steps * self.n_state_variables,
            bias=self.bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variables = self.flatten(inputs)  # flatten time steps
        variables = self.feedforward(variables)
        outputs = torch.reshape(variables, shape=inputs.shape)

        if self.skip_connection:
            outputs = inputs + outputs

        return outputs
