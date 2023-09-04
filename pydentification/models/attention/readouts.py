import torch


class LinearReadout(torch.nn.Module):
    """
    Linear readout is generally the same as LinearEmbedding layer, but it is used at the end of the model to produce
    desired output shape. It can be used to up or down project state variables and shorten or extend time series length.
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
        super(LinearReadout, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.projection = torch.nn.Linear(
            in_features=self.n_input_time_steps * self.n_input_state_variables,
            out_features=self.n_output_time_steps * self.n_output_state_variables,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        variables = self.flatten(inputs)
        variables = self.projection(variables)
        outputs = torch.reshape(variables, shape=(batch_size, self.n_output_time_steps, self.n_output_state_variables))
        return outputs
