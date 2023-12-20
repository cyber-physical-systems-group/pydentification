import torch


class TimeSeriesLSTM(torch.nn.Module):
    """
    LSTM model for adjusted for time series prediction. It uses stack of LSTM layers and can
    produce output from last layer hidden state or from last time step of another LSTM layer.

    Exact same code as pydentification/models/recurrect/gru.py using LSTM instead of GRU.
    """

    def __init__(
        self,
        n_input_state_variables: int,
        n_output_state_variables: int,
        n_hidden_state_variables: int,
        n_hidden_layers: int = 1,
        dropout: float = 0.0,
        predict_from_hidden_state: bool = False,
    ):
        """
        :param n_input_state_variables: number of input state variables
        :param n_output_state_variables: number of output state variables
        :param n_hidden_state_variables: hidden state size in GRU layers
        :param n_hidden_layers: number of LSTM layers
        :param dropout: dropout probability
        :param predict_from_hidden_state: if True output is produced from hidden state of last layer using linear layer
                                          otherwise output is produced as last time step of another GRU layer with
                                          hidden size equal to number of output state variables
        """
        super(TimeSeriesLSTM, self).__init__()

        self.predict_from_hidden_state = predict_from_hidden_state

        self.lstm = torch.nn.LSTM(
            input_size=n_input_state_variables,
            hidden_size=n_hidden_state_variables,
            num_layers=n_hidden_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.predict_from_hidden_state:
            self.linear = torch.nn.Linear(in_features=n_hidden_state_variables, out_features=n_output_state_variables)
        else:
            self.readout = torch.nn.LSTM(
                input_size=n_hidden_state_variables,
                hidden_size=n_output_state_variables,
                num_layers=1,
                batch_first=True,
                dropout=dropout,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variables, hidden_states = self.lstm(inputs)

        if self.predict_from_hidden_state:
            outputs = self.linear(hidden_states[-1, :, :])  # return outputs from last layer hidden state
            return outputs.unsqueeze(1)  # (batch_size, 1, n_output_state_variables)
        else:
            variables, _ = self.readout(variables)  # return outputs from last time step of readout GRU layer
            return variables[:, -1, :].unsqueeze(1)  # (batch_size, 1, n_output_state_variables)
