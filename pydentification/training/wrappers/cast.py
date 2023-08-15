import torch


class CastWrapper(torch.nn.Module):
    """Wrapper for torch module that casts input and output to specified dtypes"""

    def __init__(self, module: torch.nn.Module, in_dtype: torch.dtype, out_dtype: torch.dtype):
        """
        :param module: wrapped torch module
        :param in_dtype: dtype of input to the module
        :param out_dtype: dtype of output from the module
        """
        super(CastWrapper, self).__init__()
        self.module = module

        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        return self.module(inputs.to(self.in_dtype), *args, **kwargs).to(self.out_dtype)
