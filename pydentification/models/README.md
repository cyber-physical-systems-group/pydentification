# Models

This package contains building blocks and composite models, the PyTorch conventions are followed and models are 
implemented using `torch.nn.Module` class, so they can be arranged into larger models by compositing and sequentially
stacking.

This package contains following submodules:
* `modules` - building blocks of the models, extending PyTorch modules with custom layers, activation functions, etc.
* `networks` - models composed of building blocks, such as transformer etc.
* `nonparametric` - non-parametric models and supporting utils
