# Dynamical Self Attention

This module implements self-attention model for dynamical system. It is really similar to transformer (encoder or
decoder), but it does not share feedforward network weights between layers. In classical transformer, the inputs have
shape of (batch_size, seq_len, input_dim), which is also true for this module, however the dimension of input_dim is
usually much lower and sequence length can be higher. This is due to the fact, that real-world dynamical system
typically have fewer dimensions than what is expected in transformer embeddings.

This package implements 5 torch modules (one of them is a block of submodules) and 1 lightning class for training,
those torch modules can be combined with other modules, such as activations or various learned layers. The modules are:
* `LinearProjection` - used for converting the signal to embedding and converting produced embedding to prediction
* `DynamicalSelfAttention` - the module computes self-attention for dynamical systems
* `DelayLineFeedforward` - learned linear transformation of the input signal using delay line
* `TransformerFeedforward` - the module contains feedforward network in transformer style for dynamical systems
* `DynamicalSelfAttentionBlock` - module contains DynamicalSelfAttention, DelayLineFeedforward and any activation

Additionally `LightningTrainingModule`, which can be used for training the model with pytorch-lightning framework.

### Linear Projection

Module converting time series with shape (batch_size, n_input_time_steps, n_input_state_variables) into time series
with shape (batch_size, n_output_time_steps, n_output_state_variables) using learned linear transformation.

### Feedforward

Feedforward network is a linear layer, which is applied using delay-line of inputs. This means that single weight
processes single time-step for each dimension of the system. For a system with 3 dimensions and 10 steps the feedforward
network would have 30 weights (+ optionally bias parameters). There are two variants, plain delay line without
activation or transformer-like two layered feedforward with activation in between.  

### Self-Attention

The same as regular self-attention in transformers. The operations are the same, but the module keeps shapes aligned
with tensor shapes expected in modelling a dynamical system, which is (batch, sequence, state_dim).

### Example

```python
import torch

from pydentification.models.attention import (
    DelayLineFeedforward,
    DynamicalSelfAttention,
    LinearProjection,
    LinearProjection,
)

model = torch.nn.Sequential(
    # make sequence shorter in first layer
    LinearProjection(n_input_time_steps=64, n_input_state_variables=1, n_output_time_steps=16, n_output_state_variables=1),
    # add three layers of self-attention block with delay-line feedforward and GELU 
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    # final layer producing output time-step
    LinearProjection(n_input_time_steps=16, n_output_time_steps=1, n_input_state_variables=1, n_output_state_variables=1, bias=True),
)
```

### Training

The model can be trained with lightning framework in the straight forward way. Using `model` (created as in the example
above), `LightningTrainingModule` needs to be created and following code can be used to run exemplary training:

```python
import torch
import lightning.pytorch as pl


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # assume model already exists
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

early_stopping = pl.callbacks.EarlyStopping(monitor="validation/loss", patience=10, mode="min", verbose=True)
checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f"models", monitor="validation/loss", every_n_epochs=1)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    default_root_dir=f"models",
    callbacks=[early_stopping, checkpoint],
)

# after running this code, the model will be trained
trainer.fit(model, train_loader, validation_loader)  # assume train_loader and validation_loader already exist
# prediction can be done with model.forward(x) or using trainer.predict(model, test_loader)
y_hat = trainer.predict(model, test_loader)  # assume test_loader already exists
y_pred = torch.cat(y_hat).numpy()
```

**Warning**: when training on multiple GPUs the predict function will not work as expected, this can be solved with
separate trainer object just for prediction (can be on CPU) or implementing PredictionWriterCallback in lightning
