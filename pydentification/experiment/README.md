# Experiment

This package contains utils for running experiments with W&B, including single runs, sweeps etc.
The code here can be used only with W&B, but this is not required to use other packages

## Reporters

Reporters are standalone functions used to log commonly needed experiment properties to W&B, including plots.
To use then run them in the experiment code:

* `report_prediction_plot` - adds interactive plotly graphic to W&B 
* `report_metrics` - adds numeric value for each regression metrics
* `report_trainable_parameters` - adds number of trainable parameters of the model

### Example

To use reporters run following example code:

```python
y_hat = trainer.predict(model, test_loader)  # assume trainer and model are trained 
y_pred = torch.cat(y_hat).numpy()
y_true = torch.cat([y for _, y in test_loader]).numpy()

metrics = regression_metrics(y_pred=y_pred.flatten(), y_true=y_true.flatten())  # function from pydentification.metrics

# run reporters
report_metrics(metrics)
report_trainable_parameters(model)
report_prediction_plot(predictions=y_pred, targets=y_true)
```
