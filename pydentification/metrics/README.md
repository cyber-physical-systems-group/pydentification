# Metrics

Package containing utilities for regression models, which are missing in `scikit-learn`. It also contains utilities for
evaluating multiple metrics at once. 

## Example
```python
import numpy as np
from pydentification.metrics import regression_report

example_y_pred = np.random.rand(10)
example_y_true = np.random.rand(10)

print(regression_report(y_true=example_y_true, y_pred=example_y_pred, precision=2))
```

```text
                ABS             NORM            
MSE:            0.26            0.99            
RMSE:           0.51            1.94            
MAE:            0.40            1.52            
MAXE:           0.92            3.48            
R2                              -2.75           


                TRUE            PRED            
MEAN:           0.58            0.46            
STD:            0.27            0.24              
```
