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
                                Absolute                        Normalized                      
Mean Squared Error:             0.20                            1.45                            
Root Mean Squared Error:        0.44                            0.89                            
Mean Absolute Error:            0.36                            1.45                            
Max Error:                      0.89                            3.57                            
R2                                                              -2.19                           


                                True                            Predicted                       
Mean:                           0.61                            0.42                            
std:                            0.25                            0.25    
```
