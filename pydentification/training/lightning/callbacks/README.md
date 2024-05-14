# Callbacks

Callbacks implementation is split into `functional` code with shared logic, which mutates the `pl.Trainer` object
passed to them (use with caution) and `classes` code, which use the `pl.Callback` interface. 

Code in functional part is verbose by default, getting `name` for logging for each callback. The functions should not
be used outside of `pl.Trainer` context, ideally only in training. 