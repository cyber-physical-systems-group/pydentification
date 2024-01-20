# Bounded Training

This package contains training module for neural network to identify nonlinear dynamical systems or static nonlinear
functions with guarantees by using bounded activation incorporating theoretical bounds from the kernel regression model.
The approach is limited to finite memory single-input single-output dynamical systems, which can be converted to
static multiple-input single-output systems by using delay-line. Bounds are computed using kernel regression working
with the same data, but we are able to guarantees of the estimation, which are used to activate a network during and
after training, in order to ensure that the predictions are never outside of those theoretical bounds.

Kernel regression, needs to compute distance from the point where the function is estimated to all the points in the
known dataset (explanatory data), which can lead to extreme memory consumption and slow computation. To mitigate this
issue, we use approximated nearest neighbours algorithm to select memory dynamically from entire known data. Bounds are
computed using those fetched points only, so the nearest neighbour algorithm does not affect the certainty.

## Dependencies

The implementation of hybrid training is based on `pydentification` packages in models, namely: `nonparametric` and
`modules.activations`.