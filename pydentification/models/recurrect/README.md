# Recurrent

Recurrent models package contains a number of standard models using GRU or LSTM cells for dynamical system modelling, 
both simulation and prediction. This package main use is benchmarking.

## Models

* `DynamicalGRULayer` - single GRU layer with skip connection option
* `DynamicalStackedGRU` - stack of N GRUs (with skip connection option) creating prediction with linear or another GRU
