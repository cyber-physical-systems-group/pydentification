# Activations

This module contains a set of activation functions, extending default torch activation functions. It contains
primitives, not entire models.

Currently, two activations are implemented:
* `BoundedLinearUnit` - linear activation with dynamically bounded output
* `UniversalActivation` - activation allowing universal approximation with fixed number of neurons [1]

*Note*: UniversalActivation has interesting theoretical properties, but it is not advised to use in practice.  

## References

<a id="1">[1]</a> 
Zuowei Shen and Haizhao Yang and Shijun Zhang (2022). 
*Deep Network Approximation: Achieving Arbitrary Accuracy with Fixed Number of Neurons*
https://www.jmlr.org/papers/volume23/21-1404/21-1404.pdf
