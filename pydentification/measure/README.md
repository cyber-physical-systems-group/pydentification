# Measure

This module contains functionality for measuring the "inner-workings" of neural network system identification models. 
For functionality related to testing models' performance, see the `metrics` module. Current submodules are:
* `eigen` - measure eigenvalues and eigen vectors of matrices in the model
* `parameters` - measure parameter norms for matrices and vectors
* `orthogonality` - measure how close to orthogonal are the transformations inside the identification model

# Orthogonality

The measurement, which aims to quantify the orthogonality of the transformations by given matrix. It checks, if the
given matrix is close to being unitary. Given matrix $Q$, the measurement is defined as:

$$ M = || \frac{(I - QQ^T) + (I - Q^TQ)}{2} ||_F $$

*Note*: for complex matrices complex conjugate transpose is used.

The measure $M$ is normalized, to be 1 for unitary matrices (such as identity or rotation matrices) and closer to 0 for matrices, which have the Frobenius large norm (Frobenius norm is used) of the quantity $\frac{(I - QQ^T) + (I - Q^TQ)}{2}$. Normalization is done using:

$$ exp \left ( \frac{-M}{\sqrt{nm}} \right ) $$

where $n$ and $m$ is the size of the matrix.