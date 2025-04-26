import numpy as np
from numpy.typing import NDArray


def random_sine_signal(
    t: NDArray,
    dimensions: int,
    n_waves: int = 1,
    frequency_range: tuple[float, float] = (float(0), float(1)),
    phase_range: tuple[float, float] = (float(0), float(1)),
    amplitude_range: tuple[float, float] = (float(0), float(1)),
    noise_magnitude: float = float(0),
    rng: np.random.Generator | None = None,
    seed: int = 0,
):
    """
    Generate a random sine signal with given parameters composed of multiple sine waves.

    :param t: Time array for the signal.
    :param dimensions: Number of dimensions for the signal.
    :param n_waves: Number of sine waves to generate, summed in each dimension.
    :param frequency_range: Range of frequencies for the sine waves.
    :param phase_range: Range of phases for the sine waves.
    :param amplitude_range: Range of amplitudes for the sine waves.
    :param noise_magnitude: Magnitude of noise to add to the signal.
    :param rng: Optional random number generator. If None, a default generator is created.
    :param seed: Seed for the random number generator, default is 0.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_time_steps = len(t)
    frequencies = rng.uniform(frequency_range[0], frequency_range[1], size=dimensions)
    phases = rng.uniform(phase_range[0], phase_range[1], size=dimensions)
    amplitudes = rng.uniform(amplitude_range[0], amplitude_range[1], size=dimensions)

    signal = np.zeros([n_time_steps, dimensions])
    for i in range(n_waves):
        wave = np.zeros(n_time_steps)
        for f, p, a in zip(frequencies, phases, amplitudes):
            wave += a * np.sin(f * t + p)
        if noise_magnitude > 0:  # add noise
            wave += rng.normal(0, noise_magnitude, size=n_time_steps)
        signal[:, i] = wave

    return signal
