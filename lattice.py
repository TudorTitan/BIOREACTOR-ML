from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# D3Q19 velocity vectors.
# Array shape: (19, 3)
C = jnp.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=jnp.int32,
)


# D3Q19 lattice weights.
W = jnp.asarray(
    [
        1.0 / 3.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ],
    dtype=jnp.float32,
)


# Opposite lattice direction for every D3Q19 population.
OPPOSITE = jnp.asarray(
    [
        0,
        2,
        1,
        4,
        3,
        6,
        5,
        8,
        7,
        10,
        9,
        12,
        11,
        14,
        13,
        16,
        15,
        18,
        17,
    ],
    dtype=jnp.int32,
)


SPEED_OF_SOUND_SQUARED = 1.0 / 3.0


def equilibrium(
    rho: Array,
    velocity: Array,
) -> Array:
    """
    Return the D3Q19 equilibrium distribution.

    Parameters
    ----------
    rho:
        Density field with shape (nx, ny, nz).

    velocity:
        Velocity field with shape (3, nx, ny, nz).

    Returns
    -------
    Array
        Distribution field with shape (19, nx, ny, nz).
    """

    # c_i dot u for every lattice direction.
    cu = jnp.einsum("ia,axyz->ixyz", C, velocity)

    velocity_squared = jnp.sum(
        velocity * velocity,
        axis=0,
    )

    return (
        W[:, None, None, None]
        * rho[None, :, :, :]
        * (
            1.0
            + 3.0 * cu
            + 4.5 * cu * cu
            - 1.5 * velocity_squared[None, :, :, :]
        )
    )