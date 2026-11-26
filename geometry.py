from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    length_x: float
    length_y: float
    length_z: float

    def coordinates(self) -> tuple[Array, Array, Array]:
        x = jnp.linspace(0.0, self.length_x, self.nx)
        y = jnp.linspace(0.0, self.length_y, self.ny)
        z = jnp.linspace(0.0, self.length_z, self.nz)

        return jnp.meshgrid(x, y, z, indexing="ij")


def gyroid_field(
    x: Array,
    y: Array,
    z: Array,
    *,
    period: float,
    level: float = 0.0,
) -> Array:
    """Return the implicit gyroid level-set field."""

    k = 2.0 * jnp.pi / period

    return (
        jnp.sin(k * x) * jnp.cos(k * y)
        + jnp.sin(k * y) * jnp.cos(k * z)
        + jnp.sin(k * z) * jnp.cos(k * x)
        - level
    )


def gyroid_solid_mask(
    x: Array,
    y: Array,
    z: Array,
    *,
    period: float,
    level: float = 0.0,
    sheet_half_thickness: float = 0.15,
) -> Array:
    """
    Create a sheet-based gyroid solid.

    Solid:
        abs(phi) <= sheet_half_thickness

    Fluid:
        abs(phi) > sheet_half_thickness
    """

    phi = gyroid_field(
        x,
        y,
        z,
        period=period,
        level=level,
    )

    return jnp.abs(phi) <= sheet_half_thickness