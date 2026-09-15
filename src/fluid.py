from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from src.lattice import (
    C,
    OPPOSITE,
    SPEED_OF_SOUND_SQUARED,
    W,
    equilibrium,
)


# D3Q19 directions as Python constants. Python integer shifts allow XLA to
# compile every shift statically.
C_SHIFTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, -1),
    (1, 0, -1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
)

OPPOSITE_PYTHON: tuple[int, ...] = (
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
)

NUMBER_OF_DIRECTIONS = len(C_SHIFTS)


def tpms_field(
        params,
    x: Array,
    y: Array,
    z: Array,
    *,
    period: float,
) -> Array:
    """Return the analytical gyroid field."""

    if period <= 0.0:
        raise ValueError("period must be positive")

    wave_number = 2.0 * jnp.pi / period

    gyroid = (
        jnp.sin(wave_number * x) * jnp.cos(wave_number * y)
        + jnp.sin(wave_number * y) * jnp.cos(wave_number * z)
        + jnp.sin(wave_number * z) * jnp.cos(wave_number * x)
    )

    schwarz_P = (
         jnp.cos(wave_number * y) +
         jnp.cos(wave_number * z) +
         jnp.cos(wave_number * x)
    )

    schwarz_D = (
        jnp.cos(wave_number * x) * jnp.cos(wave_number * y) * jnp.cos(wave_number * z)
        - jnp.sin(wave_number * x) * jnp.sin(wave_number * y) * jnp.sin(wave_number * z)
    )

    return params['A']*gyroid + params['B']*schwarz_P + params['C']*schwarz_D


def gyroid_field(
    x: Array,
    y: Array,
    z: Array,
    *,
    period: float,
) -> Array:
    """Return the analytical gyroid field."""

    if period <= 0.0:
        raise ValueError("period must be positive")

    wave_number = 2.0 * jnp.pi / period

    return (
        jnp.sin(wave_number * x) * jnp.cos(wave_number * y)
        + jnp.sin(wave_number * y) * jnp.cos(wave_number * z)
        + jnp.sin(wave_number * z) * jnp.cos(wave_number * x)
    )


def build_filled_gyroid_geometry(
        params,
    shape: tuple[int, int, int],
    *,
    length_x: float = 2.0,
    length_y: float = 1.0,
    length_z: float = 1.0,
    period: float = 0.5,
    cylinder_radius: float = 0.45,
    level: float = 0.0,
    solid_side: str = "positive",
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array]:
    """
    Construct a one-sided, filled gyroid inside an open cylinder.

    The returned ``boundary_field`` is negative in fluid and non-negative in
    solid. Its zero level is the analytical fluid-solid boundary used to
    calculate link intersection fractions.

    ``solid_side='positive'`` means ``phi >= level`` is solid.
    ``solid_side='negative'`` means ``phi <= level`` is solid.

    The region outside the cylindrical cross-section is also solid. The x=0
    and x=length_x ends are left open for the inlet and outlet.
    """

    if len(shape) != 3 or any(size < 2 for size in shape):
        raise ValueError("shape must contain three dimensions of size >= 2")
    if length_x <= 0.0 or length_y <= 0.0 or length_z <= 0.0:
        raise ValueError("domain lengths must be positive")
    if cylinder_radius <= 0.0:
        raise ValueError("cylinder_radius must be positive")
    if solid_side not in {"positive", "negative"}:
        raise ValueError("solid_side must be 'positive' or 'negative'")

    nx, ny, nz = shape

    x = jnp.linspace(0.0, length_x, nx, dtype=dtype)
    y = jnp.linspace(0.0, length_y, ny, dtype=dtype)
    z = jnp.linspace(0.0, length_z, nz, dtype=dtype)

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    phi = tpms_field(params,X, Y, Z, period=period)

    if solid_side == "positive":
        gyroid_boundary_field = phi - level
    else:
        gyroid_boundary_field = level - phi

    centre_y = 0.5 * length_y
    centre_z = 0.5 * length_z

    # Negative inside the cylinder and positive outside it.
    cylinder_boundary_field = (
        (Y - centre_y) ** 2
        + (Z - centre_z) ** 2
        - cylinder_radius**2
    )

    # Solid is the union of the filled gyroid side and the outside-cylinder
    # region. For sign fields, union corresponds to max().
    boundary_field = jnp.maximum(
        gyroid_boundary_field,
        cylinder_boundary_field,
    ).astype(dtype)

    solid = boundary_field >= 0.0

    return solid, boundary_field


def macroscopic_fields(
    distributions: Array,
    force: Array,
    solid: Array,
) -> tuple[Array, Array]:
    """Recover density and velocity from D3Q19 distributions."""

    rho = jnp.sum(distributions, axis=0)

    momentum = jnp.einsum(
        "ia,ixyz->axyz",
        C,
        distributions,
        optimize=True,
    )

    momentum = momentum + 0.5 * force[:, None, None, None]

    inverse_density = jnp.reciprocal(jnp.maximum(rho, 1.0e-12))
    velocity = momentum * inverse_density[None, ...]

    velocity = jnp.where(solid[None, ...], 0.0, velocity)

    return rho, velocity


def pull_shift_no_wrap(
    field: Array,
    shift: tuple[int, int, int],
    *,
    fill_value: float | bool,
) -> Array:
    """Return ``output[x] = field[x - shift]`` without wrapping."""

    padding = tuple(
        (max(component, 0), max(-component, 0))
        for component in shift
    )

    padded = jnp.pad(
        field,
        pad_width=padding,
        mode="constant",
        constant_values=fill_value,
    )

    starts = tuple(max(-component, 0) for component in shift)

    return lax.dynamic_slice(
        padded,
        start_indices=starts,
        slice_sizes=field.shape,
    )


def valid_source_mask(
    shape: tuple[int, int, int],
    shift: tuple[int, int, int],
) -> Array:
    """True where ``x - shift`` lies inside the computational domain."""

    x = jnp.arange(shape[0])[:, None, None]
    y = jnp.arange(shape[1])[None, :, None]
    z = jnp.arange(shape[2])[None, None, :]

    sx, sy, sz = shift

    return (
        (x - sx >= 0)
        & (x - sx < shape[0])
        & (y - sy >= 0)
        & (y - sy < shape[1])
        & (z - sz >= 0)
        & (z - sz < shape[2])
    )


def guo_force_term(
    velocity: Array,
    force: Array,
    omega: Array | float,
) -> Array:
    """Calculate the Guo forcing term for the D3Q19 lattice."""

    velocity_projection = jnp.einsum(
        "ia,axyz->ixyz",
        C,
        velocity,
        optimize=True,
    )

    lattice_force_projection = jnp.einsum(
        "ia,a->i",
        C,
        force,
        optimize=True,
    )

    velocity_force_projection = jnp.einsum(
        "axyz,a->xyz",
        velocity,
        force,
        optimize=True,
    )

    lattice_force_projection_4d = (
        lattice_force_projection[:, None, None, None]
    )

    force_kernel = (
        3.0
        * (
            lattice_force_projection_4d
            - velocity_force_projection[None, ...]
        )
        + 9.0
        * velocity_projection
        * lattice_force_projection_4d
    )

    return (
        W[:, None, None, None]
        * (1.0 - 0.5 * omega)
        * force_kernel
    )

def precompute_bfl_auxiliary(
    solid: Array,
) -> Array:
    """
    Precompute whether the second fluid node required by
    the BFL q < 0.5 branch exists.

    Returns
    -------
    Array
        Boolean array with shape
        (19, nx, ny, nz).
    """

    fluid = ~solid

    masks = []

    for shift in C_SHIFTS:
        away_shift = tuple(
            -component
            for component in shift
        )

        second_node_is_fluid = (
            pull_shift_no_wrap(
                fluid,
                away_shift,
                fill_value=False,
            )
        )

        second_node_is_valid = (
            valid_source_mask(
                solid.shape,
                away_shift,
            )
        )

        masks.append(
            second_node_is_fluid
            & second_node_is_valid
        )

    return jnp.stack(
        masks,
        axis=0,
    )

def precompute_link_geometry(
    solid: Array,
    boundary_field: Array | None = None,
) -> tuple[Array, Array]:
    """
    Precompute fluid-to-solid links and their wall-intersection fractions.

    Parameters
    ----------
    solid:
        Boolean mask, True in solid.

    boundary_field:
        Scalar field with negative values in fluid, non-negative values in
        solid, and zero on the analytical boundary. When omitted, all wall
        positions default to halfway bounce-back (q = 0.5).

    Returns
    -------
    source_solid:
        Boolean array with shape ``(19, nx, ny, nz)``. At destination x and
        direction i, this is True when source ``x - c_i`` is solid.

    wall_fraction:
        Fraction q of the lattice link from destination fluid node x towards
        source solid node ``x - c_i`` at which the wall is located.
    """

    solid = jnp.asarray(solid, dtype=jnp.bool_)

    if boundary_field is not None:
        boundary_field = jnp.asarray(
            boundary_field,
            dtype=jnp.float32,
        )
        if boundary_field.shape != solid.shape:
            raise ValueError(
                "boundary_field and solid must have identical shapes"
            )

    source_masks = []
    wall_fractions = []

    for shift in C_SHIFTS:
        valid_source = valid_source_mask(solid.shape, shift)

        shifted_solid = pull_shift_no_wrap(
            solid,
            shift,
            fill_value=False,
        )

        wall_link = (
            (~solid)
            & shifted_solid
            & valid_source
        )

        source_masks.append(wall_link)

        if boundary_field is None:
            q = jnp.full(solid.shape, 0.5, dtype=jnp.float32)
        else:
            field_here = boundary_field
            field_source = pull_shift_no_wrap(
                boundary_field,
                shift,
                fill_value=1.0,
            )

            # Linear interpolation of the zero crossing. On wall links,
            # field_here < 0 and field_source >= 0.
            denominator = field_source - field_here
            q = -field_here / jnp.maximum(denominator, 1.0e-12)
            q = jnp.clip(q, 1.0e-6, 1.0)

        wall_fractions.append(
            jnp.where(wall_link, q, 0.5)
        )

    return (
        jnp.stack(source_masks, axis=0),
        jnp.stack(wall_fractions, axis=0),
    )


def pull_stream_and_halfway_bounce(
    post_collision,
    source_solid,
    solid,
    *,
    inlet_velocity,
    inlet_density,
):
    streamed_directions = []

    for direction, shift in enumerate(C_SHIFTS):
        opposite = OPPOSITE_PYTHON[direction]

        pulled = pull_shift_no_wrap(
            post_collision[direction],
            shift,
            fill_value=0.0,
        )

        bounced = post_collision[opposite]

        streamed_directions.append(
            jnp.where(
                source_solid[direction],
                bounced,
                pulled,
            )
        )

    streamed = jnp.stack(
        streamed_directions,
        axis=0,
    )

    return apply_axial_boundaries(
        streamed,
        solid,
        inlet_velocity=inlet_velocity,
        inlet_density=inlet_density,
    )


def pull_stream_and_bounce(
    post_collision: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    solid: Array,
    *,
    inlet_velocity: float,
    inlet_density: float,
) -> Array:
    """
    Non-periodic pull streaming with BFL linear interpolated bounce-back.

    Parameters
    ----------
    post_collision:
        Post-collision distributions with shape
        (19, nx, ny, nz).

    source_solid:
        Boolean fluid-to-solid link mask with shape
        (19, nx, ny, nz).

    wall_fraction:
        BFL wall position q for every lattice link, shape
        (19, nx, ny, nz).

    second_fluid_available:
        Precomputed Boolean mask with shape
        (19, nx, ny, nz).

        True where the second fluid node required for the
        q < 0.5 BFL branch exists and is valid.

    solid:
        Boolean solid mask with shape
        (nx, ny, nz).

    inlet_velocity:
        Prescribed axial inlet velocity.

    inlet_density:
        Fixed outlet/reference density.
    """

    streamed_directions = []

    for direction, shift in enumerate(
        C_SHIFTS
    ):
        opposite = (
            OPPOSITE_PYTHON[direction]
        )

        # -----------------------------------------------------
        # Ordinary pull streaming
        # -----------------------------------------------------

        pulled = pull_shift_no_wrap(
            post_collision[direction],
            shift,
            fill_value=0.0,
        )

        # -----------------------------------------------------
        # Bounce-back quantities
        # -----------------------------------------------------

        halfway_bounce = (
            post_collision[opposite]
        )

        q = wall_fraction[direction]

        # For the q < 0.5 BFL branch we need the population
        # one additional fluid lattice node away from the wall.
        away_shift = tuple(
            -component
            for component in shift
        )

        opposite_at_second_fluid = (
            pull_shift_no_wrap(
                post_collision[opposite],
                away_shift,
                fill_value=0.0,
            )
        )

        can_use_near_wall_branch = (
            second_fluid_available[
                direction
            ]
        )

        # -----------------------------------------------------
        # BFL interpolation
        # -----------------------------------------------------

        # q < 0.5:
        #
        # use the neighbouring fluid node when available.
        near_wall_bounce = (
            2.0
            * q
            * halfway_bounce
            + (
                1.0
                - 2.0 * q
            )
            * opposite_at_second_fluid
        )

        # q >= 0.5:
        #
        # interpolation uses populations at the current
        # fluid node.
        safe_q = jnp.maximum(
            q,
            1.0e-6,
        )

        far_wall_bounce = (
            halfway_bounce
            / (2.0 * safe_q)
            + (
                (
                    2.0 * safe_q
                    - 1.0
                )
                / (
                    2.0 * safe_q
                )
            )
            * post_collision[direction]
        )

        interpolated_bounce = jnp.where(
            q < 0.5,
            jnp.where(
                can_use_near_wall_branch,
                near_wall_bounce,
                halfway_bounce,
            ),
            far_wall_bounce,
        )

        # -----------------------------------------------------
        # Use BFL only on actual fluid-to-solid links.
        # Everywhere else use ordinary streamed population.
        # -----------------------------------------------------

        streamed_direction = jnp.where(
            source_solid[direction],
            interpolated_bounce,
            pulled,
        )

        streamed_directions.append(
            streamed_direction
        )

    streamed = jnp.stack(
        streamed_directions,
        axis=0,
    )

    return apply_axial_boundaries(
        streamed,
        solid,
        inlet_velocity=inlet_velocity,
        inlet_density=inlet_density,
    )

def _lbm_step_zero_force(
    distributions,
    solid,
    source_solid,
    wall_fraction,
    second_fluid_available,
    omega,
    inlet_velocity,
    inlet_density,
):
    rho, velocity = macroscopic_fields(
        distributions,
        jnp.zeros(
            3,
            dtype=distributions.dtype,
        ),
        solid,
    )

    equilibrium_distributions = equilibrium(
        rho,
        velocity,
    )

    post_collision = (
        distributions
        - omega
        * (
            distributions
            - equilibrium_distributions
        )
    )

    return pull_stream_and_bounce(
        post_collision,
        source_solid,
        wall_fraction,
        second_fluid_available,
        solid,
        inlet_velocity=inlet_velocity,
        inlet_density=inlet_density,
    )

@jax.jit
def lbm_step(
    distributions: Array,
    solid: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    omega: Array | float,
    inlet_velocity: float,
    inlet_density: float = 1.0,
) -> Array:
    """Advance one zero-force BFL LBM timestep."""
    return _lbm_step_zero_force(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid_available,
        omega,
        inlet_velocity,
        inlet_density,
    )


@partial(
    jax.jit,
    static_argnames=("number_of_steps",),
    donate_argnums=(0,),
)
def run_steps(
    distributions,
    solid,
    source_solid,
    wall_fraction,
    second_fluid_available,
    omega,
    inlet_velocity,
    inlet_density=1.0,
    *,
    number_of_steps,
):
    def body(
        _,
        current_distributions,
    ):
        return _lbm_step_zero_force(
            current_distributions,
            solid,
            source_solid,
            wall_fraction,
            second_fluid_available,
            omega,
            inlet_velocity,
            inlet_density,
        )

    return lax.fori_loop(
        0,
        number_of_steps,
        body,
        distributions,
    )





def initialize_distributions(
    solid,
    *,
    density: float = 1.0,
    dtype=jnp.float32,
    inlet_velocity: float = 0.0,
    initial_velocity=None,
):
    rho = jnp.full(
        solid.shape,
        density,
        dtype=dtype,
    )

    if initial_velocity is None:
        velocity = jnp.zeros(
            (3,) + solid.shape,
            dtype=dtype,
        )

        velocity = velocity.at[0].set(
            inlet_velocity
        )
    else:
        velocity = jnp.asarray(
            initial_velocity,
            dtype=dtype,
        )

        expected_shape = (3,) + solid.shape

        if velocity.shape != expected_shape:
            raise ValueError(
                "initial_velocity must have shape "
                f"{expected_shape}, but received "
                f"{velocity.shape}"
            )

    velocity = jnp.where(
        solid[None, ...],
        0.0,
        velocity,
    )

    # Keep whatever equilibrium calculation is already used here.
    return equilibrium(rho, velocity)

def prepare_geometry(
    solid: Array,
    boundary_field: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Prepare a geometry and its link-wise boundary data.

    Supplying ``boundary_field`` enables interpolated bounce-back. Omitting it
    retains ordinary halfway bounce-back while using the same solver API.
    """

    solid = jnp.asarray(solid, dtype=jnp.bool_)

    source_solid, wall_fraction = precompute_link_geometry(
        solid,
        boundary_field,
    )

    return solid, source_solid, wall_fraction


def prepare_filled_gyroid_geometry(
        params,
    shape: tuple[int, int, int],
    *,
    length_x: float = 2.0,
    length_y: float = 1.0,
    length_z: float = 1.0,
    period: float = 0.5,
    cylinder_radius: float = 0.45,
    level: float = 0.0,
    solid_side: str = "positive",
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Build and prepare a one-sided filled gyroid with interpolation."""

    solid, boundary_field = build_filled_gyroid_geometry(
        params,
        shape,
        length_x=length_x,
        length_y=length_y,
        length_z=length_z,
        period=period,
        cylinder_radius=cylinder_radius,
        level=level,
        solid_side=solid_side,
        dtype=dtype,
    )

    return prepare_geometry(solid, boundary_field)


def viscosity_from_omega(omega: float) -> float:
    """Return lattice kinematic viscosity from the BGK relaxation value."""

    relaxation_time = 1.0 / omega

    return float(
        SPEED_OF_SOUND_SQUARED
        * (relaxation_time - 0.5)
    )


def omega_from_viscosity(viscosity: float) -> float:
    """Return the BGK relaxation value from lattice viscosity."""

    relaxation_time = viscosity / SPEED_OF_SOUND_SQUARED + 0.5
    return float(1.0 / relaxation_time)


def apply_axial_boundaries(
    distributions: Array,
    solid: Array,
    *,
    inlet_velocity: float,
    inlet_density: float = 1.0,
) -> Array:
    """
    D3Q19 velocity inlet and fixed-density outlet.

    Inlet, x = 0:
        Prescribe ux = inlet_velocity.
        Recover rho from the known streamed populations.
        Reconstruct the unknown cx = +1 populations.

    Outlet, x = nx - 1:
        Prescribe rho = inlet_density.
        Recover ux from the known streamed populations.
        Reconstruct the unknown cx = -1 populations.

    The implementation assumes uy = uz = 0 at both open boundaries.
    """

    result = distributions

    # D3Q19 direction groups for the ordering used in lattice.py.
    zero_x = (0, 3, 4, 5, 6, 15, 16, 17, 18)
    positive_x = (1, 7, 9, 11, 13)
    negative_x = (2, 8, 10, 12, 14)

    # Only impose an open boundary where the boundary node and its adjacent
    # interior node are both fluid.
    inlet_open = (
        (~solid[0, :, :])
        & (~solid[1, :, :])
    )

    outlet_open = (
        (~solid[-1, :, :])
        & (~solid[-2, :, :])
    )

    # ================================================================
    # INLET: x = 0
    # Unknown populations have cx = +1.
    # ================================================================

    inlet_plane = result[:, 0, :, :]

    inlet_zero_sum = sum(
        inlet_plane[direction]
        for direction in zero_x
    )

    inlet_negative_sum = sum(
        inlet_plane[direction]
        for direction in negative_x
    )

    # From:
    #
    # rho = S0 + S+ + S-
    # rho * ux = S+ - S-
    #
    # with prescribed ux:
    #
    # rho = (S0 + 2*S-) / (1 - ux)
    rho_inlet = (
        inlet_zero_sum
        + 2.0 * inlet_negative_sum
    ) / jnp.maximum(
        1.0 - inlet_velocity,
        1.0e-12,
    )

    velocity_inlet = jnp.zeros(
        (3, 1, solid.shape[1], solid.shape[2]),
        dtype=distributions.dtype,
    )

    velocity_inlet = velocity_inlet.at[0, 0].set(
        inlet_velocity
    )

    rho_inlet_3d = rho_inlet[None, :, :]

    equilibrium_inlet = equilibrium(
        rho_inlet_3d,
        velocity_inlet,
    )[:, 0, :, :]

    # Non-equilibrium bounce-back:
    #
    # f_i - f_i^eq = f_opp - f_opp^eq
    for direction in positive_x:
        opposite = OPPOSITE_PYTHON[direction]

        reconstructed = (
            inlet_plane[opposite]
            + equilibrium_inlet[direction]
            - equilibrium_inlet[opposite]
        )

        result = result.at[direction, 0].set(
            jnp.where(
                inlet_open,
                reconstructed,
                result[direction, 0],
            )
        )

    # ================================================================
    # OUTLET: x = nx - 1
    # Unknown populations have cx = -1.
    # ================================================================

    outlet_plane = result[:, -1, :, :]

    outlet_zero_sum = sum(
        outlet_plane[direction]
        for direction in zero_x
    )

    outlet_positive_sum = sum(
        outlet_plane[direction]
        for direction in positive_x
    )

    rho_outlet = jnp.full(
        solid.shape[1:],
        inlet_density,
        dtype=distributions.dtype,
    )

    # This is the exact velocity required for the reconstructed
    # populations to have density rho_outlet.
    ux_outlet = (
        outlet_zero_sum
        + 2.0 * outlet_positive_sum
    ) / jnp.maximum(
        rho_outlet,
        1.0e-12,
    ) - 1.0

    # Do not clip ux_outlet. Clipping destroys the fixed-density
    # condition and causes mass accumulation.

    # Obtain transverse velocity from the adjacent interior plane.
    velocity_outlet = jnp.zeros(
        (3, 1, solid.shape[1], solid.shape[2]),
        dtype=distributions.dtype,
    )

    velocity_outlet = velocity_outlet.at[0, 0].set(
        ux_outlet
    )

    rho_outlet_3d = rho_outlet[None, :, :]

    equilibrium_outlet = equilibrium(
        rho_outlet_3d,
        velocity_outlet,
    )[:, 0, :, :]

    for direction in negative_x:
        opposite = OPPOSITE_PYTHON[direction]

        reconstructed = (
            outlet_plane[opposite]
            + equilibrium_outlet[direction]
            - equilibrium_outlet[opposite]
        )

        result = result.at[direction, -1].set(
            jnp.where(
                outlet_open,
                reconstructed,
                result[direction, -1],
            )
        )

    return result


@partial(
    jax.jit,
    static_argnames=("number_of_steps",),
)
def run_steps_with_residual(
    distributions: Array,
    solid: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    omega: Array | float,
    inlet_velocity: float,
    inlet_density: float,
    interior_mask: Array,
    *,
    number_of_steps: int,
):
    """
    Run one block and return:

        final_distributions,
        relative block-wise residual

    The residual compares the state before and after the whole block.
    """

    initial_distributions = distributions

    def body(
        _,
        current_distributions,
    ):
        return _lbm_step_zero_force(
            current_distributions,
            solid,
            source_solid,
            wall_fraction,
            second_fluid_available,
            omega,
            inlet_velocity,
            inlet_density,
        )

    final_distributions = lax.fori_loop(
        0,
        number_of_steps,
        body,
        distributions,
    )

    block_difference = (
        final_distributions
        - initial_distributions
    )

    mask = interior_mask[None, ...]

    difference_squared = jnp.sum(
        jnp.where(
            mask,
            block_difference**2,
            0.0,
        )
    )

    initial_squared = jnp.sum(
        jnp.where(
            mask,
            initial_distributions**2,
            0.0,
        )
    )

    block_residual = jnp.sqrt(
        difference_squared
        / jnp.maximum(
            initial_squared,
            1.0e-12,
        )
    )

    return (
        final_distributions,
        block_residual,
    )



@jax.jit
def lbm_fixed_point_residual(
    distributions: Array,
    solid: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    omega: Array | float,
    inlet_velocity: float,
    inlet_density: float = 1.0,
) -> Array:
    """
    One-step steady-state residual on physical fluid nodes.

    At a steady LBM solution,

        f = G(f)

    on fluid nodes.
    """

    next_distributions = _lbm_step_zero_force(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid_available,
        omega,
        inlet_velocity,
        inlet_density,
    )

    residual = (
        distributions
        - next_distributions
    )

    return jnp.where(
        (~solid)[None, ...],
        residual,
        0.0,
    )

@jax.jit
def lbm_fixed_point_loss(
    distributions: Array,
    solid: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    omega: Array | float,
    inlet_velocity: float,
    inlet_density: float = 1.0,
) -> Array:
    """
    Relative squared fixed-point residual on fluid nodes only.
    """

    residual = lbm_fixed_point_residual(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid_available,
        omega,
        inlet_velocity,
        inlet_density,
    )

    fluid_mask = (
        (~solid)[None, ...]
    )

    residual_squared = jnp.sum(
        jnp.where(
            fluid_mask,
            residual**2,
            0.0,
        )
    )

    state_squared = jnp.sum(
        jnp.where(
            fluid_mask,
            distributions**2,
            0.0,
        )
    )

    return (
        residual_squared
        / jnp.maximum(
            state_squared,
            1.0e-12,
        )
    )


def lbm_residual_from_params(
    distributions,
    params,
    *,
    shape,
    length_x,
    length_y,
    length_z,
    period,
    cylinder_radius,
    level,
    inlet_buffer_cells,
    outlet_buffer_cells,
    omega,
    inlet_velocity,
    inlet_density=1.0,
    solid_side="positive",
):
    solid, boundary_field = build_filled_gyroid_geometry(
        params,
        shape=shape,
        length_x=length_x,
        length_y=length_y,
        length_z=length_z,
        period=period,
        cylinder_radius=cylinder_radius,
        level=level,
        solid_side=solid_side,
        dtype=distributions.dtype,
    )

    # Clear TPMS from inlet/outlet buffers here
    # exactly as you currently do in init_geometry().
    #
    # Then:

    solid, source_solid, wall_fraction = prepare_geometry(
        solid,
        boundary_field,
    )

    second_fluid_available = precompute_bfl_auxiliary(
        solid
    )

    return lbm_fixed_point_residual(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid_available,
        omega,
        inlet_velocity,
        inlet_density,
    )

@jax.jit
def lbm_raw_residual_loss(
    distributions: Array,
    solid: Array,
    source_solid: Array,
    wall_fraction: Array,
    second_fluid_available: Array,
    omega: Array | float,
    inlet_velocity: float,
    inlet_density: float = 1.0,
) -> Array:
    """
    Raw RMS residual of the steady LBM fixed-point equation

        f - G(f) = 0.

    Returns one scalar. Zero means the current distribution field
    is an exact steady solution of the discrete LBM equations.
    """

    next_distributions = _lbm_step_zero_force(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid_available,
        omega,
        inlet_velocity,
        inlet_density,
    )

    residual = (
        next_distributions
        - distributions
    )

    fluid = ~solid

    residual_squared = jnp.where(
        fluid[None, ...],
        residual**2,
        0.0,
    )

    number_of_fluid_values = (
        jnp.sum(fluid)
        * distributions.shape[0]
    )

    return jnp.sqrt(
        jnp.sum(residual_squared)
        / jnp.maximum(
            number_of_fluid_values,
            1,
        )
    )


