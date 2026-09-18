# src/tracer.py

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt


@partial(
    jax.jit,
    static_argnames=(
        "diffusivity",
        "dt",
    ),
)
def tracer_step_batched(
    fields: jax.Array,
    velocity: jax.Array,
    fluid: jax.Array,
    inlet_values: jax.Array,
    *,
    diffusivity: float = 0.01,
    dt: float = 1.0,
) -> jax.Array:
    """
    Transport multiple passive-scalar fields simultaneously.

    Parameters
    ----------
    fields:
        Shape (number_of_fields, nx, ny, nz).

    velocity:
        Shape (3, nx, ny, nz).

    fluid:
        Boolean fluid mask with shape (nx, ny, nz).

    inlet_values:
        Inlet value for each field, shape (number_of_fields,).
    """
    fields = jnp.where(
        fluid[None, ...],
        fields,
        0.0,
    )

    ux, uy, uz = velocity

    number_of_fields, nx, ny, nz = (
        fields.shape
    )

    # ---------------------------------------------------------
    # x-direction flux
    # ---------------------------------------------------------

    open_x_face = (
        fluid[:-1]
        & fluid[1:]
    )

    ux_face = 0.5 * (
        ux[:-1]
        + ux[1:]
    )

    upwind_x = jnp.where(
        ux_face[None, ...] >= 0.0,
        fields[:, :-1],
        fields[:, 1:],
    )

    internal_flux_x = (
        ux_face[None, ...] * upwind_x
        - diffusivity
        * (
            fields[:, 1:]
            - fields[:, :-1]
        )
    )

    internal_flux_x = jnp.where(
        open_x_face[None, ...],
        internal_flux_x,
        0.0,
    )

    flux_x = jnp.zeros(
        (
            number_of_fields,
            nx + 1,
            ny,
            nz,
        ),
        dtype=fields.dtype,
    )

    flux_x = flux_x.at[
        :,
        1:nx,
        :,
        :,
    ].set(
        internal_flux_x
    )

    inlet_velocity = jnp.maximum(
        ux[0],
        0.0,
    )

    inlet_flux = (
        inlet_velocity[None, ...]
        * inlet_values[:, None, None]
    )

    inlet_flux = jnp.where(
        fluid[0][None, ...],
        inlet_flux,
        0.0,
    )

    flux_x = flux_x.at[
        :,
        0,
        :,
        :,
    ].set(
        inlet_flux
    )

    outlet_velocity = jnp.maximum(
        ux[-1],
        0.0,
    )

    outlet_flux = (
        outlet_velocity[None, ...]
        * fields[:, -1]
    )

    outlet_flux = jnp.where(
        fluid[-1][None, ...],
        outlet_flux,
        0.0,
    )

    flux_x = flux_x.at[
        :,
        nx,
        :,
        :,
    ].set(
        outlet_flux
    )

    # ---------------------------------------------------------
    # y-direction flux
    # ---------------------------------------------------------

    open_y_face = (
        fluid[:, :-1]
        & fluid[:, 1:]
    )

    uy_face = 0.5 * (
        uy[:, :-1]
        + uy[:, 1:]
    )

    upwind_y = jnp.where(
        uy_face[None, ...] >= 0.0,
        fields[:, :, :-1],
        fields[:, :, 1:],
    )

    internal_flux_y = (
        uy_face[None, ...] * upwind_y
        - diffusivity
        * (
            fields[:, :, 1:]
            - fields[:, :, :-1]
        )
    )

    internal_flux_y = jnp.where(
        open_y_face[None, ...],
        internal_flux_y,
        0.0,
    )

    flux_y = jnp.zeros(
        (
            number_of_fields,
            nx,
            ny + 1,
            nz,
        ),
        dtype=fields.dtype,
    )

    flux_y = flux_y.at[
        :,
        :,
        1:ny,
        :,
    ].set(
        internal_flux_y
    )

    # ---------------------------------------------------------
    # z-direction flux
    # ---------------------------------------------------------

    open_z_face = (
        fluid[:, :, :-1]
        & fluid[:, :, 1:]
    )

    uz_face = 0.5 * (
        uz[:, :, :-1]
        + uz[:, :, 1:]
    )

    upwind_z = jnp.where(
        uz_face[None, ...] >= 0.0,
        fields[:, :, :, :-1],
        fields[:, :, :, 1:],
    )

    internal_flux_z = (
        uz_face[None, ...] * upwind_z
        - diffusivity
        * (
            fields[:, :, :, 1:]
            - fields[:, :, :, :-1]
        )
    )

    internal_flux_z = jnp.where(
        open_z_face[None, ...],
        internal_flux_z,
        0.0,
    )

    flux_z = jnp.zeros(
        (
            number_of_fields,
            nx,
            ny,
            nz + 1,
        ),
        dtype=fields.dtype,
    )

    flux_z = flux_z.at[
        :,
        :,
        :,
        1:nz,
    ].set(
        internal_flux_z
    )

    divergence = (
        flux_x[:, 1:]
        - flux_x[:, :-1]
        + flux_y[:, :, 1:]
        - flux_y[:, :, :-1]
        + flux_z[:, :, :, 1:]
        - flux_z[:, :, :, :-1]
    )

    new_fields = (
        fields
        - dt * divergence
    )

    new_fields = jnp.maximum(
        new_fields,
        0.0,
    )

    return jnp.where(
        fluid[None, ...],
        new_fields,
        0.0,
    )


# =============================================================================
# Contact-time extension
# =============================================================================



def build_contact_mask(
    solid: jax.Array | np.ndarray,
    *,
    contact_distance_cells: float = 2.0,
) -> jax.Array:
    """Return fluid cells within `contact_distance_cells` of solid."""
    if contact_distance_cells <= 0.0:
        raise ValueError("contact_distance_cells must be positive.")

    solid_np = np.asarray(solid, dtype=bool)
    fluid_np = ~solid_np
    wall_distance = distance_transform_edt(fluid_np)

    return jnp.asarray(
        fluid_np & (wall_distance <= contact_distance_cells),
        dtype=jnp.bool_,
    )


@partial(
    jax.jit,
    static_argnames=(
        "number_of_steps",
        "pulse_duration",
        "sample_every",
        "diffusivity",
        "dt",
        "measurement_x",
    ),
)
def _run_tracer_scan_with_contact(
    velocity: jax.Array,
    solid: jax.Array,
    contact_mask: jax.Array,
    *,
    number_of_steps: int,
    pulse_duration: int,
    inlet_concentration: float,
    diffusivity: float,
    dt: float,
    measurement_x: int,
    sample_every: int,
):
    """Run tracer transport together with a concentration-weighted contact-age scalar."""
    fluid = ~solid

    concentration_0 = jnp.zeros(solid.shape, dtype=velocity.dtype)
    contact_age_0 = jnp.zeros(solid.shape, dtype=velocity.dtype)
    contact_age_squared_0 = jnp.zeros(
        solid.shape,
        dtype=velocity.dtype,
    )

    inlet_fluid = fluid[0]
    inlet_velocity = jnp.maximum(velocity[0, 0], 0.0)

    measurement_fluid = fluid[measurement_x]
    measurement_velocity = jnp.maximum(
        velocity[0, measurement_x],
        0.0,
    )

    measurement_fluid_count = jnp.maximum(
        jnp.sum(measurement_fluid),
        1,
    )

    zero_inlet = jnp.asarray(0.0, dtype=velocity.dtype)

    def scan_step(carry, step):
        (
            concentration,
            contact_age,
            contact_age_squared,
            cumulative_injected_mass,
            cumulative_outlet_mass,
            cumulative_surface_exposure,
        ) = carry

        active_inlet_concentration = jnp.where(
            step < pulse_duration,
            jnp.asarray(inlet_concentration, dtype=concentration.dtype),
            zero_inlet,
        )

        fields = jnp.stack(
            [
                concentration,
                contact_age,
                contact_age_squared,
            ],
            axis=0,
        )

        inlet_values = jnp.asarray(
            [
                active_inlet_concentration,
                0.0,
                0.0,
            ],
            dtype=velocity.dtype,
        )

        transported_fields = tracer_step_batched(
            fields,
            velocity,
            fluid,
            inlet_values,
            diffusivity=diffusivity,
            dt=dt,
        )

        new_concentration = transported_fields[0]
        transported_contact_age = transported_fields[1]
        transported_contact_age_squared = (
            transported_fields[2]
        )

        contact_increment = dt * contact_mask

        new_contact_age = (
            transported_contact_age
            + contact_increment * new_concentration
        )

        # If A is cumulative contact time, then
        # (A + dt)^2 = A^2 + 2 dt A + dt^2.
        new_contact_age_squared = (
            transported_contact_age_squared
            + 2.0
            * contact_increment
            * transported_contact_age
            + contact_increment
            * contact_increment
            * new_concentration
        )

        new_contact_age = jnp.where(
            fluid, new_contact_age, 0.0
        )
        new_contact_age_squared = jnp.where(
            fluid, new_contact_age_squared, 0.0
        )

        injected_mass_flux = jnp.sum(
            jnp.where(
                inlet_fluid,
                inlet_velocity * active_inlet_concentration,
                0.0,
            )
        )

        measured_concentration = new_concentration[measurement_x]
        measured_contact_age = new_contact_age[measurement_x]
        measured_contact_age_squared = (
            new_contact_age_squared[measurement_x]
        )

        measured_mass_flux = jnp.sum(
            jnp.where(
                measurement_fluid,
                measurement_velocity * measured_concentration,
                0.0,
            )
        )

        measured_contact_age_flux = jnp.sum(
            jnp.where(
                measurement_fluid,
                measurement_velocity * measured_contact_age,
                0.0,
            )
        )

        measured_contact_age_squared_flux = jnp.sum(
            jnp.where(
                measurement_fluid,
                measurement_velocity
                * measured_contact_age_squared,
                0.0,
            )
        )

        measured_mean_concentration = (
            jnp.sum(
                jnp.where(
                    measurement_fluid,
                    measured_concentration,
                    0.0,
                )
            )
            / measurement_fluid_count
        )

        total_tracer_mass = jnp.sum(new_concentration)

        contact_region_tracer_mass = jnp.sum(
            jnp.where(contact_mask, new_concentration, 0.0)
        )

        new_cumulative_injected_mass = (
            cumulative_injected_mass + injected_mass_flux * dt
        )
        new_cumulative_outlet_mass = (
            cumulative_outlet_mass + measured_mass_flux * dt
        )
        new_cumulative_surface_exposure = (
            cumulative_surface_exposure
            + contact_region_tracer_mass * dt
        )

        mass_balance_error = (
            new_cumulative_injected_mass
            - new_cumulative_outlet_mass
            - total_tracer_mass
        )

        new_carry = (
            new_concentration,
            new_contact_age,
            new_contact_age_squared,
            new_cumulative_injected_mass,
            new_cumulative_outlet_mass,
            new_cumulative_surface_exposure,
        )

        measurements = (
            measured_mass_flux,
            measured_contact_age_flux,
            measured_contact_age_squared_flux,
            measured_mean_concentration,
            total_tracer_mass,
            contact_region_tracer_mass,
            new_cumulative_injected_mass,
            new_cumulative_outlet_mass,
            new_cumulative_surface_exposure,
            mass_balance_error,
        )

        return new_carry, measurements

    initial_carry = (
        concentration_0,
        contact_age_0,
        contact_age_squared_0,
        jnp.asarray(0.0, dtype=velocity.dtype),
        jnp.asarray(0.0, dtype=velocity.dtype),
        jnp.asarray(0.0, dtype=velocity.dtype),
    )

    steps = jnp.arange(number_of_steps, dtype=jnp.int32)
    final_carry, histories = jax.lax.scan(
        scan_step,
        initial_carry,
        steps,
    )

    final_concentration = final_carry[0]
    final_contact_age = final_carry[1]
    final_contact_age_squared = final_carry[2]

    (
        outlet_mass_flux,
        outlet_contact_age_flux,
        outlet_contact_age_squared_flux,
        outlet_mean_concentration,
        tracer_mass,
        contact_region_tracer_mass,
        cumulative_injected_mass,
        cumulative_outlet_mass,
        cumulative_surface_exposure,
        mass_balance_error,
    ) = histories

    sample_indices = jnp.arange(
        0,
        number_of_steps,
        sample_every,
    )

    return (
        final_concentration,
        final_contact_age,
        final_contact_age_squared,
        outlet_mass_flux[sample_indices],
        outlet_contact_age_flux[sample_indices],
        outlet_contact_age_squared_flux[sample_indices],
        outlet_mean_concentration[sample_indices],
        tracer_mass[sample_indices],
        contact_region_tracer_mass[sample_indices],
        cumulative_injected_mass[sample_indices],
        cumulative_outlet_mass[sample_indices],
        cumulative_surface_exposure[sample_indices],
        mass_balance_error[sample_indices],
    )


class TracerResult(NamedTuple):
    concentration: jax.Array
    contact_age: jax.Array
    contact_age_squared: jax.Array
    contact_mask: jax.Array

    outlet_mass_flux: np.ndarray
    outlet_contact_age_flux: np.ndarray
    outlet_contact_age_squared_flux: np.ndarray
    outlet_mean_contact_time: np.ndarray
    outlet_contact_time_standard_deviation: np.ndarray
    outlet_mean_concentration: np.ndarray

    tracer_mass: np.ndarray
    contact_region_tracer_mass: np.ndarray

    cumulative_injected_mass: np.ndarray
    cumulative_outlet_mass: np.ndarray
    cumulative_surface_exposure: np.ndarray

    mass_balance_error: np.ndarray

    mean_exited_contact_time: float
    contact_time_standard_deviation: float
    contact_time_coefficient_of_variation: float
    mean_contact_time_per_injected_mass: float

    mean_residence_time: float
    residence_time_standard_deviation: float
    residence_time_coefficient_of_variation: float
    residence_time_skewness: float
    residence_time_peak: float
    total_outlet_mass: float

    contact_fraction: float
    surface_exposure: float
    score: float
    score_penalty_weight: float

    contact_distance_cells: float
    sample_every: int
    dt: float
    measurement_x: int


def run_tracer_experiment(
    velocity: jax.Array,
    solid: jax.Array,
    *,
    number_of_steps: int,
    pulse_duration: int,
    inlet_concentration: float = 1.0,
    diffusivity: float = 0.01,
    dt: float = 1.0,
    measurement_x: int | None = None,
    sample_every: int = 1,
    contact_distance_cells: float = 2.0,
    score_penalty_weight: float = 0.5,
    print_summary: bool = True,
) -> TracerResult:
    """Run tracer transport and compute near-wall contact-time metrics."""
    if velocity.ndim != 4 or velocity.shape[0] != 3:
        raise ValueError("velocity must have shape (3, nx, ny, nz).")

    if solid.shape != velocity.shape[1:]:
        raise ValueError("solid must have shape velocity.shape[1:].")

    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive.")

    if pulse_duration < 0:
        raise ValueError("pulse_duration cannot be negative.")

    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")

    if score_penalty_weight < 0.0:
        raise ValueError(
            "score_penalty_weight cannot be negative."
        )

    nx = solid.shape[0]

    if measurement_x is None:
        measurement_x = nx - 1

    if not 0 <= measurement_x < nx:
        raise ValueError(
            f"measurement_x must lie between 0 and {nx - 1}."
        )

    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    solid = jnp.asarray(solid, dtype=jnp.bool_)

    contact_mask = build_contact_mask(
        solid,
        contact_distance_cells=contact_distance_cells,
    )

    outputs = _run_tracer_scan_with_contact(
        velocity,
        solid,
        contact_mask,
        number_of_steps=number_of_steps,
        pulse_duration=pulse_duration,
        inlet_concentration=inlet_concentration,
        diffusivity=diffusivity,
        dt=dt,
        measurement_x=measurement_x,
        sample_every=sample_every,
    )

    outputs = jax.tree.map(
        lambda value: value.block_until_ready(),
        outputs,
    )

    (
        final_concentration,
        final_contact_age,
        final_contact_age_squared,
        outlet_mass_flux,
        outlet_contact_age_flux,
        outlet_contact_age_squared_flux,
        outlet_mean_concentration,
        tracer_mass,
        contact_region_tracer_mass,
        cumulative_injected_mass,
        cumulative_outlet_mass,
        cumulative_surface_exposure,
        mass_balance_error,
    ) = outputs

    outlet_mass_flux_np = np.asarray(outlet_mass_flux)
    outlet_contact_age_flux_np = np.asarray(
        outlet_contact_age_flux
    )
    outlet_contact_age_squared_flux_np = np.asarray(
        outlet_contact_age_squared_flux
    )

    sampled_dt = dt * sample_every

    total_measured_outlet_mass = (
        np.sum(outlet_mass_flux_np) * sampled_dt
    )
    total_measured_contact_age = (
        np.sum(outlet_contact_age_flux_np) * sampled_dt
    )
    total_measured_contact_age_squared = (
        np.sum(outlet_contact_age_squared_flux_np)
        * sampled_dt
    )

    mean_exited_contact_time = (
        total_measured_contact_age
        / max(total_measured_outlet_mass, 1.0e-12)
    )
    mean_exited_contact_time_squared = (
        total_measured_contact_age_squared
        / max(total_measured_outlet_mass, 1.0e-12)
    )
    contact_time_variance = max(
        mean_exited_contact_time_squared
        - mean_exited_contact_time**2,
        0.0,
    )
    contact_time_standard_deviation = float(
        np.sqrt(contact_time_variance)
    )
    contact_time_coefficient_of_variation = (
        contact_time_standard_deviation
        / max(mean_exited_contact_time, 1.0e-12)
    )

    outlet_mean_contact_time = np.divide(
        outlet_contact_age_flux_np,
        outlet_mass_flux_np,
        out=np.zeros_like(
            outlet_contact_age_flux_np,
            dtype=float,
        ),
        where=outlet_mass_flux_np > 1.0e-12,
    )
    outlet_mean_contact_time_squared = np.divide(
        outlet_contact_age_squared_flux_np,
        outlet_mass_flux_np,
        out=np.zeros_like(
            outlet_contact_age_squared_flux_np,
            dtype=float,
        ),
        where=outlet_mass_flux_np > 1.0e-12,
    )
    outlet_contact_time_standard_deviation = np.sqrt(
        np.maximum(
            outlet_mean_contact_time_squared
            - outlet_mean_contact_time**2,
            0.0,
        )
    )

    residence_statistics = residence_time_statistics(
        outlet_mass_flux_np,
        dt=dt,
        sample_every=sample_every,
    )
    mean_residence_time = float(
        residence_statistics["mean"]
    )

    residence_time_standard_deviation = float(
        residence_statistics["standard_deviation"]
    )

    residence_time_coefficient_of_variation = float(
        residence_statistics["coefficient_of_variation"]
    )

    residence_time_skewness = float(
        residence_statistics["skewness"]
    )

    residence_time_peak = float(
        residence_statistics["peak_time"]
    )

    total_outlet_mass = float(
        residence_statistics["total_outlet_mass"]
    )

    contact_fraction = (
        mean_exited_contact_time
        / max(mean_residence_time, 1.0e-12)
    )
    score = (
        contact_fraction
        - score_penalty_weight
        * contact_time_coefficient_of_variation
    )

    surface_exposure = float(
        np.asarray(cumulative_surface_exposure)[-1]
    )
    final_injected_mass = float(
        np.asarray(cumulative_injected_mass)[-1]
    )
    mean_contact_time_per_injected_mass = (
        surface_exposure
        / max(final_injected_mass, 1.0e-12)
    )

    result = TracerResult(
        concentration=final_concentration,
        contact_age=final_contact_age,
        contact_age_squared=final_contact_age_squared,
        contact_mask=contact_mask,
        outlet_mass_flux=outlet_mass_flux_np,
        outlet_contact_age_flux=outlet_contact_age_flux_np,
        outlet_contact_age_squared_flux=(
            outlet_contact_age_squared_flux_np
        ),
        outlet_mean_contact_time=outlet_mean_contact_time,
        outlet_contact_time_standard_deviation=(
            outlet_contact_time_standard_deviation
        ),
        outlet_mean_concentration=np.asarray(
            outlet_mean_concentration
        ),
        tracer_mass=np.asarray(tracer_mass),
        contact_region_tracer_mass=np.asarray(
            contact_region_tracer_mass
        ),
        cumulative_injected_mass=np.asarray(
            cumulative_injected_mass
        ),
        cumulative_outlet_mass=np.asarray(
            cumulative_outlet_mass
        ),
        cumulative_surface_exposure=np.asarray(
            cumulative_surface_exposure
        ),
        mass_balance_error=np.asarray(mass_balance_error),
        mean_exited_contact_time=float(
            mean_exited_contact_time
        ),
        contact_time_standard_deviation=float(
            contact_time_standard_deviation
        ),
        contact_time_coefficient_of_variation=float(
            contact_time_coefficient_of_variation
        ),
        mean_contact_time_per_injected_mass=float(
            mean_contact_time_per_injected_mass
        ),
        mean_residence_time=mean_residence_time,
        residence_time_standard_deviation=float(
            residence_time_standard_deviation
        ),
        residence_time_coefficient_of_variation=float(
            residence_time_coefficient_of_variation
        ),
        residence_time_skewness=float(
            residence_time_skewness
        ),
        residence_time_peak=float(
            residence_time_peak
        ),
        total_outlet_mass=float(
            total_outlet_mass
        ),
        contact_fraction=float(contact_fraction),
        surface_exposure=surface_exposure,
        score=float(score),
        score_penalty_weight=float(score_penalty_weight),
        contact_distance_cells=float(
            contact_distance_cells
        ),
        sample_every=sample_every,
        dt=dt,
        measurement_x=measurement_x,
    )

    if print_summary:
        print(
            "Tracer experiment complete: "
            f"{number_of_steps} steps, "
            f"{result.outlet_mass_flux.size} stored samples."
        )
        print(
            "Mean exited contact time:",
            result.mean_exited_contact_time,
        )
        print(
            "Mean residence time:",
            result.mean_residence_time,
        )
        print(
            "Residence-time standard deviation:",
            result.residence_time_standard_deviation,
        )
        print(
            "Residence-time coefficient of variation:",
            result.residence_time_coefficient_of_variation,
        )
        print(
            "Residence-time skewness:",
            result.residence_time_skewness,
        )
        print(
            "Residence-time peak:",
            result.residence_time_peak,
        )
        print(
            "Contact-time standard deviation:",
            result.contact_time_standard_deviation,
        )
        print(
            "Contact-time coefficient of variation:",
            result.contact_time_coefficient_of_variation,
        )
        print(
            "Contact fraction:",
            result.contact_fraction,
        )
        print(
            "Surface exposure:",
            result.surface_exposure,
        )
        print(
            "Bioreactor score:",
            result.score,
            f"(penalty weight={result.score_penalty_weight:g})",
        )

    return result


def run_tracer_experiments_batch(
    velocities: jax.Array,
    solids: jax.Array,
    *,
    number_of_steps: int,
    pulse_duration: int,
    inlet_concentration: float = 1.0,
    diffusivity: float = 0.01,
    dt: float = 1.0,
    measurement_x: int | None = None,
    sample_every: int = 1,
    contact_distance_cells: float = 2.0,
    score_penalty_weight: float = 0.5,
) -> list[TracerResult]:
    """Run the expensive tracer scans for several geometries with vmap.

    Contact masks are prepared on the CPU (SciPy EDT), the JAX tracer scan is
    vmapped over geometries, and scalar/statistical post-processing is then
    performed per geometry.
    """
    velocities = jnp.asarray(velocities, dtype=jnp.float32)
    solids = jnp.asarray(solids, dtype=jnp.bool_)

    if velocities.ndim != 5 or velocities.shape[1] != 3:
        raise ValueError("velocities must have shape (batch, 3, nx, ny, nz).")
    if solids.ndim != 4 or solids.shape != (velocities.shape[0],) + velocities.shape[2:]:
        raise ValueError("solids must have shape (batch, nx, ny, nz).")
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive.")
    if pulse_duration < 0:
        raise ValueError("pulse_duration cannot be negative.")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive.")
    if score_penalty_weight < 0.0:
        raise ValueError("score_penalty_weight cannot be negative.")

    nx = solids.shape[1]
    if measurement_x is None:
        measurement_x = nx - 1
    if not 0 <= measurement_x < nx:
        raise ValueError(f"measurement_x must lie between 0 and {nx - 1}.")

    # SciPy EDT is intentionally outside vmap.
    contact_masks = jnp.stack([
        build_contact_mask(s, contact_distance_cells=contact_distance_cells)
        for s in np.asarray(solids)
    ])

    def one(velocity, solid, contact_mask):
        return _run_tracer_scan_with_contact(
            velocity,
            solid,
            contact_mask,
            number_of_steps=number_of_steps,
            pulse_duration=pulse_duration,
            inlet_concentration=inlet_concentration,
            diffusivity=diffusivity,
            dt=dt,
            measurement_x=measurement_x,
            sample_every=sample_every,
        )

    outputs = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))(
        velocities, solids, contact_masks
    )
    outputs = jax.tree.map(lambda value: value.block_until_ready(), outputs)

    results = []
    for i in range(velocities.shape[0]):
        values = jax.tree.map(lambda value: value[i], outputs)
        (
            final_concentration,
            final_contact_age,
            final_contact_age_squared,
            outlet_mass_flux,
            outlet_contact_age_flux,
            outlet_contact_age_squared_flux,
            outlet_mean_concentration,
            tracer_mass,
            contact_region_tracer_mass,
            cumulative_injected_mass,
            cumulative_outlet_mass,
            cumulative_surface_exposure,
            mass_balance_error,
        ) = values

        outlet_mass_flux_np = np.asarray(outlet_mass_flux)
        outlet_contact_age_flux_np = np.asarray(outlet_contact_age_flux)
        outlet_contact_age_squared_flux_np = np.asarray(outlet_contact_age_squared_flux)
        sampled_dt = dt * sample_every

        total_measured_outlet_mass = np.sum(outlet_mass_flux_np) * sampled_dt
        total_measured_contact_age = np.sum(outlet_contact_age_flux_np) * sampled_dt
        total_measured_contact_age_squared = (
            np.sum(outlet_contact_age_squared_flux_np) * sampled_dt
        )

        mean_exited_contact_time = (
            total_measured_contact_age / max(total_measured_outlet_mass, 1.0e-12)
        )
        mean_exited_contact_time_squared = (
            total_measured_contact_age_squared
            / max(total_measured_outlet_mass, 1.0e-12)
        )
        contact_time_variance = max(
            mean_exited_contact_time_squared - mean_exited_contact_time**2, 0.0
        )
        contact_time_standard_deviation = float(np.sqrt(contact_time_variance))
        contact_time_coefficient_of_variation = (
            contact_time_standard_deviation / max(mean_exited_contact_time, 1.0e-12)
        )

        outlet_mean_contact_time = np.divide(
            outlet_contact_age_flux_np,
            outlet_mass_flux_np,
            out=np.zeros_like(outlet_contact_age_flux_np, dtype=float),
            where=outlet_mass_flux_np > 1.0e-12,
        )
        outlet_mean_contact_time_squared = np.divide(
            outlet_contact_age_squared_flux_np,
            outlet_mass_flux_np,
            out=np.zeros_like(outlet_contact_age_squared_flux_np, dtype=float),
            where=outlet_mass_flux_np > 1.0e-12,
        )
        outlet_contact_time_standard_deviation = np.sqrt(
            np.maximum(
                outlet_mean_contact_time_squared - outlet_mean_contact_time**2,
                0.0,
            )
        )

        residence_statistics = residence_time_statistics(
            outlet_mass_flux_np, dt=dt, sample_every=sample_every
        )
        mean_residence_time = float(residence_statistics["mean"])
        residence_time_standard_deviation = float(
            residence_statistics["standard_deviation"]
        )
        residence_time_coefficient_of_variation = float(
            residence_statistics["coefficient_of_variation"]
        )
        residence_time_skewness = float(residence_statistics["skewness"])
        residence_time_peak = float(residence_statistics["peak_time"])
        total_outlet_mass = float(residence_statistics["total_outlet_mass"])

        contact_fraction = (
            mean_exited_contact_time / max(mean_residence_time, 1.0e-12)
        )
        score = (
            contact_fraction
            - score_penalty_weight * contact_time_coefficient_of_variation
        )
        surface_exposure = float(np.asarray(cumulative_surface_exposure)[-1])
        final_injected_mass = float(np.asarray(cumulative_injected_mass)[-1])
        mean_contact_time_per_injected_mass = (
            surface_exposure / max(final_injected_mass, 1.0e-12)
        )

        results.append(TracerResult(
            concentration=final_concentration,
            contact_age=final_contact_age,
            contact_age_squared=final_contact_age_squared,
            contact_mask=contact_masks[i],
            outlet_mass_flux=outlet_mass_flux_np,
            outlet_contact_age_flux=outlet_contact_age_flux_np,
            outlet_contact_age_squared_flux=outlet_contact_age_squared_flux_np,
            outlet_mean_contact_time=outlet_mean_contact_time,
            outlet_contact_time_standard_deviation=outlet_contact_time_standard_deviation,
            outlet_mean_concentration=np.asarray(outlet_mean_concentration),
            tracer_mass=np.asarray(tracer_mass),
            contact_region_tracer_mass=np.asarray(contact_region_tracer_mass),
            cumulative_injected_mass=np.asarray(cumulative_injected_mass),
            cumulative_outlet_mass=np.asarray(cumulative_outlet_mass),
            cumulative_surface_exposure=np.asarray(cumulative_surface_exposure),
            mass_balance_error=np.asarray(mass_balance_error),
            mean_exited_contact_time=float(mean_exited_contact_time),
            contact_time_standard_deviation=contact_time_standard_deviation,
            contact_time_coefficient_of_variation=float(
                contact_time_coefficient_of_variation
            ),
            mean_contact_time_per_injected_mass=float(
                mean_contact_time_per_injected_mass
            ),
            mean_residence_time=mean_residence_time,
            residence_time_standard_deviation=residence_time_standard_deviation,
            residence_time_coefficient_of_variation=(
                residence_time_coefficient_of_variation
            ),
            residence_time_skewness=residence_time_skewness,
            residence_time_peak=residence_time_peak,
            total_outlet_mass=total_outlet_mass,
            contact_fraction=float(contact_fraction),
            surface_exposure=surface_exposure,
            score=float(score),
            score_penalty_weight=float(score_penalty_weight),
            contact_distance_cells=float(contact_distance_cells),
            sample_every=sample_every,
            dt=dt,
            measurement_x=measurement_x,
        ))

    return results


def residence_time_statistics(
    outlet_mass_flux: np.ndarray,
    *,
    dt: float = 1.0,
    sample_every: int = 1,
) -> dict[str, float | np.ndarray]:
    """
    Calculate a normalized residence-time distribution and its statistics.

    Parameters
    ----------
    outlet_mass_flux:
        Sampled tracer mass flux through the measurement plane.

    dt:
        Tracer timestep in lattice-time units.

    sample_every:
        Number of tracer timesteps between stored samples.

    Returns
    -------
    dict
        Contains:

        - time
        - distribution
        - mean
        - standard_deviation
        - coefficient_of_variation
        - skewness
        - peak_time
        - total_outlet_mass
    """
    outlet_mass_flux = np.asarray(
        outlet_mass_flux,
        dtype=float,
    )

    if outlet_mass_flux.ndim != 1:
        raise ValueError(
            "outlet_mass_flux must be one-dimensional."
        )

    if outlet_mass_flux.size == 0:
        raise ValueError(
            "outlet_mass_flux cannot be empty."
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if sample_every <= 0:
        raise ValueError(
            "sample_every must be positive."
        )

    sampled_dt = (
        dt * sample_every
    )

    times = (
        np.arange(
            outlet_mass_flux.size,
            dtype=float,
        )
        * sampled_dt
    )

    total_outlet_mass = (
        np.sum(outlet_mass_flux)
        * sampled_dt
    )

    if total_outlet_mass <= 0.0:
        raise ValueError(
            "No tracer reached the measurement plane."
        )

    # This is a probability density over time:
    #
    # integral E(t) dt = 1.
    residence_time_distribution = (
        outlet_mass_flux
        / total_outlet_mass
    )

    mean_residence_time = np.sum(
        times
        * residence_time_distribution
        * sampled_dt
    )

    centred_times = (
        times - mean_residence_time
    )

    variance = np.sum(
        centred_times**2
        * residence_time_distribution
        * sampled_dt
    )

    variance = max(
        float(variance),
        0.0,
    )

    standard_deviation = float(
        np.sqrt(variance)
    )

    coefficient_of_variation = (
        standard_deviation
        / max(
            float(mean_residence_time),
            1.0e-12,
        )
    )

    if standard_deviation > 1.0e-12:
        third_central_moment = np.sum(
            centred_times**3
            * residence_time_distribution
            * sampled_dt
        )

        skewness = (
            third_central_moment
            / standard_deviation**3
        )
    else:
        skewness = 0.0

    peak_index = int(
        np.argmax(outlet_mass_flux)
    )

    return {
        "time": times,
        "distribution": residence_time_distribution,
        "mean": float(
            mean_residence_time
        ),
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": float(
            coefficient_of_variation
        ),
        "skewness": float(
            skewness
        ),
        "peak_time": float(
            times[peak_index]
        ),
        "total_outlet_mass": float(
            total_outlet_mass
        ),
    }