import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import json

import jax
import jax.numpy as jnp

import plotly.graph_objects as go
from skimage.measure import marching_cubes

from src.fluid import (
    build_filled_gyroid_geometry,
    initialize_distributions,
    macroscopic_fields,
    omega_from_viscosity,
    prepare_geometry,
    run_steps,
    run_steps_with_residual,
    precompute_bfl_auxiliary,
    viscosity_from_omega,
    lbm_fixed_point_loss,
    lbm_fixed_point_residual
)
from src.geometry import Grid

from src.lattice import equilibrium

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt


def heuristic_velocity_initialisation(
    *,
    solid,
    geometry_field,
    inlet_velocity: float,
    boundary_layer_width: float = 2.0,
):
    fluid = ~solid

    # ---------------------------------------------------------
    # 1. Local cross-sectional speed scaling
    # ---------------------------------------------------------
    area_x = jnp.sum(fluid, axis=(1, 2))

    inlet_area = jnp.maximum(area_x[0], 1)

    slice_speed = (
        inlet_velocity
        * inlet_area
        / jnp.maximum(area_x, 1)
    )

    # Shape: (nx, 1, 1)
    slice_speed = slice_speed[:, None, None]

    # ---------------------------------------------------------
    # 2. Wall-following direction from the implicit surface
    # ---------------------------------------------------------
    gx, gy, gz = jnp.gradient(geometry_field)

    grad_norm = jnp.sqrt(
        gx**2 + gy**2 + gz**2 + 1e-12
    )

    normal = jnp.stack(
        [
            gx / grad_norm,
            gy / grad_norm,
            gz / grad_norm,
        ],
        axis=0,
    )

    axial = jnp.zeros_like(normal)
    axial = axial.at[0].set(1.0)

    axial_dot_normal = jnp.sum(
        axial * normal,
        axis=0,
        keepdims=True,
    )

    direction = axial - axial_dot_normal * normal

    direction_norm = jnp.linalg.norm(
        direction,
        axis=0,
        keepdims=True,
    )

    direction = direction / jnp.maximum(
        direction_norm,
        1e-8,
    )

    # Prevent accidental backwards initial flow
    direction = jnp.where(
        direction[0:1] >= 0.0,
        direction,
        -direction,
    )

    # ---------------------------------------------------------
    # 3. No-slip-inspired distance-to-wall profile
    # ---------------------------------------------------------
    wall_distance = distance_transform_edt(
        np.asarray(fluid)
    )

    wall_distance = jnp.asarray(
        wall_distance,
        dtype=jnp.float32,
    )

    wall_weight = jnp.tanh(
        wall_distance / boundary_layer_width
    )

    # ---------------------------------------------------------
    # 4. Assemble the guess
    # ---------------------------------------------------------
    velocity = (
        direction
        * slice_speed[None, ...]
        * wall_weight[None, ...]
    )

    velocity = jnp.where(
        fluid[None, ...],
        velocity,
        0.0,
    )

    # ---------------------------------------------------------
    # 5. Renormalise each slice to preserve target flux
    # ---------------------------------------------------------
    current_flux = jnp.sum(
        velocity[0],
        axis=(1, 2),
    )

    target_flux = inlet_velocity * inlet_area

    correction = (
        target_flux
        / jnp.maximum(current_flux, 1e-12)
    )

    velocity = velocity * correction[None, :, None, None]

    velocity = jnp.where(
        fluid[None, ...],
        velocity,
        0.0,
    )

    return velocity



# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


with open("CONFIG.json", 'rb') as f:
    CONFIG = json.load(f)


grid = Grid(
    nx=int(CONFIG['length'] * CONFIG['resolution']),
    ny=int(CONFIG['diameter'] * CONFIG['resolution']),
    nz=int(CONFIG['diameter'] * CONFIG['resolution']),
    length_x=(int(CONFIG['length'] * CONFIG['resolution'])/int(CONFIG['diameter'] * CONFIG['resolution']))*1.0,
    length_y=1.0,
    length_z=1.0,
)


x, y, z = grid.coordinates()





# ---------------------------------------------------------------------------
# Filled gyroid geometry
# ---------------------------------------------------------------------------

#gyroin, schwarz_p, schwarz_d
params = {"A": 0.0, "B" : 1.0, "C": 0.0}

def init_geometry(params,grid, gyroid_period, gyroid_level, radius, gyroid_solid_side, inlet_buffer_cells, outlet_buffer_cells):

    solid, boundary_field = build_filled_gyroid_geometry(
        params,
        shape=(grid.nx, grid.ny, grid.nz),
        length_x=grid.length_x,
        length_y=grid.length_y,
        length_z=grid.length_z,
        period=gyroid_period,
        cylinder_radius=radius,
        level=gyroid_level,
        solid_side=gyroid_solid_side,
        dtype=jnp.float32,
    )

    centre_y = grid.length_y / 2.0
    centre_z = grid.length_z / 2.0

    radial_distance_squared = (
        (y - centre_y) ** 2
        + (z - centre_z) ** 2
    )

    outside_cylinder = (
        radial_distance_squared > radius**2
    )

    cylinder_boundary_field = (
        radial_distance_squared
        - radius**2
    ).astype(jnp.float32)


    # Remove the gyroid from the inlet buffer while retaining the cylinder wall.
    solid = solid.at[
        :inlet_buffer_cells,
        :,
        :,
    ].set(
        outside_cylinder[
            :inlet_buffer_cells,
            :,
            :,
        ]
    )

    boundary_field = boundary_field.at[
        :inlet_buffer_cells,
        :,
        :,
    ].set(
        cylinder_boundary_field[
            :inlet_buffer_cells,
            :,
            :,
        ]
    )


# Remove the gyroid from the outlet buffer while retaining the cylinder wall.
    solid = solid.at[
        -outlet_buffer_cells:,
        :,
        :,
    ].set(
        outside_cylinder[
            -outlet_buffer_cells:,
            :,
            :,
        ]
    )

    boundary_field = boundary_field.at[
        -outlet_buffer_cells:,
        :,
        :,
    ].set(
        cylinder_boundary_field[
            -outlet_buffer_cells:,
            :,
            :,
        ]
    )


    solid, source_solid, wall_fraction = (
        prepare_geometry(
            solid,
            boundary_field,
        )
    )

    second_fluid_available = (
        precompute_bfl_auxiliary(
            solid
        )
    )

    return solid, source_solid, wall_fraction, boundary_field, second_fluid_available

# ---------------------------------------------------------------------------
# Geometry diagnostics
# ---------------------------------------------------------------------------


solid, source_solid, wall_fraction, boundary_field, second_fluid_available = init_geometry(params, 
                                                                   grid,
                                                                   CONFIG['gyroid_period'],
                                                                   CONFIG['gyroid_level'], 
                                                                   CONFIG['radius'], 
                                                                   CONFIG['gyroid_solid_side'], 
                                                                   CONFIG['inlet_buffer_cells'], 
                                                                   CONFIG['outlet_buffer_cells'])



inlet_density = 1.0
inlet_velocity = 0.003
lattice_viscosity = 0.1


def lattice_viscosity_from_flow_rate(
    *,
    flow_rate_ul_per_min: float,
    tube_length_mm: float,
    tube_diameter_mm: float,
    lattice_diameter: float,
    lattice_inlet_velocity: float,
    physical_kinematic_viscosity: float = 1.0e-6,
) -> float:
    """
    Returns lattice kinematic viscosity that reproduces the
    physical Reynolds number.
    """

    # m^3/s
    Q = flow_rate_ul_per_min * 1e-9 / 60.0

    D = tube_diameter_mm * 1e-3

    area = np.pi * (D / 2.0) ** 2

    U = Q / area

    Re = U * D / physical_kinematic_viscosity

    return lattice_inlet_velocity * lattice_diameter / Re



def init_fluid(solid,inlet_density, inlet_velocity, boundary_field, flow_rate = 500, warm_up = False):
    fluid = ~solid


    distributions = initialize_distributions(
        solid,
        density=inlet_density,
        dtype=jnp.float32,
        inlet_velocity=inlet_velocity,
    )

    # smart init
    if warm_up:
        velocity_guess = heuristic_velocity_initialisation(
            solid=solid,
            geometry_field=boundary_field,
            inlet_velocity=inlet_velocity,
        )

        rho_guess = jnp.ones(
            solid.shape,
            dtype=jnp.float32,
        )

        distributions = equilibrium(
            rho_guess,
            velocity_guess,
        )

    lattice_viscosity = lattice_viscosity_from_flow_rate(
        flow_rate_ul_per_min=flow_rate,
        tube_length_mm=10,
        tube_diameter_mm=2.9,
        lattice_diameter=2 * CONFIG['radius'] * grid.ny,
        lattice_inlet_velocity=inlet_velocity,
    )

    omega = omega_from_viscosity(
        lattice_viscosity
    )

    force = jnp.zeros(
        3,
        dtype=jnp.float32,
    )

    return fluid, distributions, omega, force

fluid, distributions, omega, force = init_fluid(solid, inlet_density, inlet_velocity, boundary_field, 500, False)



# prepare distribution when making a small change to geometry

def warm_start_new_geometry(
    old_distributions,
    old_solid,
    new_solid,
    *,
    force,
    inlet_density: float = 1.0,
    inlet_velocity: float = 0.003,
    new_boundary_field=None,
    use_heuristic_for_new_fluid: bool = True,
):
    """
    Transfer a converged LBM solution from one geometry to a nearby geometry.

    Parameters
    ----------
    old_distributions:
        Converged distributions for the old geometry, with shape
        (19, nx, ny, nz).

    old_solid:
        Boolean mask for the old geometry.

    new_solid:
        Boolean mask for the new geometry.

    force:
        Force vector used by macroscopic_fields.

    inlet_density:
        Reference density.

    inlet_velocity:
        Target inlet velocity.

    new_boundary_field:
        Continuous boundary field for the new geometry. Used by the
        heuristic when initialising newly opened fluid nodes.

    use_heuristic_for_new_fluid:
        If True, use the geometry heuristic for nodes that have become fluid.
        Otherwise, use the nearest old-fluid velocity.

    Returns
    -------
    jax.Array
        Warm-started distributions for the new geometry.
    """
    old_distributions = jnp.asarray(
        old_distributions,
        dtype=jnp.float32,
    )

    old_solid = jnp.asarray(
        old_solid,
        dtype=jnp.bool_,
    )

    new_solid = jnp.asarray(
        new_solid,
        dtype=jnp.bool_,
    )

    if old_solid.shape != new_solid.shape:
        raise ValueError(
            "Warm starting requires the old and new geometries "
            "to use the same grid shape."
        )

    if old_distributions.shape[1:] != old_solid.shape:
        raise ValueError(
            "old_distributions must have shape "
            "(19,) + old_solid.shape."
        )

    old_fluid = ~old_solid
    new_fluid = ~new_solid

    common_fluid = (
        old_fluid
        & new_fluid
    )

    newly_opened = (
        old_solid
        & new_fluid
    )

    newly_closed = (
        old_fluid
        & new_solid
    )

    # Recover the converged old macroscopic fields.
    old_rho, old_velocity = macroscopic_fields(
        old_distributions,
        force,
        old_solid,
    )

    # ---------------------------------------------------------
    # Construct a guess for the new geometry
    # ---------------------------------------------------------

    rho_guess = jnp.where(
        common_fluid,
        old_rho,
        inlet_density,
    )

    velocity_guess = jnp.where(
        common_fluid[None, ...],
        old_velocity,
        0.0,
    )

    if bool(jnp.any(newly_opened)):
        if (
            use_heuristic_for_new_fluid
            and new_boundary_field is not None
        ):
            heuristic_velocity = (
                heuristic_velocity_initialisation(
                    solid=new_solid,
                    geometry_field=new_boundary_field,
                    inlet_velocity=inlet_velocity,
                )
            )

            velocity_guess = jnp.where(
                newly_opened[None, ...],
                heuristic_velocity,
                velocity_guess,
            )

        else:
            # Fill newly opened nodes from the nearest node that was fluid
            # in the old geometry.
            old_fluid_np = np.asarray(
                old_fluid,
                dtype=bool,
            )

            _, nearest_indices = distance_transform_edt(
                ~old_fluid_np,
                return_indices=True,
            )

            old_rho_np = np.asarray(
                old_rho
            )

            old_velocity_np = np.asarray(
                old_velocity
            )

            nearest_rho = old_rho_np[
                nearest_indices[0],
                nearest_indices[1],
                nearest_indices[2],
            ]

            nearest_velocity = old_velocity_np[
                :,
                nearest_indices[0],
                nearest_indices[1],
                nearest_indices[2],
            ]

            nearest_rho = jnp.asarray(
                nearest_rho,
                dtype=jnp.float32,
            )

            nearest_velocity = jnp.asarray(
                nearest_velocity,
                dtype=jnp.float32,
            )

            rho_guess = jnp.where(
                newly_opened,
                nearest_rho,
                rho_guess,
            )

            velocity_guess = jnp.where(
                newly_opened[None, ...],
                nearest_velocity,
                velocity_guess,
            )

    # Solids must have zero velocity.
    velocity_guess = jnp.where(
        new_solid[None, ...],
        0.0,
        velocity_guess,
    )

    rho_guess = jnp.where(
        new_solid,
        inlet_density,
        rho_guess,
    )

    guessed_equilibrium = equilibrium(
        rho_guess,
        velocity_guess,
    )

    # ---------------------------------------------------------
    # Preserve the converged non-equilibrium state wherever the
    # geometry remains fluid
    # ---------------------------------------------------------

    warm_distributions = jnp.where(
        common_fluid[None, ...],
        old_distributions,
        guessed_equilibrium,
    )

    # A benign state for nodes that have become solid.
    rest_velocity = jnp.zeros_like(
        velocity_guess
    )

    rest_equilibrium = equilibrium(
        jnp.full_like(
            rho_guess,
            inlet_density,
        ),
        rest_velocity,
    )

    warm_distributions = jnp.where(
        new_solid[None, ...],
        rest_equilibrium,
        warm_distributions,
    )

    print(
        "Warm-start transfer:"
    )

    print(
        "  common fluid nodes:",
        int(jnp.sum(common_fluid)),
    )

    print(
        "  newly opened nodes:",
        int(jnp.sum(newly_opened)),
    )

    print(
        "  newly closed nodes:",
        int(jnp.sum(newly_closed)),
    )

    print(
        "  fraction of new fluid reused:",
        float(
            jnp.sum(common_fluid)
            / jnp.maximum(
                jnp.sum(new_fluid),
                1,
            )
        ),
    )

    return warm_distributions


old_params = params.copy()

old_solid = solid
old_boundary_field = boundary_field
old_distributions = distributions
old_velocity = velocity
old_rho = rho

distributions = warm_start_new_geometry(
    old_distributions,
    old_solid,
    solid,
    force=force,
    inlet_density=inlet_density,
    inlet_velocity=inlet_velocity,
    new_boundary_field=boundary_field,
    use_heuristic_for_new_fluid=True,
)




# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def train_fast(
    distributions,
    *,
    number_of_blocks,
    steps_per_block,
    convergence_tolerance=1.1e-4,
):
    interior_mask = ~solid

    interior_mask = interior_mask.at[
        :CONFIG["inlet_buffer_cells"],
        :,
        :,
    ].set(False)

    interior_mask = interior_mask.at[
        -CONFIG["outlet_buffer_cells"]:,
        :,
        :,
    ].set(False)

    for block in range(number_of_blocks):

        distributions, block_residual = (
            run_steps_with_residual(
                distributions,
                solid,
                source_solid,
                wall_fraction,
                omega,
                inlet_velocity,
                inlet_density,
                interior_mask,
                number_of_steps=steps_per_block,
            )
        )

        # Synchronize only once per block.
        block_residual = float(
            block_residual
        )

        print(
            f"Block {block + 1:03d}: "
            f"residual = {block_residual:.6e}"
        )

        if (
            block_residual
            < convergence_tolerance
        ):
            print(
                "Early stopping due to convergence."
            )
            break

    return distributions

def train(distributions,number_of_blocks,steps_per_block):
    mean_velocities = []
    outlet_mean_velocities = []
    mass_history = []


    measurement_x = (
        grid.nx
        - CONFIG['outlet_buffer_cells']
        - 2
    )

    target_inlet_velocity = inlet_velocity


    interior_mask = ~solid

    interior_mask = interior_mask.at[
        :CONFIG['inlet_buffer_cells'],
        :,
        :,
    ].set(False)

    interior_mask = interior_mask.at[
        -CONFIG['outlet_buffer_cells']:,
        :,
        :,
    ].set(False)

    for block in range(number_of_blocks):
        previous_distributions = jnp.array(
            distributions,
            copy=True,
        )
        
        completed_steps = block * steps_per_block

        # Use the velocity corresponding to the end of this block.

        current_inlet_velocity = (
            target_inlet_velocity 
        )

        distributions = run_steps(
            distributions,
            solid,
            source_solid,
            wall_fraction,
            omega,
            current_inlet_velocity,  # changed
            inlet_density,
            number_of_steps=steps_per_block,
        ).block_until_ready()

        rho, velocity = macroscopic_fields(
            distributions,
            force,
            solid,
        )

        fluid = ~solid

        fluid_count = jnp.maximum(
            jnp.sum(fluid),
            1,
        )

        mean_velocity_x = (
            jnp.sum(
                jnp.where(
                    fluid,
                    velocity[0],
                    0.0,
                )
            )
            / fluid_count
        )

        outlet_fluid = fluid[
            measurement_x,
            :,
            :,
        ]

        outlet_fluid_count = jnp.maximum(
            jnp.sum(outlet_fluid),
            1,
        )

        outlet_mean_velocity_x = (
            jnp.sum(
                jnp.where(
                    outlet_fluid,
                    velocity[
                        0,
                        measurement_x,
                        :,
                        :,
                    ],
                    0.0,
                )
            )
            / outlet_fluid_count
        )

        total_fluid_mass = jnp.sum(
            jnp.where(
                fluid,
                rho,
                0.0,
            )
        )

        mean_velocities.append(
            float(mean_velocity_x)
        )

        outlet_mean_velocities.append(
            float(outlet_mean_velocity_x)
        )

        mass_history.append(
            float(total_fluid_mass)
        )

        # ------------------------------------------------------------
        # One-step fixed-point diagnostic
        # ------------------------------------------------------------


        block_difference = (
            distributions
            - previous_distributions
        )

        block_residual = jnp.sqrt(
            jnp.sum(
                jnp.where(
                    interior_mask[None, ...],
                    block_difference**2,
                    0.0,
                )
            )
            / jnp.maximum(
                jnp.sum(
                    jnp.where(
                        interior_mask[None, ...],
                        previous_distributions**2,
                        0.0,
                    )
                ),
                1.0e-12,
            )
        )

        block_residual = float(
            block_residual
        )



        print(
            f"Block {block + 1:03d}: "
            f"inlet ux = {current_inlet_velocity:.6e}, "
            f"mean ux = {mean_velocities[-1]:.6e}, "
            f"outlet ux = {outlet_mean_velocities[-1]:.6e}, "
            f"mass = {mass_history[-1]:.6e}, "
            f"delta_loss = {block_residual:.6e}"
        )

        #if low loss and it increases then converged
        if block_residual < 0.00011:
            print("Early stopping due to convergence")
            break

    return distributions,     mean_velocities, outlet_mean_velocities, mass_history


distributions = train(distributions, number_of_blocks=100, steps_per_block=1000)



#mean ux = 5.404868e-03, outlet ux = 7.135368e-03, mass = 2.925739e+05

#mean ux = 5.793736e-03, outlet ux = 7.166232e-03, mass = 3.169567e+05

#mean ux = 5.620393e-03, outlet ux = 6.650450e-03, mass = 9.659979e+05

# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def convergence_plot(number_of_blocks, steps_per_block, simulation_steps, mean_velocities, outlet_mean_velocities):
    simulation_steps = (
        np.arange(
            1,
            number_of_blocks + 1,
        )
        * steps_per_block
    )

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        simulation_steps,
        mean_velocities,
        marker="o",
        label="Domain mean",
    )

    plt.plot(
        simulation_steps,
        outlet_mean_velocities,
        marker="o",
        label="Outlet mean",
    )

    plt.xlabel(
        "LBM time step"
    )

    plt.ylabel(
        "Mean x velocity"
    )

    plt.title(
        "Flow convergence"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/convergence.png",
        dpi=200,
    )


# ---------------------------------------------------------------------------
# Mass history
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    simulation_steps,
    mass_history,
)

plt.xlabel(
    "LBM time step"
)

plt.ylabel(
    "Total fluid mass"
)

plt.title(
    "Fluid mass history"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "plots/fluid_mass.png",
    dpi=200,
)

plt.show()


# ---------------------------------------------------------------------------
# Final macroscopic fields
# ---------------------------------------------------------------------------

rho, velocity = macroscopic_fields(
    distributions,
    force,
    solid,
)

velocity = jax.lax.stop_gradient(
    velocity
)


# ---------------------------------------------------------------------------
# Tracer experiment
# ---------------------------------------------------------------------------

from src.tracer import (
    run_tracer_experiment,
)


tracer_result = run_tracer_experiment(
    velocity,
    solid,
    number_of_steps=100_000,
    pulse_duration=200,
    inlet_concentration=1.0,
    diffusivity=0.01,
    dt=1.0,
    measurement_x=measurement_x,
    contact_distance_cells=2.0,
    score_penalty_weight=0.5,
)

# ---------------------------------------------------------------------------
# Tracer breakthrough curve
# ---------------------------------------------------------------------------

def skewness(mass_flux):
    peak = jnp.argmax(mass_flux)
    after = jnp.sum(mass_flux[peak+1:])
    before = jnp.sum(mass_flux[:peak])
    return (after - before) / before

skewness(tracer_result.outlet_mass_flux)


#0.5, 0.5, 0.0 - 0.78099173

#1.0, 0.0, 0.0 - 0.575047
"""Mean exited contact time: 20780.19921875
Mean residence time: 46247.95701005748
Contact-time standard deviation: 6217.0322265625
Contact-time coefficient of variation: 0.2991805970668793
Contact fraction: 0.449321448802948
Bioreactor score: 0.29973113536834717 (penalty weight=0.5)"""

#0.0, 0.0, 1.0 - 0.44986516
"""Mean exited contact time: 24569.091796875
Mean residence time: 45947.30161696904
Contact-time standard deviation: 5107.4658203125
Contact-time coefficient of variation: 0.20788174867630005
Contact fraction: 0.5347232818603516
Bioreactor score: 0.43078240752220154 (penalty weight=0.5)"""

#0.0, 1.0, 0.0 - 0.4326212
"""Mean exited contact time: 18017.998046875
Mean residence time: 50150.04763361113
Contact fraction: 0.3592817783355713"""


tracer_time = np.arange(
    tracer_result.outlet_mass_flux.size
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    tracer_time,
    tracer_result.outlet_mass_flux, label = 'gyroid'
)

plt.xlabel(
    "Time [lattice timesteps]"
)

plt.ylabel(
    "Tracer mass flux"
)

plt.title(
    "Tracer breakthrough curve"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/tracer.png",
    dpi=200,
)

plt.show()


# ---------------------------------------------------------------------------
# Tracer statistics
# ---------------------------------------------------------------------------

statistics = residence_time_statistics(
    tracer_result.outlet_mass_flux,
    dt=1.0,
    sample_every=1,
)

print(
    "Mean contact time:",
    tracer_result.mean_exited_contact_time,
)

print(
    "Contact-time std:",
    tracer_result.contact_time_standard_deviation,
)

print(
    "Contact-time CV:",
    tracer_result.contact_time_coefficient_of_variation,
)

print(
    "Contact fraction:",
    tracer_result.contact_fraction,
)

print(
    "Bioreactor score:",
    tracer_result.score,
)


# ---------------------------------------------------------------------------
# Tracer mass-balance error
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    tracer_time,
    tracer_result.mass_balance_error,
)

plt.xlabel(
    "Time [lattice timesteps]"
)

plt.ylabel(
    "Injected - exited - remaining"
)

plt.title(
    "Tracer mass-balance error"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "plots/tracer_mass_balance.png",
    dpi=200,
)

plt.show()


# ---------------------------------------------------------------------------
# Standalone gyroid plotting
# ---------------------------------------------------------------------------

def plot_gyroid(*,
    period: float = 0.5,
    level: float = 0.0,
    opacity: float = 0.6,
) -> tuple[np.ndarray, go.Figure]:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    length_x = grid.length_x
    length_y = grid.length_y
    length_z = grid.length_z
    """
    Plot the analytical gyroid level surface phi = level.
    """

    x_coordinates = np.linspace(
        0.0,
        length_x,
        nx,
    )

    y_coordinates = np.linspace(
        0.0,
        length_y,
        ny,
    )

    z_coordinates = np.linspace(
        0.0,
        length_z,
        nz,
    )

    X, Y, Z = np.meshgrid(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        indexing="ij",
    )

    wave_number = (
        2.0
        * np.pi
        / period
    )

    phi = (
        np.sin(wave_number * X)
        * np.cos(wave_number * Y)
        + np.sin(wave_number * Y)
        * np.cos(wave_number * Z)
        + np.sin(wave_number * Z)
        * np.cos(wave_number * X)
    )

    phi = np.array(
        phi,
        dtype=np.float32,
        copy=True,
    )

    if not float(phi.min()) < level < float(phi.max()):
        raise ValueError(
            f"level={level} is outside phi range "
            f"[{phi.min():.4f}, {phi.max():.4f}]"
        )

    dx = (
        length_x
        / max(nx - 1, 1)
    )

    dy = (
        length_y
        / max(ny - 1, 1)
    )

    dz = (
        length_z
        / max(nz - 1, 1)
    )

    vertices, faces, _, _ = marching_cubes(
        phi,
        level=level,
        spacing=(dx, dy, dz),
    )

    figure = go.Figure(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=opacity,
            flatshading=False,
            color="lightgrey",
            name="Gyroid",
        )
    )

    figure.update_layout(
        title=f"Gyroid, period={period}",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        width=1000,
        height=750,
    )

    figure.show()

    return phi, figure


# ---------------------------------------------------------------------------
# Velocity and gyroid plotting
# ---------------------------------------------------------------------------

def plot_tpms_velocity(
    velocity,
    solid,
    *,
    period: float = 0.5,
    cylinder_radius: float = 0.45,
    level: float = 0.0,
    cone_spacing: int = 5,
    surface_opacity: float = 0.25,
    cone_size: float = 2.0,
    title: str = "Velocity field through filled cylindrical TPMS",
) -> go.Figure:
    """
    Plot the actual filled solid geometry used by the solver together with
    velocity cones at fluid nodes.

    Notes
    -----
    The `period`, `cylinder_radius`, and `level` arguments are retained for
    compatibility with existing calls, but the plotted surface is extracted
    directly from `solid`.

    Therefore, the displayed geometry includes the filled gyroid, cylindrical
    wall, and any inlet or outlet buffers contained in the solver mask.
    """
    length_x = grid.length_x
    length_y = grid.length_y
    length_z = grid.length_z

    velocity_np = np.asarray(
        velocity,
        dtype=np.float32,
    )

    solid_np = np.asarray(
        solid,
        dtype=bool,
    )

    if velocity_np.ndim != 4 or velocity_np.shape[0] != 3:
        raise ValueError(
            "velocity must have shape (3, nx, ny, nz), "
            f"but received {velocity_np.shape}"
        )

    nx_local, ny_local, nz_local = velocity_np.shape[1:]

    if solid_np.shape != (nx_local, ny_local, nz_local):
        raise ValueError(
            "solid must have shape "
            f"{(nx_local, ny_local, nz_local)}, "
            f"but received {solid_np.shape}"
        )

    if cone_spacing < 1:
        raise ValueError(
            "cone_spacing must be at least 1"
        )

    if not 0.0 <= surface_opacity <= 1.0:
        raise ValueError(
            "surface_opacity must lie between 0 and 1"
        )

    # ------------------------------------------------------------------
    # Physical coordinates
    # ------------------------------------------------------------------

    x_coordinates = np.linspace(
        0.0,
        length_x,
        nx_local,
        dtype=np.float32,
    )

    y_coordinates = np.linspace(
        0.0,
        length_y,
        ny_local,
        dtype=np.float32,
    )

    z_coordinates = np.linspace(
        0.0,
        length_z,
        nz_local,
        dtype=np.float32,
    )

    X, Y, Z = np.meshgrid(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        indexing="ij",
    )

    dx = length_x / max(nx_local - 1, 1)
    dy = length_y / max(ny_local - 1, 1)
    dz = length_z / max(nz_local - 1, 1)

    # ------------------------------------------------------------------
    # Extract the filled solver geometry
    # ------------------------------------------------------------------

    solid_float = solid_np.astype(
        np.float32
    )

    if solid_np.all():
        raise ValueError(
            "The solid mask contains no fluid nodes."
        )

    if not solid_np.any():
        raise ValueError(
            "The solid mask contains no solid nodes."
        )

    vertices, faces, _, _ = marching_cubes(
        solid_float,
        level=0.5,
        spacing=(dx, dy, dz),
    )

    figure = go.Figure()

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=surface_opacity,
            flatshading=False,
            color="lightgrey",
            name="Filled solid geometry",
            hoverinfo="skip",
            showscale=False,
            lighting={
                "ambient": 0.45,
                "diffuse": 0.75,
                "specular": 0.2,
                "roughness": 0.7,
                "fresnel": 0.1,
            },
            lightposition={
                "x": 100,
                "y": 100,
                "z": 100,
            },
        )
    )

    # ------------------------------------------------------------------
    # Velocity-cone sampling
    # ------------------------------------------------------------------

    x_indices = np.arange(
        1,
        nx_local - 1,
        cone_spacing,
    )

    y_indices = np.arange(
        1,
        ny_local - 1,
        cone_spacing,
    )

    z_indices = np.arange(
        1,
        nz_local - 1,
        cone_spacing,
    )

    ix, iy, iz = np.meshgrid(
        x_indices,
        y_indices,
        z_indices,
        indexing="ij",
    )

    valid_samples = ~solid_np[
        ix,
        iy,
        iz,
    ]

    plot_x = X[ix, iy, iz][valid_samples]
    plot_y = Y[ix, iy, iz][valid_samples]
    plot_z = Z[ix, iy, iz][valid_samples]

    plot_u = velocity_np[
        0,
        ix,
        iy,
        iz,
    ][valid_samples]

    plot_v = velocity_np[
        1,
        ix,
        iy,
        iz,
    ][valid_samples]

    plot_w = velocity_np[
        2,
        ix,
        iy,
        iz,
    ][valid_samples]

    finite_velocity = (
        np.isfinite(plot_u)
        & np.isfinite(plot_v)
        & np.isfinite(plot_w)
    )

    plot_x = plot_x[finite_velocity]
    plot_y = plot_y[finite_velocity]
    plot_z = plot_z[finite_velocity]

    plot_u = plot_u[finite_velocity]
    plot_v = plot_v[finite_velocity]
    plot_w = plot_w[finite_velocity]

    sampled_speed = np.sqrt(
        plot_u**2
        + plot_v**2
        + plot_w**2
    )

    maximum_speed = max(
        float(np.max(sampled_speed))
        if sampled_speed.size
        else 0.0,
        1.0e-12,
    )

    if sampled_speed.size:
        figure.add_trace(
            go.Cone(
                x=plot_x,
                y=plot_y,
                z=plot_z,
                u=plot_u,
                v=plot_v,
                w=plot_w,
                sizemode="absolute",
                sizeref=maximum_speed / cone_size,
                colorscale="Viridis",
                colorbar={
                    "title": "Velocity",
                },
                showscale=True,
                name="Velocity",
            )
        )

    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        width=1050,
        height=750,
    )

    return figure
# ---------------------------------------------------------------------------
# Generate velocity visualisation
# ---------------------------------------------------------------------------

rho, velocity = macroscopic_fields(
    distributions,
    force,
    solid,
)

figure = plot_tpms_velocity(
    velocity,
    solid,
    period=CONFIG['gyroid_period'],
    cylinder_radius=CONFIG['radius'],
    level=CONFIG['gyroid_level'],
    cone_spacing=5,
    surface_opacity=1.0,
)

figure.write_html(
    "plots/plot(0.5,0.5,0).html",
    auto_open=False,
)

figure.show()


print(
    "fluid on inlet:",
    int(jnp.sum(~solid[0])),
)

print(
    "fluid on x=1:",
    int(jnp.sum(~solid[1])),
)


#

fluid = ~solid

flow_rate_by_x = jnp.sum(
    jnp.where(
        fluid,
        rho * velocity[0],
        0.0,
    ),
    axis=(1, 2),
)

interior_flow_rates = flow_rate_by_x[1:-1]

print(
    "interior flow rates:",
    np.asarray(interior_flow_rates),
)

print(
    "mean interior flow rate:",
    float(jnp.mean(interior_flow_rates)),
)

print(
    "relative variation:",
    float(
        jnp.std(interior_flow_rates)
        / jnp.maximum(jnp.abs(jnp.mean(interior_flow_rates)), 1.0e-12)
    ),
)


