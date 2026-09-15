
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
    lbm_raw_residual_loss,
    lbm_residual_from_params,
    lbm_fixed_point_loss,
    lbm_fixed_point_residual
)

from src.geometry import Grid
from src.lattice import equilibrium


from scipy.ndimage import distance_transform_edt
import numpy as np
import jax.numpy as jnp
import json
import time

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



def init_geometry(params,grid, gyroid_period, gyroid_level, radius, gyroid_solid_side, inlet_buffer_cells, outlet_buffer_cells):

    x, y, z = grid.coordinates()

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




def init_fluid(solid,inlet_density, inlet_velocity, boundary_field, flow_rate = 500):
    fluid = ~solid


    distributions = initialize_distributions(
        solid,
        density=inlet_density,
        dtype=jnp.float32,
        inlet_velocity=inlet_velocity,
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




def run(params, warm_start = None, grid = grid):
    solid, source_solid, wall_fraction, boundary_field, second_fluid_available = init_geometry(params, 
                                                                    grid,
                                                                    CONFIG['gyroid_period'],
                                                                    CONFIG['gyroid_level'], 
                                                                    CONFIG['radius'], 
                                                                    CONFIG['gyroid_solid_side'], 
                                                                    CONFIG['inlet_buffer_cells'], 
                                                                    CONFIG['outlet_buffer_cells'])

    if warm_start:
        distributions = warm_start_new_geometry(
            warm_start['distributions'],
            warm_start['solid'],
            solid,
            force=force,
            inlet_density=CONFIG['inlet_density'],
            inlet_velocity=CONFIG['inlet_velocity'],
            new_boundary_field=boundary_field,
            use_heuristic_for_new_fluid=True,
        )

    else:
        fluid, distributions, omega, force = init_fluid(solid, CONFIG['inlet_density'], CONFIG['inlet_velocity'], boundary_field, 500)



    def train(distributions,number_of_blocks,steps_per_block):
        measurement_x = (
            grid.nx
            - CONFIG['outlet_buffer_cells']
            - 2
        )


        prev_loss = 1.0
        prev_ns_loss = float("inf")
        mean_velocities = []
        outlet_mean_velocities = []
        mass_history = []



        target_inlet_velocity = CONFIG['inlet_velocity']


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
                second_fluid_available,   # <-- THIS MUST BE HERE
                omega,
                current_inlet_velocity,
                CONFIG['inlet_density'],
                number_of_steps=steps_per_block,
            ).block_until_ready()

            rho, velocity = macroscopic_fields(
                distributions,
                force,
                solid,
            )

            raw_loss = float(
                lbm_raw_residual_loss(
                    distributions,
                    solid,
                    source_solid,
                    wall_fraction,
                    second_fluid_available,
                    omega,
                    current_inlet_velocity,
                    CONFIG['inlet_density'],
                )
            )

            ns_loss = float(raw_loss)


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
                f"delta_loss = {block_residual:.6e}, "
                f"ns_loss = {ns_loss:.6e}"

            )

            #if low loss and it increases then converged
            if (block_residual < 0.00011) & (block_residual > prev_loss):
                print("Early stopping due to convergence")
                break

            prev_loss = block_residual
            prev_ns_loss = ns_loss

        return distributions, mean_velocities, outlet_mean_velocities, mass_history


    start = time.perf_counter()

    distributions, mean_velocities, outlet_mean_velocities, mass_history = train(distributions, number_of_blocks=100, steps_per_block=1000)


    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.2f} s")

    return distributions, solid, mean_velocities, outlet_mean_velocities, mass_history



def grid_search(start_params, n_steps = 10, grid = grid):
    pass

