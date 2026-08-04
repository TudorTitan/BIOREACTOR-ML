import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np

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
    viscosity_from_omega,
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
# Output folders
# ---------------------------------------------------------------------------

os.makedirs(
    "plots",
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

RESOLUTION = 1.5

nx = int(200 * RESOLUTION)
ny = int(64 * RESOLUTION)
nz = int(64 * RESOLUTION)


length_x=(nx/ny)*1.0
length_y=1.0
length_z=1.0

grid = Grid(
    nx=nx,
    ny=ny,
    nz=nz,
    length_x=length_x,
    length_y=length_y,
    length_z=length_z,
)

x, y, z = grid.coordinates()


# ---------------------------------------------------------------------------
# Cylindrical tube
# ---------------------------------------------------------------------------

radius = 0.45


gyroid_period = 0.25 #0.5
gyroid_level = 0.0

gyroid_solid_side = "positive"

inlet_buffer_cells = 12
outlet_buffer_cells = 16

# ---------------------------------------------------------------------------
# Filled gyroid geometry
# ---------------------------------------------------------------------------

#gyroin, schwarz_p, schwarz_d
params = {"A": 0.0, "B" : 0.0, "C": 1.0}

def init_geometry(params,grid,nx,ny,nz, gyroid_period, gyroid_level, radius, gyroid_solid_side, inlet_buffer_cells, outlet_buffer_cells):

    solid, boundary_field = build_filled_gyroid_geometry(
        params,
        shape=(nx, ny, nz),
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


    solid, source_solid, wall_fraction = prepare_geometry(
        solid,
        boundary_field,
    )

    return solid, source_solid, wall_fraction, boundary_field

# ---------------------------------------------------------------------------
# Geometry diagnostics
# ---------------------------------------------------------------------------


solid, source_solid, wall_fraction, boundary_field = init_geometry(params, grid,nx,ny,nz,gyroid_period, gyroid_level, radius, gyroid_solid_side, inlet_buffer_cells, outlet_buffer_cells)



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
        lattice_diameter=2 * radius * ny,
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



print(
    "grid shape:",
    solid.shape,
)

print(
    "solid fraction:",
    float(jnp.mean(solid)),
)

print(
    "fluid fraction:",
    float(jnp.mean(fluid)),
)

print(
    "fluid nodes at inlet:",
    int(jnp.sum(fluid[0])),
)

print(
    "omega:",
    omega,
)

print(
    "recovered viscosity:",
    viscosity_from_omega(omega),
)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

number_of_blocks = 60
steps_per_block = 1000

mean_velocities = []
outlet_mean_velocities = []
mass_history = []


# Measure slightly upstream from the numerical outlet plane.
measurement_x = (
    nx
    - outlet_buffer_cells
    - 2
)

target_inlet_velocity = inlet_velocity

# Number of simulation steps over which to reach the target.
# With steps_per_block = 500, this is a 20-block ramp.
velocity_ramp_steps = 1000

for block in range(number_of_blocks):
    completed_steps = block * steps_per_block

    # Use the velocity corresponding to the end of this block.
    ramp_fraction = min(
        (completed_steps + steps_per_block)
        / velocity_ramp_steps,
        1.0,
    )

    current_inlet_velocity = (
        target_inlet_velocity * ramp_fraction
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

    print(
        f"Block {block + 1:03d}: "
        f"inlet ux = {current_inlet_velocity:.6e}, "
        f"mean ux = {mean_velocities[-1]:.6e}, "
        f"outlet ux = {outlet_mean_velocities[-1]:.6e}, "
        f"mass = {mass_history[-1]:.6e}"
    )


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
    period=gyroid_period,
    cylinder_radius=radius,
    level=gyroid_level,
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





#-------------------- Cell convergence






class RepeatingCellComparison(NamedTuple):
    cell_index: np.ndarray

    inlet_x: np.ndarray
    outlet_x: np.ndarray

    absolute_l2_difference: np.ndarray
    relative_l2_difference: np.ndarray

    axial_l2_difference: np.ndarray
    transverse_l2_difference: np.ndarray

    inlet_mean_speed: np.ndarray
    outlet_mean_speed: np.ndarray

    inlet_flow_rate: np.ndarray
    outlet_flow_rate: np.ndarray

    common_fluid_fraction: np.ndarray


def _interpolate_x_plane(
    field: np.ndarray,
    *,
    x_position: float,
    length_x: float,
) -> np.ndarray:
    """
    Linearly interpolate an array at an exact axial position.

    Parameters
    ----------
    field:
        Array whose first spatial axis is x.

        Supported shapes:
            (nx, ny, nz)
            (components, nx, ny, nz)

    x_position:
        Position at which to evaluate the field.

    length_x:
        Physical/dimensionless axial length of the grid.
    """
    if field.ndim not in (3, 4):
        raise ValueError(
            "field must have shape (nx, ny, nz) or "
            "(components, nx, ny, nz)"
        )

    nx = field.shape[-3]

    if not 0.0 <= x_position <= length_x:
        raise ValueError(
            f"x_position={x_position} lies outside "
            f"[0, {length_x}]"
        )

    # This matches np.linspace(0, length_x, nx).
    continuous_index = (
        x_position
        * (nx - 1)
        / length_x
    )

    lower_index = int(
        np.floor(continuous_index)
    )

    upper_index = min(
        lower_index + 1,
        nx - 1,
    )

    interpolation_weight = (
        continuous_index - lower_index
    )

    if field.ndim == 4:
        lower_plane = field[
            :,
            lower_index,
            :,
            :,
        ]

        upper_plane = field[
            :,
            upper_index,
            :,
            :,
        ]
    else:
        lower_plane = field[
            lower_index,
            :,
            :,
        ]

        upper_plane = field[
            upper_index,
            :,
            :,
        ]

    return (
        (1.0 - interpolation_weight)
        * lower_plane
        + interpolation_weight
        * upper_plane
    )


def compare_repeating_cell_vector_fields(
    velocity,
    solid,
    *,
    length_x: float,
    cell_length: float,
    start_x: float,
    end_x: float | None = None,
    minimum_fluid_weight: float = 0.5,
    epsilon: float = 1.0e-12,
) -> RepeatingCellComparison:
    """
    Compare inlet and outlet velocity fields for every repeated cell.

    For cell k, compare

        velocity(start_x + k * cell_length, y, z)

    with

        velocity(start_x + (k + 1) * cell_length, y, z).

    Exact cell boundaries are evaluated by linear interpolation in x.

    Only locations fluid at both ends of the cell are included. Because
    the Boolean fluid mask is interpolated, `minimum_fluid_weight`
    determines when an interpolated location is considered fluid.

    Parameters
    ----------
    velocity:
        Velocity field with shape (3, nx, ny, nz).

    solid:
        Boolean mask with shape (nx, ny, nz).

    length_x:
        Total dimensionless domain length.

    cell_length:
        Repeating-cell length. For your TPMS this will normally equal
        `gyroid_period`.

    start_x:
        Beginning of the first complete TPMS cell. This should normally
        be the end of the inlet buffer, not x = 0.

    end_x:
        End of the region containing complete repeating cells. This
        should normally be the beginning of the outlet buffer.

    minimum_fluid_weight:
        Threshold applied to the interpolated fluid mask.

    Returns
    -------
    RepeatingCellComparison
        One result per complete repeating cell.
    """
    velocity_np = np.asarray(
        velocity,
        dtype=np.float64,
    )

    solid_np = np.asarray(
        solid,
        dtype=bool,
    )

    if (
        velocity_np.ndim != 4
        or velocity_np.shape[0] != 3
    ):
        raise ValueError(
            "velocity must have shape "
            "(3, nx, ny, nz)"
        )

    if solid_np.shape != velocity_np.shape[1:]:
        raise ValueError(
            "solid must have shape "
            "velocity.shape[1:]"
        )

    if length_x <= 0.0:
        raise ValueError(
            "length_x must be positive"
        )

    if cell_length <= 0.0:
        raise ValueError(
            "cell_length must be positive"
        )

    if end_x is None:
        end_x = length_x

    if not 0.0 <= start_x < end_x <= length_x:
        raise ValueError(
            "Require "
            "0 <= start_x < end_x <= length_x"
        )

    available_length = end_x - start_x

    number_of_cells = int(
        np.floor(
            available_length / cell_length
            + 1.0e-10
        )
    )

    if number_of_cells < 1:
        raise ValueError(
            "The selected region contains no complete cells."
        )

    fluid_float = (
        ~solid_np
    ).astype(np.float64)

    cell_indices = []
    inlet_positions = []
    outlet_positions = []

    absolute_differences = []
    relative_differences = []

    axial_differences = []
    transverse_differences = []

    inlet_mean_speeds = []
    outlet_mean_speeds = []

    inlet_flow_rates = []
    outlet_flow_rates = []

    common_fluid_fractions = []

    for cell_index in range(number_of_cells):
        inlet_x = (
            start_x
            + cell_index * cell_length
        )

        outlet_x = (
            inlet_x + cell_length
        )

        inlet_velocity = _interpolate_x_plane(
            velocity_np,
            x_position=inlet_x,
            length_x=length_x,
        )

        outlet_velocity = _interpolate_x_plane(
            velocity_np,
            x_position=outlet_x,
            length_x=length_x,
        )

        inlet_fluid_weight = _interpolate_x_plane(
            fluid_float,
            x_position=inlet_x,
            length_x=length_x,
        )

        outlet_fluid_weight = _interpolate_x_plane(
            fluid_float,
            x_position=outlet_x,
            length_x=length_x,
        )

        common_fluid = (
            (inlet_fluid_weight >= minimum_fluid_weight)
            & (
                outlet_fluid_weight
                >= minimum_fluid_weight
            )
        )

        common_count = int(
            np.sum(common_fluid)
        )

        if common_count == 0:
            absolute_difference = np.nan
            relative_difference = np.nan
            axial_difference = np.nan
            transverse_difference = np.nan
            inlet_mean_speed = np.nan
            outlet_mean_speed = np.nan
            inlet_flow_rate = np.nan
            outlet_flow_rate = np.nan
        else:
            inlet_common = inlet_velocity[
                :,
                common_fluid,
            ]

            outlet_common = outlet_velocity[
                :,
                common_fluid,
            ]

            vector_difference = (
                outlet_common - inlet_common
            )

            absolute_difference = np.sqrt(
                np.mean(
                    np.sum(
                        vector_difference**2,
                        axis=0,
                    )
                )
            )

            inlet_field_norm = np.sqrt(
                np.mean(
                    np.sum(
                        inlet_common**2,
                        axis=0,
                    )
                )
            )

            relative_difference = (
                absolute_difference
                / max(
                    inlet_field_norm,
                    epsilon,
                )
            )

            axial_difference = np.sqrt(
                np.mean(
                    (
                        outlet_common[0]
                        - inlet_common[0]
                    ) ** 2
                )
            )

            transverse_difference = np.sqrt(
                np.mean(
                    (
                        outlet_common[1]
                        - inlet_common[1]
                    ) ** 2
                    + (
                        outlet_common[2]
                        - inlet_common[2]
                    ) ** 2
                )
            )

            inlet_speed = np.sqrt(
                np.sum(
                    inlet_common**2,
                    axis=0,
                )
            )

            outlet_speed = np.sqrt(
                np.sum(
                    outlet_common**2,
                    axis=0,
                )
            )

            inlet_mean_speed = float(
                np.mean(inlet_speed)
            )

            outlet_mean_speed = float(
                np.mean(outlet_speed)
            )

            # Lattice flux through the cross-section.
            # Density is omitted here, so this is volumetric rather
            # than mass flux.
            inlet_flow_rate = float(
                np.sum(
                    np.where(
                        inlet_fluid_weight
                        >= minimum_fluid_weight,
                        inlet_velocity[0],
                        0.0,
                    )
                )
            )

            outlet_flow_rate = float(
                np.sum(
                    np.where(
                        outlet_fluid_weight
                        >= minimum_fluid_weight,
                        outlet_velocity[0],
                        0.0,
                    )
                )
            )

        cell_indices.append(cell_index)
        inlet_positions.append(inlet_x)
        outlet_positions.append(outlet_x)

        absolute_differences.append(
            absolute_difference
        )

        relative_differences.append(
            relative_difference
        )

        axial_differences.append(
            axial_difference
        )

        transverse_differences.append(
            transverse_difference
        )

        inlet_mean_speeds.append(
            inlet_mean_speed
        )

        outlet_mean_speeds.append(
            outlet_mean_speed
        )

        inlet_flow_rates.append(
            inlet_flow_rate
        )

        outlet_flow_rates.append(
            outlet_flow_rate
        )

        common_fluid_fractions.append(
            common_count
            / common_fluid.size
        )

    return RepeatingCellComparison(
        cell_index=np.asarray(
            cell_indices,
            dtype=int,
        ),
        inlet_x=np.asarray(
            inlet_positions,
            dtype=float,
        ),
        outlet_x=np.asarray(
            outlet_positions,
            dtype=float,
        ),
        absolute_l2_difference=np.asarray(
            absolute_differences,
            dtype=float,
        ),
        relative_l2_difference=np.asarray(
            relative_differences,
            dtype=float,
        ),
        axial_l2_difference=np.asarray(
            axial_differences,
            dtype=float,
        ),
        transverse_l2_difference=np.asarray(
            transverse_differences,
            dtype=float,
        ),
        inlet_mean_speed=np.asarray(
            inlet_mean_speeds,
            dtype=float,
        ),
        outlet_mean_speed=np.asarray(
            outlet_mean_speeds,
            dtype=float,
        ),
        inlet_flow_rate=np.asarray(
            inlet_flow_rates,
            dtype=float,
        ),
        outlet_flow_rate=np.asarray(
            outlet_flow_rates,
            dtype=float,
        ),
        common_fluid_fraction=np.asarray(
            common_fluid_fractions,
            dtype=float,
        ),
    )


dx = (
    grid.length_x
    / (grid.nx - 1)
)

cell_start_x = (
    inlet_buffer_cells * dx
)

cell_end_x = (
    grid.length_x
    - outlet_buffer_cells * dx
)

cell_comparison = (
    compare_repeating_cell_vector_fields(
        velocity,
        solid,
        length_x=grid.length_x,
        cell_length=gyroid_period,
        start_x=cell_start_x,
        end_x=cell_end_x,
    )
)

for index in range(
    cell_comparison.cell_index.size
):
    print(
        f"Cell {cell_comparison.cell_index[index]:02d}: "
        f"x = "
        f"{cell_comparison.inlet_x[index]:.4f}"
        f" -> "
        f"{cell_comparison.outlet_x[index]:.4f}, "
        f"relative field difference = "
        f"{cell_comparison.relative_l2_difference[index]:.6e}, "
        f"axial difference = "
        f"{cell_comparison.axial_l2_difference[index]:.6e}, "
        f"transverse difference = "
        f"{cell_comparison.transverse_l2_difference[index]:.6e}, "
        f"Qin = "
        f"{cell_comparison.inlet_flow_rate[index]:.6e}, "
        f"Qout = "
        f"{cell_comparison.outlet_flow_rate[index]:.6e}"
    )

plt.figure(
    figsize=(8, 5)
)

plt.semilogy(
    cell_comparison.cell_index,
    cell_comparison.relative_l2_difference,
    marker="o",
)

plt.xlabel(
    "Repeating-cell index"
)

plt.ylabel(
    "Relative inlet/outlet velocity-field difference"
)

plt.title(
    "Development towards a cell-periodic velocity field"
)

plt.grid(
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    "plots/cell_periodicity.png",
    dpi=200,
)

plt.show()
