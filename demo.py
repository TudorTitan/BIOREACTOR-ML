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
    lbm_raw_residual_loss,
    lbm_residual_from_params,
    lbm_fixed_point_loss,
    lbm_fixed_point_residual
)


from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from src.model import run

with open("CONFIG.json", 'rb') as f:
    CONFIG = json.load(f)



#gyroid, schwarz_p, schwarz_d
params = {"A": 0.0, "B" : 0.0, "C": 1.0}




distributions, solid, mean_velocities, outlet_mean_velocities, mass_history = run(params)


params = {"A": 0.0, "B" : 1.0, "C": 0.0}
distributions, solid, mean_velocities, outlet_mean_velocities, mass_history = run(params,warm_start={'distributions' : distributions})


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
Contact-time standard deviation: 6512.4052734375
Contact-time coefficient of variation: 0.3614344596862793
Contact fraction: 0.3592817783355713
Bioreactor score: 0.1785738468170166 (penalty weight=0.5)"""


tracer_time = np.arange(
    tracer_result.outlet_mass_flux.size
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    tracer_time,
    tracer_result.outlet_mass_flux, label = 'schwarz D'
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
    "plots/tracer_schwarz_d.png",
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


