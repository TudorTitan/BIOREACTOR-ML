import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple







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


def plot_repeating_cell_comparison(
    comparison: RepeatingCellComparison,
    *,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot the relative inlet/outlet velocity-field difference
    across successive repeating geometry cells.

    Parameters
    ----------
    comparison:
        Output from `compare_repeating_cell_vector_fields`.

    save_path:
        Optional path for saving the figure.
        Example:
            "plots/cell_periodicity.png"

    show:
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        comparison.cell_index,
        comparison.relative_l2_difference,
        marker="o",
    )

    ax.set_xlabel(
        "Repeating-cell index"
    )

    ax.set_ylabel(
        "Relative inlet/outlet velocity-field difference"
    )

    ax.set_title(
        "Development towards a cell-periodic velocity field"
    )

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig