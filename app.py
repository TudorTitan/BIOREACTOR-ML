"""Desktop form for configuring and running a TPMS flow model."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import jax.numpy as jnp
import numpy as np

import fluid


class GyroidSimulationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("TPMS flow simulation")
        root.geometry("560x790")
        root.minsize(500, 680)

        self.vars: dict[str, tk.StringVar] = {}
        self.result: dict[str, object] | None = None
        self._make_form()

    def _make_form(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="TPMS lattice and flow settings",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        fields = [
            ("Casing inner diameter (mm)", "casing_diameter", "3.0"),
            ("Monolith outer diameter (mm)", "monolith_diameter", "2.9"),
            ("Casing inner radius (mm, calculated)", "radius", "1.5"),
            ("Monolith length (mm)", "length", "10.0"),
            ("Straight inlet/outlet buffer each end (mm)", "end_buffer", "0.6"),
            ("Lattice cell size (mm)", "cell", "0.6"),
            ("Grid points per cell", "points", "6"),
            ("Flow rate (mL/min)", "flow", "0.3"),
            ("Kinematic viscosity (m²/s)", "viscosity", "1.0e-6"),
            ("Fluid density (kg/m³)", "density", "1000"),
            ("Flow solver duration (s)", "duration", "15.0"),
            ("Physical time per solver step (s)", "dt", "1.0e-4"),
            ("Surface level", "level", "0.0"),
        ]

        for row, (label, key, default) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=4
            )
            value = tk.StringVar(value=default)
            self.vars[key] = value
            entry = ttk.Entry(frame, textvariable=value, width=22)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            if key == "radius":
                entry.state(["readonly"])
            if key == "casing_diameter":
                value.trace_add("write", self._update_radius)

        lattice_row = len(fields) + 1
        ttk.Label(frame, text="Lattice type").grid(
            row=lattice_row, column=0, sticky="w", padx=(0, 12), pady=4
        )
        self.lattice_type = tk.StringVar(value="Gyroid")
        ttk.Combobox(
            frame,
            textvariable=self.lattice_type,
            values=("Gyroid", "Schwarz D", "Schwarz P"),
            state="readonly",
        ).grid(row=lattice_row, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Solid lattice side").grid(
            row=lattice_row + 1, column=0, sticky="w", padx=(0, 12), pady=4
        )
        self.solid_side = tk.StringVar(value="positive")
        ttk.Combobox(
            frame,
            textvariable=self.solid_side,
            values=("positive", "negative"),
            state="readonly",
        ).grid(row=lattice_row + 1, column=1, sticky="ew", pady=4)

        ttk.Label(
            frame,
            text=(
                "Lengths use millimetres. The physical timestep and viscosity "
                "set the LBM scaling; the requested flow rate sets the inlet."
            ),
            wraplength=470,
        ).grid(row=lattice_row + 2, column=0, columnspan=2, sticky="w", pady=(10, 6))

        self.run_button = ttk.Button(frame, text="Run simulation", command=self._start)
        self.run_button.grid(row=lattice_row + 3, column=0, sticky="w", pady=(8, 4))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status, wraplength=470).grid(
            row=lattice_row + 4, column=0, columnspan=2, sticky="w", pady=8
        )

        self.output = tk.Text(frame, height=9, wrap="word", state="disabled")
        self.output.grid(row=lattice_row + 5, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        frame.rowconfigure(lattice_row + 5, weight=1)

    def _update_radius(self, *_: object) -> None:
        try:
            radius = float(self.vars["casing_diameter"].get()) / 2.0
        except ValueError:
            radius = 0.0
        self.vars["radius"].set(f"{radius:g}")

    def _read_parameters(self) -> dict[str, float | int | str]:
        def number(key: str) -> float:
            try:
                value = float(self.vars[key].get())
            except ValueError as exc:
                raise ValueError(f"Enter a valid number for {key}.") from exc
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            return value

        p: dict[str, float | int | str] = {
            "casing_diameter_mm": number("casing_diameter"),
            "monolith_diameter_mm": number("monolith_diameter"),
            "length_mm": number("length"),
            "end_buffer_mm": number("end_buffer"),
            "cell_mm": number("cell"),
            "points_per_cell": int(number("points")),
            "flow_ml_min": number("flow"),
            "viscosity_m2_s": number("viscosity"),
            "fluid_density_kg_m3": number("density"),
            "duration_s": number("duration"),
            "dt_s": number("dt"),
            "level": number("level"),
            "lattice_type": self.lattice_type.get(),
            "solid_side": self.solid_side.get(),
        }
        for key in (
            "casing_diameter_mm", "monolith_diameter_mm", "length_mm", "cell_mm", "flow_ml_min",
            "viscosity_m2_s", "duration_s", "dt_s", "end_buffer_mm",
            "fluid_density_kg_m3",
        ):
            if float(p[key]) <= 0.0:
                raise ValueError(f"{key.replace('_', ' ')} must be positive.")
        if int(p["points_per_cell"]) < 4:
            raise ValueError("Use at least 4 grid points per gyroid cell.")
        if float(p["monolith_diameter_mm"]) > float(p["casing_diameter_mm"]):
            raise ValueError("The monolith outer diameter cannot exceed the casing inner diameter.")
        return p

    def _start(self) -> None:
        try:
            params = self._read_parameters()
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Check the inputs", str(exc), parent=self.root)
            return
        self.run_button.state(["disabled"])
        self.status.set("Building geometry and running the flow solver…")
        self._write_output("")
        threading.Thread(
            target=self._run_worker,
            args=(params,),
            daemon=True,
        ).start()

    def _run_worker(self, p: dict[str, float | int | str]) -> None:
        try:
            result = _run_simulation(p)
        except Exception as exc:  # surface solver and setup errors in the form
            self.root.after(0, self._failed, str(exc))
            return
        self.root.after(0, self._finished, result)

    def _failed(self, error: str) -> None:
        self.run_button.state(["!disabled"])
        self.status.set("Simulation failed.")
        messagebox.showerror("Simulation failed", error, parent=self.root)

    def _finished(self, result: dict[str, object]) -> None:
        self.result = result
        self.run_button.state(["!disabled"])
        self.status.set("Simulation complete.")
        self._write_output(str(result["summary"]))
        if messagebox.askyesno(
            "Save results", "Save the final flow field and geometry as an NPZ file?",
            parent=self.root,
        ):
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save gyroid simulation results",
                defaultextension=".npz",
                filetypes=(("NumPy archive", "*.npz"),),
            )
            if path:
                np.savez_compressed(
                    path,
                    velocity=result["velocity"],
                    density=result["density"],
                    solid=result["solid"],
                    contact_solid=result["contact_solid"],
                    pressure_profile_pa=result["pressure_profile_pa"],
                    pressure_drop_total_pa=result["pressure_drop_total_pa"],
                    pressure_drop_core_pa=(
                        np.nan
                        if result["pressure_drop_core_pa"] is None
                        else result["pressure_drop_core_pa"]
                    ),
                    dx_mm=result["dx_mm"],
                    parameters=result["parameters"],
                    lattice_type=result["lattice_type"],
                )

    def _write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def _run_simulation(p: dict[str, float | int | str]) -> dict[str, object]:
    casing_diameter_mm = float(p.get("casing_diameter_mm", p.get("diameter_mm", 3.0)))
    monolith_diameter_mm = float(p.get("monolith_diameter_mm", casing_diameter_mm))
    geometry_mode = str(p.get("geometry_mode", "tpms"))
    if geometry_mode not in {"tpms", "empty_casing"}:
        raise ValueError("Choose TPMS monolith or empty casing geometry.")
    empty_casing = geometry_mode == "empty_casing"
    if casing_diameter_mm <= 0.0 or monolith_diameter_mm <= 0.0:
        raise ValueError("Casing and monolith diameters must be positive.")
    if not empty_casing and monolith_diameter_mm > casing_diameter_mm:
        raise ValueError("The monolith outer diameter cannot exceed the casing inner diameter.")
    length_mm = float(p["length_mm"])
    end_buffer_mm = float(p.get("end_buffer_mm", 0.6))
    if end_buffer_mm < 0.0:
        raise ValueError("Straight inlet/outlet buffers cannot be negative.")
    period_mm = float(p["cell_mm"])
    dx_target_mm = period_mm / int(p["points_per_cell"])
    ny = max(3, int(round(casing_diameter_mm / dx_target_mm)) + 1)
    dx_mm = casing_diameter_mm / (ny - 1)
    core_steps = max(1, int(round(length_mm / dx_mm)))
    buffer_steps = max(2, int(round(end_buffer_mm / dx_mm)))
    nx = core_steps + 2 * buffer_steps + 1
    length_x_mm = (nx - 1) * dx_mm
    actual_buffer_mm = buffer_steps * dx_mm
    monolith_length_mm = core_steps * dx_mm
    shape = (nx, ny, ny)
    radial_clearance_mm = (
        0.0 if empty_casing else (casing_diameter_mm - monolith_diameter_mm) / 2.0
    )
    clearance_cells = radial_clearance_mm / dx_mm if not empty_casing else 0.0
    clearance_note = (
        f"\nWARNING: radial gap is only {clearance_cells:.2g} grid cells wide; "
        "the current grid cannot resolve the bypass accurately."
        if not empty_casing and 0.0 < clearance_cells < 2.0
        else ""
    )

    # Convert physical viscosity and time to LBM relaxation.
    dt_s = float(p["dt_s"])
    dx_m = dx_mm * 1.0e-3
    viscosity_lattice = (
        float(p["viscosity_m2_s"]) * dt_s / (dx_m * dx_m)
    )
    omega = 1.0 / (3.0 * viscosity_lattice + 0.5)
    if not 0.5 < omega < 2.0:
        raise ValueError(
            f"These viscosity and timestep inputs give omega={omega:.4g}; "
            "adjust them so 0.5 < omega < 2.0."
        )

    lattice_type = str(p.get("lattice_type", "Gyroid"))
    if empty_casing:
        # Straight empty bore used to calibrate the casing-only RTD. The same
        # cylindrical wall and axial domain are retained, with no TPMS insert.
        y = jnp.linspace(0.0, casing_diameter_mm, ny, dtype=jnp.float32)
        z = jnp.linspace(0.0, casing_diameter_mm, ny, dtype=jnp.float32)
        Y, Z = jnp.meshgrid(y, z, indexing="ij")
        radial_distance = jnp.sqrt(
            (Y - casing_diameter_mm / 2.0) ** 2
            + (Z - casing_diameter_mm / 2.0) ** 2
        )
        outside_casing_field = radial_distance - casing_diameter_mm / 2.0
        boundary_field = jnp.broadcast_to(
            outside_casing_field[None, :, :], shape
        ).astype(jnp.float32)
        solid = boundary_field >= 0.0
        lattice_type = "Empty casing (no insert)"
    else:
        solid, boundary_field = fluid.build_filled_tpms_geometry(
            shape,
            lattice_type=lattice_type,
            length_x=length_x_mm,
            length_y=casing_diameter_mm,
            length_z=casing_diameter_mm,
            period=period_mm,
            cylinder_radius=casing_diameter_mm / 2.0,
            monolith_radius=monolith_diameter_mm / 2.0,
            inlet_buffer_mm=actual_buffer_mm,
            outlet_buffer_mm=actual_buffer_mm,
            level=float(p["level"]),
            solid_side=str(p["solid_side"]),
            rotation_degrees=float(p.get("lattice_rotation_deg", 0.0)),
        )
    # The inlet boundary condition acts only on fluid nodes. Use their
    # discretized cross-sectional area to convert flow rate to inlet speed.
    solid_mask_for_area = np.asarray(solid, dtype=bool)
    inlet_boundary_open = (~solid_mask_for_area[0]) & (~solid_mask_for_area[1])
    open_area_m2 = int(np.sum(inlet_boundary_open)) * dx_m * dx_m
    if open_area_m2 <= 0.0:
        raise ValueError("The selected TPMS leaves no open fluid area at the inlet.")
    flow_m3_s = float(p["flow_ml_min"]) * 1.0e-6 / 60.0
    mean_velocity_m_s = flow_m3_s / open_area_m2
    inlet_velocity = mean_velocity_m_s * dt_s / dx_m
    if inlet_velocity > 0.1:
        raise ValueError(
            "The requested flow gives a lattice inlet speed above 0.1. "
            "Increase the physical timestep or reduce the flow rate."
        )

    solid, source_solid, wall_fraction = fluid.prepare_geometry(
        solid, boundary_field
    )
    second_fluid = fluid.precompute_bfl_auxiliary(solid)

    distributions = fluid.initialize_distributions(
        solid,
        inlet_velocity=inlet_velocity,
    )
    requested_flow_ml_min = float(p["flow_ml_min"])
    # Keep the requested inlet speed fixed. Scaling it from a single outlet
    # snapshot confounds transient/boundary effects with the imposed flow and
    # can silently reduce the inlet below the user's requested rate.
    distributions = fluid.run_for_seconds(
        distributions,
        solid,
        source_solid,
        wall_fraction,
        second_fluid,
        omega,
        inlet_velocity,
        duration_seconds=float(p["duration_s"]),
        time_step_seconds=dt_s,
    )
    total_flow_seconds = round(float(p["duration_s"]) / dt_s) * dt_s
    rho, velocity = fluid.macroscopic_fields(
        distributions,
        jnp.zeros((3,), dtype=distributions.dtype),
        solid,
    )
    velocity_np = np.asarray(velocity)
    rho_np = np.asarray(rho)

    # Measure flow on the first and last interior planes, away from the
    # reconstructed inlet/outlet populations. Use rho*u (mass flux) and
    # normalize by the plane's mean density to report equivalent volume flow.
    def plane_flow_ml_min(plane_index: int) -> float:
        plane_solid = solid_mask_for_area[plane_index]
        plane_fluid = ~plane_solid
        if not np.any(plane_fluid):
            return 0.0
        plane_rho = rho_np[plane_index][plane_fluid]
        plane_ux = velocity_np[0, plane_index][plane_fluid]
        mean_density = float(np.mean(plane_rho))
        if not np.isfinite(mean_density) or mean_density <= 0.0:
            return 0.0
        lattice_volume_flux = float(np.sum(plane_rho * plane_ux)) / mean_density
        return lattice_volume_flux * dx_m**3 / dt_s * 60.0e6

    measured_inlet_flow_ml_min = plane_flow_ml_min(1)
    measured_flow_ml_min = plane_flow_ml_min(-2)

    # Isothermal LBM pressure follows p_lu = c_s^2 * rho_lu. Convert the
    # cross-section-averaged density differences to physical pressure units.
    reference_density_kg_m3 = float(p.get("fluid_density_kg_m3", 1000.0))
    pressure_scale_pa = reference_density_kg_m3 * (dx_m / dt_s) ** 2
    cs2 = float(fluid.SPEED_OF_SOUND_SQUARED)

    def plane_mean_density(plane_index: int) -> float:
        plane_fluid = ~solid_mask_for_area[plane_index]
        if not np.any(plane_fluid):
            return float("nan")
        return float(np.mean(rho_np[plane_index][plane_fluid]))

    pressure_density_profile = np.asarray(
        [plane_mean_density(index) for index in range(nx)], dtype=float
    )
    pressure_profile_pa = (
        pressure_density_profile - 1.0
    ) * cs2 * pressure_scale_pa
    pressure_drop_total_pa = float(
        pressure_profile_pa[1] - pressure_profile_pa[-2]
    )
    pressure_drop_core_pa: float | None = None
    if not empty_casing:
        core_inlet_index = min(max(buffer_steps, 1), nx - 2)
        core_outlet_index = min(max(nx - buffer_steps - 1, 1), nx - 2)
        pressure_drop_core_pa = float(
            pressure_profile_pa[core_inlet_index]
            - pressure_profile_pa[core_outlet_index]
        )

    solid_np = np.asarray(solid)
    # Keep the enzyme-contact proxy tied to the TPMS insert, excluding the
    # casing wall and straight buffer sections from the active surface mask.
    active_surface_solid = np.zeros_like(solid_np, dtype=bool)
    if not empty_casing:
        y_coords = np.linspace(0.0, casing_diameter_mm, ny)
        z_coords = np.linspace(0.0, casing_diameter_mm, ny)
        Y_np, Z_np = np.meshgrid(y_coords, z_coords, indexing="ij")
        radial_from_axis = np.sqrt(
            (Y_np - casing_diameter_mm / 2.0) ** 2
            + (Z_np - casing_diameter_mm / 2.0) ** 2
        )
        within_insert = (
            (radial_from_axis <= monolith_diameter_mm / 2.0 + 0.5 * dx_mm)
            & (radial_from_axis < casing_diameter_mm / 2.0 - 0.5 * dx_mm)
        )
        core_planes = np.zeros(nx, dtype=bool)
        core_planes[buffer_steps : nx - buffer_steps] = True
        active_surface_solid = (
            solid_np
            & core_planes[:, None, None]
            & within_insert[None, :, :]
        )
    fluid_density = rho_np[~solid_np]
    if not np.all(np.isfinite(fluid_density)) or not np.all(np.isfinite(velocity_np[:, ~solid_np])):
        raise ValueError("The flow solver produced non-finite values in the fluid region.")
    density_min = float(np.min(fluid_density))
    density_max = float(np.max(fluid_density))
    density_deviation_pct = 100.0 * max(
        abs(density_min - 1.0),
        abs(density_max - 1.0),
    )
    density_note = (
        "\nWARNING: density variation exceeds 5%; treat this flow as provisional."
        if density_deviation_pct > 5.0
        else ""
    )
    simulated_void_volume_uL = float(np.count_nonzero(~solid_np) * dx_mm**3)
    casing_volume_uL = float(
        np.pi * (casing_diameter_mm / 2.0) ** 2 * length_x_mm
    )
    simulated_void_fraction = simulated_void_volume_uL / casing_volume_uL
    outlet_flow_error_pct = (
        100.0 * (measured_flow_ml_min - requested_flow_ml_min)
        / requested_flow_ml_min
    )
    flow_balance_error_pct = (
        100.0
        * (measured_flow_ml_min - measured_inlet_flow_ml_min)
        / max(abs(measured_inlet_flow_ml_min), 1.0e-12)
    )
    actual_seconds = total_flow_seconds
    grid_reference_label = "Grid-sizing reference" if empty_casing else "Cell size"
    geometry_summary = (
        "Geometry: Empty casing (no insert)\n"
        if empty_casing
        else (
            f"Lattice: {lattice_type}\n"
            f"Lattice rotation about flow axis: {float(p.get('lattice_rotation_deg', 0.0)):g}°\n"
            f"Casing ID: {casing_diameter_mm:g} mm; monolith OD: {monolith_diameter_mm:g} mm; "
            f"radial clearance: {radial_clearance_mm:g} mm ({clearance_cells:.2g} grid cells)"
            f"{clearance_note}\n"
        )
    )
    core_summary = (
        f"Casing ID: {casing_diameter_mm:g} mm; empty test section: "
        f"{monolith_length_mm:g} mm; straight end sections: {actual_buffer_mm:g} mm at each end\n"
        if empty_casing
        else (
            f"TPMS core length: {monolith_length_mm:g} mm; straight buffers: "
            f"{actual_buffer_mm:g} mm at each end\n"
        )
    )
    pressure_summary = (
        f"Estimated pressure drop, full domain: {pressure_drop_total_pa:+.5g} Pa\n"
    )
    if pressure_drop_core_pa is not None:
        pressure_summary += (
            f"Estimated pressure drop, TPMS core: {pressure_drop_core_pa:+.5g} Pa\n"
        )
    pressure_summary += (
        f"Pressure conversion assumes fluid density {reference_density_kg_m3:g} kg/m³; "
        "estimate from LBM density and requires benchmark validation.\n"
    )
    summary = (
        f"Grid: {nx} × {ny} × {ny} nodes\n"
        f"{grid_reference_label}: {period_mm:g} mm; grid spacing: {dx_mm:.5g} mm\n"
        f"{geometry_summary}"
        f"{core_summary}"
        f"Total domain: {length_x_mm:g} mm\n"
        f"Voxel-estimated fluid volume: {simulated_void_volume_uL:.4g} µL "
        f"({100.0 * simulated_void_fraction:.1f}% of casing volume)\n"
        f"Requested flow: {float(p['flow_ml_min']):g} mL/min\n"
        f"Measured flow at inlet interior plane: {measured_inlet_flow_ml_min:.6g} mL/min\n"
        f"Measured flow at outlet interior plane after {actual_seconds:g} s: {measured_flow_ml_min:.6g} mL/min\n"
        f"Outlet flow error: {outlet_flow_error_pct:+.1f}%\n"
        f"Inlet-to-outlet balance error: {flow_balance_error_pct:+.1f}%\n"
        f"{pressure_summary}"
        f"Fluid density range (LBM units): {density_min:.5g} to {density_max:.5g}; "
        f"max deviation from 1: {density_deviation_pct:.2f}%{density_note}\n"
        f"Applied LBM inlet speed: {inlet_velocity:.5g}; relaxation omega: {omega:.5g}"
    )
    return {
        "velocity": velocity_np,
        "density": np.asarray(rho),
        "solid": solid_np,
        "contact_solid": active_surface_solid,
        "boundary_field": np.asarray(boundary_field),
        "dx_mm": dx_mm,
        "length_x_mm": length_x_mm,
        "monolith_length_mm": monolith_length_mm,
        "end_buffer_mm": actual_buffer_mm,
        "simulated_void_volume_uL": simulated_void_volume_uL,
        "simulated_void_fraction": simulated_void_fraction,
        "measured_inlet_flow_ml_min": measured_inlet_flow_ml_min,
        "measured_outlet_flow_ml_min": measured_flow_ml_min,
        "outlet_flow_error_pct": outlet_flow_error_pct,
        "flow_balance_error_pct": flow_balance_error_pct,
        "density_deviation_pct": density_deviation_pct,
        "pressure_drop_total_pa": pressure_drop_total_pa,
        "pressure_drop_core_pa": pressure_drop_core_pa,
        "fluid_density_kg_m3": reference_density_kg_m3,
        "pressure_profile_pa": pressure_profile_pa,
        "parameters": np.asarray([monolith_diameter_mm,
                                   float(p["length_mm"]), period_mm,
                                   float(p["flow_ml_min"]), actual_seconds]),
        "lattice_type": lattice_type,
        "summary": summary,
    }


def main() -> None:
    root = tk.Tk()
    GyroidSimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
