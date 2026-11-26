# TPMS Flow and RTD Simulator

## Start the browser dashboard in WSL

```bash
cd "/mnt/c/Users/greg4/OneDrive - University College London/Goals/UCL/PhD/Research/Future Publications/Primary Publications/ML Reactors/CFD Code"
.venv/bin/python web_app.py
```

Then open `http://127.0.0.1:8765` in the browser. Keep the terminal running while using the dashboard; press `Ctrl+C` there to stop it.

## Main files

- `web_app.py` — browser interface, job handling, RTD and contact summaries.
- `app.py` — shared flow simulation and the older desktop interface.
- `fluid.py` — D3Q19 lattice Boltzmann flow solver and TPMS geometry.
- `tracer.py` — passive tracer, RTD, and near-wall exposure calculations.
- `lattice.py` — D3Q19 lattice constants and equilibrium populations.
- `geometry.py` — geometry helpers.
- `cell_periodicity_experiment.py` — separate grid/periodicity investigation.
- `blanking_approach_500uLmin_notes.md` — notes on the empty-casing blank approach.

## Saved results

- `results/empty_casing_blank/` — empty-casing comparison data and plots.
- `results/gyroid_rtd_comparisons/` — Gyroid RTD overlays and convolution outputs.
- `results/simulation_archives/` — saved full-field simulation archives (`.npz`).

The reported pressure drop is inferred from the LBM density field and physical unit scaling. Treat it as an estimate until checked against a simple pressure-drop benchmark. Near-wall time is a geometric transport proxy; it does not simulate enzyme kinetics or conversion.
