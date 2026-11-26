"""Local browser dashboard for TPMS flow and RTD results."""

from __future__ import annotations

import json
import math
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Avoid reserving most GPU memory before the simulation has allocated its arrays.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

import fluid
import tracer
from app import _run_simulation


HOST = "127.0.0.1"
PORT = 8765
STATE_LOCK = threading.Lock()
JOB: dict[str, object] = {"status": "idle", "message": "Ready"}
PARTIAL_RESULT: dict[str, object] | None = None


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TPMS Flow Results</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { color-scheme: dark; --bg:#101820; --panel:#182631; --edge:#29414e; --ink:#e7f0f3; --muted:#a9bdc6; --teal:#66e0cc; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,Segoe UI,sans-serif; }
    header { padding:22px clamp(18px,4vw,48px); border-bottom:1px solid var(--edge); }
    h1 { margin:0; font-size:24px; } header p { color:var(--muted); margin:4px 0 0; }
    main { display:grid; grid-template-columns:minmax(270px,350px) minmax(0,1fr); gap:18px; padding:18px clamp(12px,3vw,36px); }
    .panel { background:var(--panel); border:1px solid var(--edge); border-radius:12px; padding:16px; }
    h2 { margin:0 0 12px; font-size:17px; }
    label { display:block; color:var(--muted); font-size:12px; margin:9px 0 3px; }
    input,select { width:100%; background:#0d171e; color:var(--ink); border:1px solid #36515e; border-radius:6px; padding:8px; font:inherit; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { border:0; border-radius:7px; padding:10px 14px; background:var(--teal); color:#09211e; font-weight:700; cursor:pointer; width:100%; margin-top:14px; }
    .field-controls { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 8px; }
    .field-controls button { width:auto; margin:0; padding:7px 10px; }
    button:disabled { opacity:.55; cursor:wait; }
    #status { color:var(--teal); min-height:22px; margin:10px 0 0; }
    .note { color:var(--muted); font-size:12px; }
    .results { display:grid; gap:18px; }
    .charts { display:grid; grid-template-columns:1fr; gap:18px; }
    #structure,#rtd { width:100%; height:480px; }
    #rtd { height:390px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(110px,1fr)); gap:10px; }
    .metric { border:1px solid var(--edge); border-radius:8px; padding:10px; }
    .metric span { display:block; color:var(--muted); font-size:12px; }
    .metric strong { font-size:18px; }
    #summary { white-space:pre-wrap; color:var(--muted); font:13px/1.5 ui-monospace,monospace; }
    @media(max-width:900px) { main { grid-template-columns:1fr; } #structure {height:380px;} }
    @media(max-width:560px) { .metrics {grid-template-columns:1fr 1fr;} .grid2 {grid-template-columns:1fr;} }
  </style>
</head>
<body>
<header><h1>TPMS flow and residence-time dashboard</h1><p>Run a baseline structure, inspect its 3D surface, and view the outlet RTD against physical time.</p></header>
<main>
  <section class="panel">
    <h2>Simulation inputs</h2>
    <label>Geometry case</label>
    <select id="geometry_mode" onchange="updateGeometryMode()">
      <option value="empty_casing" selected>Empty casing baseline (no insert)</option>
      <option value="tpms">TPMS monolith</option>
    </select>
    <div class="tpms-only"><label>Lattice type</label>
    <select id="lattice_type"><option>Gyroid</option><option>Schwarz D</option><option>Schwarz P</option></select></div>
    <div class="grid2">
      <div><label>Casing inner diameter (mm)</label><input id="casing_diameter_mm" type="number" value="3.0" step="any"></div>
      <div class="tpms-only"><label>Monolith outer diameter (mm)</label><input id="monolith_diameter_mm" type="number" value="2.9" step="any"></div>
      <div><label id="length_label">Monolith length (mm)</label><input id="length_mm" type="number" value="10" step="any"></div>
      <div><label>Straight inlet/outlet buffer each end (mm)</label><input id="end_buffer_mm" type="number" value="0.6" min="0.01" step="any"></div>
      <div><label id="cell_label">Cell size (mm)</label><input id="cell_mm" type="number" value="0.6" step="any"></div>
      <div><label id="points_label">Grid points per cell</label><input id="points_per_cell" type="number" value="6" min="4" step="1"></div>
      <div><label>Flow rate (mL/min)</label><input id="flow_ml_min" type="number" value="0.5" step="any"></div>
      <div><label>Kinematic viscosity (m²/s)</label><input id="viscosity_m2_s" type="number" value="1e-6" step="any"></div>
      <div><label>Fluid density (kg/m³)</label><input id="fluid_density_kg_m3" type="number" value="1000" step="any"></div>
      <div><label>Acetone diffusivity in water (m²/s, approx. 25 °C)</label><input id="tracer_diffusivity_m2_s" type="number" value="1.28e-9" step="any"></div>
      <div><label>LBM time step (s)</label><input id="dt_s" type="number" value="1e-4" step="any"></div>
      <div class="tpms-only"><label>Surface level</label><input id="level" type="number" value="0" step="any"></div>
      <div class="tpms-only"><label>Lattice rotation around flow axis (°)</label><input id="lattice_rotation_deg" type="number" value="0" step="15"></div>
      <div><label>Flow run duration before tracer (s)</label><input id="flow_duration_s" type="number" value="30" step="any"></div>
      <div><label>Tracer observation duration (s)</label><input id="tracer_duration_s" type="number" value="150" step="any"></div>
      <div><label>Tracer pulse volume (µL)</label><input id="pulse_volume_uL" type="number" value="7.5" step="any" oninput="updatePulseDuration()"></div>
      <div><label>Calculated tracer pulse duration (s)</label><input id="pulse_duration_s" type="number" value="0.9" step="any" readonly></div>
      <div><label>RTD sample interval (s)</label><input id="sample_interval_s" type="number" value="0.1" step="any"></div>
      <div class="tpms-only"><label>Near-wall band for contact estimate (mm)</label><input id="contact_distance_mm" type="number" value="0.2" min="0.001" step="any"></div>
    </div>
    <p class="note" id="geometry_note">Empty 3 mm casing with no insert. The 10 mm test section and 0.6 mm end sections approximate the insert and buffer lengths used in the CFD.</p>
    <div class="tpms-only"><label>Which TPMS side is solid?</label><select id="solid_side"><option value="positive">Positive side: φ ≥ level is solid</option><option value="negative">Negative side: φ ≤ level is solid</option></select></div>
    <p class="note tpms-only">The contact estimate counts tracer time within the selected distance of TPMS solid. It is a grid-based transport proxy, not a simulated enzyme reaction.</p>
    <button id="run" onclick="startRun()">Run flow + tracer</button>
    <div id="status">Ready</div>
    <p class="note">Defaults are set for the 500 µL/min empty-casing calibration run: 3 mm bore, 10 mm test section, 30 s flow settling (about three domain residence times), 7.5 µL pulse (0.9 s at this flow), and 150 s tracer observation. The 0.1 mm grid is a practical first calibration grid. The casing-only experimental curve also contains tubing, injector and detector effects, which this geometry does not model.</p>
  </section>
  <section class="results">
    <div class="panel"><h2>3D tracer concentration heatmap</h2><div class="field-controls"><button type="button" onclick="showTracerField('exposure')">Accumulated exposure during run</button><button type="button" onclick="showTracerField('final')">Instantaneous final snapshot</button></div><div id="structure"></div><div class="note" id="tracer_view_note">Color shows concentration accumulated over time at each fluid location. Brighter regions mean greater tracer exposure. Tracer concentration is zeroed inside solid voxels for the selected solid-side setting.</div></div>
    <div class="panel"><h2>Tracer history along the full length</h2><div id="route"></div><div class="note">Rows are axial positions and columns are time. Color shows cross-section-averaged concentration; the color scale saturates near the 95th percentile so high-concentration regions remain visible.</div></div>
    <div class="panel"><h2>Outlet residence-time distribution</h2><div id="rtd"></div><div class="note">The curve is normalized so its time integral is approximately one. If no tracer reaches the outlet during the observation window, the dashboard reports that instead of drawing a misleading curve.</div></div>
    <button id="downloadRtd" style="display:none;max-width:220px" onclick="downloadRtdCsv()">Download RTD data (CSV)</button>
    <div class="panel"><h2>Flow, contact and RTD summary</h2><div class="metrics" id="metrics"></div><pre id="summary">Run a simulation to see its geometry and results.</pre></div>
  </section>
</main>
<script>
const plotLayout={paper_bgcolor:'#182631',plot_bgcolor:'#182631',font:{color:'#e7f0f3'},margin:{l:8,r:8,t:8,b:8}};
let currentRtd=null;
let latestResult=null;
let tracerView='exposure';
function values(){
 const ids=['casing_diameter_mm','monolith_diameter_mm','length_mm','end_buffer_mm','cell_mm','points_per_cell','flow_ml_min','viscosity_m2_s','fluid_density_kg_m3','tracer_diffusivity_m2_s','dt_s','level','lattice_rotation_deg','flow_duration_s','tracer_duration_s','pulse_duration_s','sample_interval_s','contact_distance_mm'];
 const p={}; for(const id of ids)p[id]=Number(document.getElementById(id).value);
 p.geometry_mode=document.getElementById('geometry_mode').value;
 p.lattice_type=document.getElementById('lattice_type').value;
 p.solid_side=document.getElementById('solid_side').value; return p;
}
function updatePulseDuration(){
 const q=Number(document.getElementById('flow_ml_min').value);
 const volume=Number(document.getElementById('pulse_volume_uL').value);
 document.getElementById('pulse_duration_s').value=(q>0&&volume>0?(volume*60/(q*1000)):0).toPrecision(6);
}
function updateGeometryMode(){
 const isTpms=document.getElementById('geometry_mode').value==='tpms';
 document.querySelectorAll('.tpms-only').forEach(el=>el.style.display=isTpms?'':'none');
 document.getElementById('length_label').textContent=isTpms?'Monolith length (mm)':'Empty casing test-section length (mm)';
 document.getElementById('cell_label').textContent=isTpms?'Cell size (mm)':'Grid-sizing reference (mm)';
 document.getElementById('points_label').textContent=isTpms?'Grid points per cell':'Grid subdivisions';
 document.getElementById('geometry_note').textContent=isTpms
  ? 'The monolith and the annular casing clearance are represented. Small radial gaps need a finer grid to resolve reliably.'
  : 'Empty 3 mm casing with no insert. Grid spacing is reference size ÷ subdivisions (defaults: 0.6 ÷ 6 = 0.1 mm). The 10 mm test section and 0.6 mm end sections match the simulated monolith and buffer lengths.';
}
document.getElementById('flow_ml_min').addEventListener('input',updatePulseDuration);
updateGeometryMode();
updatePulseDuration();
async function startRun(){
 const btn=document.getElementById('run'); btn.disabled=true;
 document.getElementById('metrics').innerHTML='';
 document.getElementById('summary').textContent='';
 document.getElementById('status').textContent='Starting…';
 try{
  const response=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values())});
  const data=await response.json(); if(!response.ok)throw new Error(data.error||'Could not start simulation');
  await poll();
 }catch(e){document.getElementById('status').textContent='Error: '+e.message;btn.disabled=false;}
}
async function poll(){
 const btn=document.getElementById('run');
 try{
  const r=await fetch('/status'); const job=await r.json();
  document.getElementById('status').textContent=job.message||job.status;
  if(job.status==='running'){setTimeout(poll,1200);return;}
  if(job.status==='done'){render(job.result);btn.disabled=false;return;}
  if(job.status==='error'){document.getElementById('summary').textContent=job.error||'Simulation failed';btn.disabled=false;return;}
  btn.disabled=false;
 }catch(e){document.getElementById('status').textContent='Connection error: '+e.message;btn.disabled=false;}
}
function render(result){
 latestResult=result;
 const tracerStopReason=result.rtd_warning||'The run returned no tracer diagnostic. Restart the dashboard from the current project folder before trying again.';
 document.getElementById('summary').textContent=result.flow_summary+(result.rtd_warning?'\n\nRTD: '+result.rtd_warning:(!result.rtd?'\n\nRTD: No RTD data was returned. '+tracerStopReason:''));
 renderStructure(result);
 if(result.tracer_space_time){
  const s=result.tracer_space_time;
  const z=s.x_mm.map((_,ix)=>s.concentration.map(row=>row[ix]));
  const positive=z.flat().filter(v=>Number.isFinite(v)&&v>0).sort((a,b)=>a-b);
  const colorMax=positive.length?positive[Math.floor(0.95*(positive.length-1))]:s.max_concentration;
  Plotly.newPlot('route',[{type:'heatmap',x:s.time_s,y:s.x_mm,z:z,colorscale:'YlOrRd',zmin:0,zmax:Math.max(colorMax,1e-12),colorbar:{title:'Mean C / inlet C'},hovertemplate:'t = %{x:.2f} s<br>x = %{y:.2f} mm<br>mean C / inlet C = %{z:.4g}<extra></extra>'}],{...plotLayout,xaxis:{title:'Time after tracer injection (s)',rangemode:'tozero',gridcolor:'#29414e'},yaxis:{title:'Axial position from inlet (mm)',rangemode:'tozero',gridcolor:'#29414e'},margin:{l:65,r:20,t:10,b:50}},{responsive:true});
 }else document.getElementById('route').textContent='Tracer travel history unavailable. '+tracerStopReason;
 if(result.rtd){
  currentRtd=result.rtd; document.getElementById('downloadRtd').style.display='block';
  Plotly.newPlot('rtd',[{type:'scatter',mode:'lines',x:result.rtd.time_s,y:result.rtd.E_per_s,line:{color:'#66e0cc',width:3},hovertemplate:'t = %{x:.6g} s<br>E(t) = %{y:.6g} s⁻¹<extra></extra>'}],{...plotLayout,xaxis:{title:'Time (s)',rangemode:'tozero',gridcolor:'#29414e'},yaxis:{title:'E(t) (s⁻¹)',rangemode:'tozero',gridcolor:'#29414e'},margin:{l:60,r:20,t:10,b:50}},{responsive:true});
 }else document.getElementById('rtd').textContent='RTD curve unavailable. '+tracerStopReason;
 const m=result.metrics||{}; const items=[];
 if(m.pressure_total_pa!==undefined)items.push(['ΔP full section',m.pressure_total_pa+' Pa']);
 if(m.pressure_core_pa!==undefined&&m.pressure_core_pa!==null)items.push(['ΔP TPMS core',m.pressure_core_pa+' Pa']);
 if(m.contact_mean_s!==undefined){items.push(['Mean near-wall time',m.contact_mean_s+' s']);items.push(['RT spent near TPMS',m.contact_fraction_pct+'%']);items.push(['Contact-time CV',m.contact_cv]);items.push(['Near-wall band',m.contact_band_mm+' mm']);}
 if(m.mean_s!==undefined){items.push(['Mean RT',m.mean_s+' s'],['RTD std. dev.',m.std_s+' s'],['RTD skewness',m.skewness],['Peak time',m.peak_s+' s'],['Tracer recovered',m.recovery_pct+'%'],['End tail / peak',m.tail_pct+'%']);}
 document.getElementById('metrics').innerHTML=items.map(([a,b])=>`<div class="metric"><span>${a}</span><strong>${b}</strong></div>`).join('');
}
function showTracerField(view){
 tracerView=view;
 if(latestResult)renderStructure(latestResult);
}
function renderStructure(result){
 const s=result.structure;
 if(window.Plotly&&s&&s.value){
  const traces=[{type:'isosurface',name:'TPMS solid boundary',x:s.x,y:s.y,z:s.z,value:s.value,isomin:-0.0001,isomax:0.0001,surface:{count:1},caps:{x:{show:false},y:{show:false},z:{show:false}},colorscale:[[0,'#718096'],[1,'#cbd5e0']],showscale:false,opacity:0.22}];
  const c=tracerView==='exposure'?result.tracer_exposure_field:result.tracer_field;
  let hasTracerHeatmap=false;
  if(c&&c.value){
   const cmax=c.value.reduce((m,v)=>Math.max(m,Number(v)||0),0);
   if(cmax>0){
    if(tracerView==='exposure'){
     traces.push({type:'volume',name:'Accumulated tracer exposure',x:c.x,y:c.y,z:c.z,value:c.value,isomin:0.01*cmax,isomax:cmax,surface:{count:18},caps:{x:{show:false},y:{show:false},z:{show:false}},colorscale:'YlOrRd',cmin:0,cmax:cmax,showscale:true,colorbar:{title:'(C / inlet C) · s'},opacity:0.12});
    }else{
     traces.push({type:'isosurface',name:'Tracer concentration at final time',x:c.x,y:c.y,z:c.z,value:c.value,isomin:0.009,isomax:0.011,surface:{count:1},caps:{x:{show:false},y:{show:false},z:{show:false}},colorscale:[[0,'#ffca5c'],[1,'#ff4a30']],cmin:0.009,cmax:0.011,showscale:false,opacity:0.95});
    }
    hasTracerHeatmap=true;
   }
  }
  Plotly.newPlot('structure',traces,{...plotLayout,scene:{xaxis:{title:'x (mm)'},yaxis:{title:'y (mm)'},zaxis:{title:'z (mm)'},aspectmode:'data'},legend:{orientation:'h'},margin:{l:0,r:0,t:8,b:0}},{responsive:true});
  document.getElementById('tracer_view_note').textContent=!hasTracerHeatmap
   ? 'No tracer concentration was recorded in the displayed field. Check the flow and tracer summary below.'
   : tracerView==='exposure'
   ? 'Color shows the time integral of concentration at each location. Bright regions indicate locations with greater tracer exposure. Solid TPMS voxels are masked out.'
   : 'This is the concentration remaining at the end of the observation window. Tracer is zeroed inside solid voxels; switch to accumulated exposure to see where it spent time during the run.';
 }else document.getElementById('structure').textContent='3D plotting library did not load. Check browser internet access.';
}
function downloadRtdCsv(){
 if(!currentRtd)return;
 const rows=['time_s,E_per_s',...currentRtd.time_s.map((t,i)=>`${t},${currentRtd.E_per_s[i]}`)];
 const blob=new Blob([rows.join('\n')],{type:'text/csv'}); const a=document.createElement('a');
 a.href=URL.createObjectURL(blob); a.download='rtd_curve.csv'; a.click(); URL.revokeObjectURL(a.href);
}
</script>
</body></html>"""


def _set_job(**updates: object) -> None:
    with STATE_LOCK:
        JOB.update(updates)


def _complete_without_rtd(result: dict[str, object], warning: str) -> None:
    global PARTIAL_RESULT
    result["rtd_warning"] = warning
    _set_job(status="done", message="Flow complete; RTD unavailable", result=result)
    with STATE_LOCK:
        PARTIAL_RESULT = None


def _json_safe(value: object) -> object:
    """Replace non-finite floats so the HTTP response is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _numeric_inputs(payload: dict[str, object]) -> dict[str, float | int | str]:
    p: dict[str, float | int | str] = {
        "casing_diameter_mm": float(payload["casing_diameter_mm"]),
        "monolith_diameter_mm": float(payload["monolith_diameter_mm"]),
        "length_mm": float(payload["length_mm"]),
        "end_buffer_mm": float(payload.get("end_buffer_mm", 0.6)),
        "cell_mm": float(payload["cell_mm"]),
        "points_per_cell": int(payload["points_per_cell"]),
        "flow_ml_min": float(payload["flow_ml_min"]),
        "viscosity_m2_s": float(payload["viscosity_m2_s"]),
        "fluid_density_kg_m3": float(payload.get("fluid_density_kg_m3", 1000.0)),
        "tracer_diffusivity_m2_s": float(payload["tracer_diffusivity_m2_s"]),
        "dt_s": float(payload["dt_s"]),
        "duration_s": float(payload["flow_duration_s"]),
        "level": float(payload["level"]),
        "lattice_rotation_deg": float(payload.get("lattice_rotation_deg", 0.0)),
        "geometry_mode": str(payload.get("geometry_mode", "tpms")),
        "lattice_type": str(payload["lattice_type"]),
        "solid_side": str(payload["solid_side"]),
        "contact_distance_mm": float(payload.get("contact_distance_mm", 0.2)),
    }
    for key in ("casing_diameter_mm", "monolith_diameter_mm", "length_mm", "end_buffer_mm", "cell_mm", "flow_ml_min", "viscosity_m2_s", "fluid_density_kg_m3", "tracer_diffusivity_m2_s", "dt_s", "duration_s", "contact_distance_mm"):
        if not np.isfinite(float(p[key])) or float(p[key]) <= 0:
            raise ValueError(f"{key.replace('_', ' ')} must be a finite positive value.")
    if not np.isfinite(float(p["lattice_rotation_deg"])):
        raise ValueError("Lattice rotation must be a finite number of degrees.")
    if int(p["points_per_cell"]) < 4:
        raise ValueError("Grid points per cell must be at least 4.")
    if p["geometry_mode"] not in {"tpms", "empty_casing"}:
        raise ValueError("Choose TPMS monolith or empty casing geometry.")
    if p["geometry_mode"] == "tpms" and float(p["monolith_diameter_mm"]) > float(p["casing_diameter_mm"]):
        raise ValueError("The monolith outer diameter cannot exceed the casing inner diameter.")
    if p["geometry_mode"] == "tpms" and p["lattice_type"] not in fluid.TPMS_COEFFICIENTS:
        raise ValueError("Choose Gyroid, Schwarz D, or Schwarz P.")
    tracer_duration = float(payload["tracer_duration_s"])
    pulse_duration = float(payload["pulse_duration_s"])
    sample_interval = float(payload["sample_interval_s"])
    if not np.isfinite(tracer_duration) or tracer_duration <= 0:
        raise ValueError("Tracer observation duration must be positive.")
    if not np.isfinite(pulse_duration) or pulse_duration <= 0:
        raise ValueError("Tracer pulse duration must be positive.")
    if pulse_duration > tracer_duration:
        raise ValueError("Tracer pulse duration cannot exceed its observation duration.")
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("RTD sample interval must be positive.")
    return p


def _sample_structure(field: np.ndarray, lengths: tuple[float, float, float], max_points: int = 64) -> dict[str, list[float]]:
    indices = [
        np.unique(np.linspace(0, size - 1, min(size, max_points), dtype=int))
        for size in field.shape
    ]
    sampled = field[np.ix_(*indices)]
    axes = [
        np.linspace(0.0, length, size, dtype=float)[index]
        for length, size, index in zip(lengths, field.shape, indices)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return {
        "x": x.ravel().tolist(),
        "y": y.ravel().tolist(),
        "z": z.ravel().tolist(),
        "value": sampled.ravel().astype(float).tolist(),
    }


def _run_job(payload: dict[str, object]) -> None:
    global PARTIAL_RESULT
    try:
        p = _numeric_inputs(payload)
        _set_job(status="running", message="Building geometry and solving flow…")
        flow_result = _run_simulation(p)
        _set_job(message="Flow complete; running tracer pulse…")

        result: dict[str, object] = {
            "flow_summary": str(flow_result["summary"]),
            "structure": _sample_structure(
                np.asarray(flow_result["boundary_field"]),
                (
                    float(flow_result["length_x_mm"]),
                    float(p["casing_diameter_mm"]),
                    float(p["casing_diameter_mm"]),
                ),
            ),
            "rtd": None,
            "rtd_warning": None,
            "metrics": None,
            "tracer_field": None,
            "tracer_exposure_field": None,
            "tracer_space_time": None,
        }
        result["metrics"] = {
            "pressure_total_pa": f"{float(flow_result['pressure_drop_total_pa']):+.5g}",
            "pressure_core_pa": (
                None
                if flow_result["pressure_drop_core_pa"] is None
                else f"{float(flow_result['pressure_drop_core_pa']):+.5g}"
            ),
        }
        with STATE_LOCK:
            PARTIAL_RESULT = result

        velocity_np = np.asarray(flow_result["velocity"], dtype=float)
        if not np.all(np.isfinite(velocity_np)):
            _complete_without_rtd(
                result,
                "The flow solver produced non-finite velocities, so the tracer and RTD could not be computed. Check the flow settings and stability.",
            )
            return
        flow_error_pct = float(flow_result["outlet_flow_error_pct"])
        flow_balance_error_pct = float(flow_result["flow_balance_error_pct"])
        density_deviation_pct = float(flow_result["density_deviation_pct"])
        if (
            abs(flow_error_pct) > 10.0
            or abs(flow_balance_error_pct) > 10.0
            or density_deviation_pct > 10.0
        ):
            _complete_without_rtd(
                result,
                "The flow has not passed the reliability check. "
                f"The requested rate is {p['flow_ml_min']:.5g} mL/min; "
                f"the interior inlet and outlet planes measure "
                f"{flow_result['measured_inlet_flow_ml_min']:.5g} and "
                f"{flow_result['measured_outlet_flow_ml_min']:.5g} mL/min. "
                f"Outlet-vs-target error is {flow_error_pct:+.1f}%; "
                f"inlet-to-outlet imbalance is {flow_balance_error_pct:+.1f}%; "
                f"maximum density deviation from the reference is {density_deviation_pct:.2f}%. "
                "The tracer run was stopped so it does not produce a misleading RTD.",
            )
            return

        fluid_mask = ~np.asarray(flow_result["solid"], dtype=bool)
        labels6, _ = ndimage.label(
            fluid_mask,
            structure=ndimage.generate_binary_structure(3, 1),
        )
        inlet_labels6 = np.unique(labels6[0][fluid_mask[0]])
        outlet_labels6 = np.unique(labels6[-1][fluid_mask[-1]])
        connected6 = np.intersect1d(
            inlet_labels6[inlet_labels6 > 0],
            outlet_labels6[outlet_labels6 > 0],
        ).size > 0
        if not connected6:
            labels18, _ = ndimage.label(
                fluid_mask,
                structure=ndimage.generate_binary_structure(3, 2),
            )
            inlet_labels18 = np.unique(labels18[0][fluid_mask[0]])
            outlet_labels18 = np.unique(labels18[-1][fluid_mask[-1]])
            connected18 = np.intersect1d(
                inlet_labels18[inlet_labels18 > 0],
                outlet_labels18[outlet_labels18 > 0],
            ).size > 0
            if connected18:
                warning = (
                    "The voxelized fluid has only diagonal connections from inlet to outlet, "
                    "but the current scalar tracer solver transports across face-sharing cells only. "
                    "This grid can carry LBM flow while blocking tracer; increasing run time will not fix it."
                )
            else:
                warning = (
                    "The voxelized fluid geometry has no connected inlet-to-outlet path, "
                    "so this case cannot produce an RTD at this grid resolution."
                )
            _complete_without_rtd(result, warning)
            return

        dt_s = float(p["dt_s"])
        dx_m = float(flow_result["dx_mm"]) * 1.0e-3
        contact_distance_mm = float(p["contact_distance_mm"])
        if contact_distance_mm < float(flow_result["dx_mm"]):
            raise ValueError(
                "The near-wall band must be at least one grid spacing wide. "
                f"Use at least {float(flow_result['dx_mm']):g} mm for this grid."
            )
        fluid_mask = ~np.asarray(flow_result["solid"], dtype=bool)
        velocity_np = np.asarray(flow_result["velocity"], dtype=float)
        max_lattice_cfl = float(
            np.max(
                np.sum(np.abs(velocity_np), axis=0)[fluid_mask]
            )
        )
        # The steady velocity field can be subcycled for tracer transport.
        # Scale velocity and diffusivity together so each tracer step advances
        # a conservative CFL-limited physical interval instead of one tiny LBM
        # interval. This preserves seconds on the RTD axis while avoiding
        # millions of nearly identical advection steps.
        tracer_step_ratio = min(
            50.0,
            0.2 / max(max_lattice_cfl, 1.0e-12),
        )
        tracer_step_ratio = max(1.0, tracer_step_ratio)
        tracer_dt_s = dt_s * tracer_step_ratio
        tracer_velocity = velocity_np * tracer_step_ratio
        sample_interval = float(payload["sample_interval_s"])
        tracer_result = tracer.run_tracer_experiment(
            jnp.asarray(tracer_velocity, dtype=jnp.float32),
            jnp.asarray(flow_result["solid"], dtype=jnp.bool_),
            duration_seconds=float(payload["tracer_duration_s"]),
            pulse_duration_seconds=float(payload["pulse_duration_s"]),
            dt_seconds=tracer_dt_s,
            diffusivity=(
                float(p["tracer_diffusivity_m2_s"])
                * tracer_dt_s
                / dx_m**2
            ),
            contact_distance_cells=contact_distance_mm / float(flow_result["dx_mm"]),
            contact_solid=jnp.asarray(
                flow_result["contact_solid"], dtype=jnp.bool_
            ),
            sample_interval_seconds=sample_interval,
            print_summary=False,
        )
        axial_profiles = np.asarray(
            tracer_result.axial_mean_concentration,
            dtype=float,
        )
        profile_sample_interval_s = tracer_dt_s * tracer_result.sample_every
        profile_stride = max(1, int(round(0.5 / profile_sample_interval_s)))
        axial_profiles = axial_profiles[::profile_stride]
        profile_times_s = (
            np.arange(axial_profiles.shape[0], dtype=float)
            * profile_stride
            * profile_sample_interval_s
            + profile_sample_interval_s
        )
        result["tracer_space_time"] = {
            "time_s": profile_times_s.tolist(),
            "x_mm": np.linspace(
                0.0,
                float(flow_result["length_x_mm"]),
                axial_profiles.shape[1],
            ).tolist(),
            "concentration": axial_profiles.tolist(),
            "max_concentration": max(
                float(np.max(axial_profiles)),
                1.0e-12,
            ),
        }
        tracer_concentration = np.asarray(tracer_result.concentration, dtype=float)
        result["tracer_field"] = _sample_structure(
            tracer_concentration,
            (
                float(flow_result["length_x_mm"]),
                float(p["casing_diameter_mm"]),
                float(p["casing_diameter_mm"]),
            ),
        )
        result["tracer_exposure_field"] = _sample_structure(
            np.asarray(tracer_result.concentration_exposure, dtype=float),
            (
                float(flow_result["length_x_mm"]),
                float(p["casing_diameter_mm"]),
                float(p["casing_diameter_mm"]),
            ),
        )
        outlet_flux = np.asarray(tracer_result.outlet_mass_flux, dtype=float)
        if not np.all(np.isfinite(outlet_flux)):
            _complete_without_rtd(
                result,
                "The tracer produced non-finite outlet flux values. The flow or tracer solution is numerically unstable for these settings.",
            )
            return
        if not np.any(outlet_flux > 0.0):
            injected_mass = float(np.asarray(tracer_result.cumulative_injected_mass)[-1])
            outlet_mass = float(np.asarray(tracer_result.cumulative_outlet_mass)[-1])
            remaining_mass = float(np.asarray(tracer_result.tracer_mass)[-1])
            _complete_without_rtd(
                result,
                "No tracer reached the measurement plane. "
                f"At {float(payload['tracer_duration_s']):g} s, injected={injected_mass:.4g}, "
                f"exited={outlet_mass:.4g}, remaining in domain={remaining_mass:.4g}, "
                f"maximum concentration={float(np.max(tracer_concentration)):.4g}. "
                "Check the reported outlet flow and inspect the tracer overlay before extending the run.",
            )
            return
        stats = tracer.residence_time_statistics(
            outlet_flux,
            dt=tracer_dt_s,
            sample_every=tracer_result.sample_every,
        )
        injected_mass = float(np.asarray(tracer_result.cumulative_injected_mass)[-1])
        outlet_mass = float(np.asarray(tracer_result.cumulative_outlet_mass)[-1])
        recovery_pct = 100.0 * outlet_mass / injected_mass if injected_mass > 0.0 else 0.0
        peak_flux = float(np.max(outlet_flux))
        tail_sample_count = max(1, int(np.ceil(0.05 * outlet_flux.size)))
        tail_flux = float(np.mean(outlet_flux[-tail_sample_count:]))
        tail_pct = 100.0 * tail_flux / peak_flux if peak_flux > 0.0 else 0.0

        time_s = np.asarray(stats["time"], dtype=float)
        distribution = np.asarray(stats["distribution"], dtype=float)
        if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(distribution)):
            _complete_without_rtd(
                result,
                "The RTD calculation returned non-finite values. Check the flow and tracer settings.",
            )
            return
        result["rtd"] = {
            "time_s": time_s.tolist(),
            "E_per_s": distribution.tolist(),
        }
        completeness_warnings = []
        if recovery_pct < 95.0:
            completeness_warnings.append(
                f"Only {recovery_pct:.1f}% of injected tracer has exited by the end of the run."
            )
        if tail_pct > 1.0:
            completeness_warnings.append(
                f"The mean outlet signal in the final 5% of the run is {tail_pct:.1f}% of its peak, so the curve has not returned near baseline."
            )
        if completeness_warnings:
            result["rtd_warning"] = (
                "Provisional RTD; do not compare as a completed curve. "
                + " ".join(completeness_warnings)
            )
        result["metrics"].update({
            "mean_s": f"{tracer_result.mean_residence_time:.5g}",
            "std_s": f"{tracer_result.residence_time_standard_deviation:.5g}",
            "skewness": f"{tracer_result.residence_time_skewness:.5g}",
            "peak_s": f"{tracer_result.residence_time_peak:.5g}",
            "recovery_pct": f"{recovery_pct:.1f}",
            "tail_pct": f"{tail_pct:.1f}",
        })
        if p["geometry_mode"] == "tpms":
            result["metrics"].update({
                "contact_mean_s": f"{tracer_result.mean_exited_contact_time:.5g}",
                "contact_fraction_pct": f"{100.0 * tracer_result.contact_fraction:.1f}",
                "contact_cv": f"{tracer_result.contact_time_coefficient_of_variation:.4g}",
                "contact_band_mm": f"{contact_distance_mm:g}",
            })
        _set_job(status="done", message="Simulation and RTD complete", result=result)
        with STATE_LOCK:
            PARTIAL_RESULT = None
    except ValueError as exc:
        # Preserve the flow/geometry result when the observation window was
        # too short for a tracer breakthrough.
        with STATE_LOCK:
            partial_result = PARTIAL_RESULT
        if isinstance(partial_result, dict):
            partial_result["rtd_warning"] = str(exc)
            _set_job(status="done", message="Flow complete; RTD unavailable", result=partial_result)
            with STATE_LOCK:
                PARTIAL_RESULT = None
        else:
            traceback.print_exc()
            detail = str(exc).strip() or "No additional error message was provided. See the WSL terminal traceback."
            _set_job(status="error", message="Simulation failed", error=f"ValueError: {detail}")
    except Exception as exc:
        traceback.print_exc()
        _set_job(status="error", message="Simulation failed", error=f"{type(exc).__name__}: {exc}")


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print("[browser dashboard] " + (fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/status":
            with STATE_LOCK:
                body = json.dumps(_json_safe(JOB), allow_nan=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        global PARTIAL_RESULT
        if urlparse(self.path).path != "/run":
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            with STATE_LOCK:
                if JOB.get("status") == "running":
                    self._send(409, b'{"error":"A simulation is already running."}', "application/json; charset=utf-8")
                    return
                JOB.clear()
                JOB.update(status="running", message="Preparing simulation…")
                PARTIAL_RESULT = None
            threading.Thread(target=_run_job, args=(payload,), daemon=True).start()
            body = b'{"status":"started"}'
            self._send(202, body, "application/json; charset=utf-8")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, body, "application/json; charset=utf-8")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"TPMS results dashboard: http://localhost:{PORT}")
    print("Leave this terminal running while using the browser dashboard. Press Ctrl+C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
