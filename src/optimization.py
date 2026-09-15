from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.fluid import macroscopic_fields, run_steps
from src.model import CONFIG, grid as DEFAULT_GRID, init_fluid, init_geometry, warm_start_new_geometry
from src.tracer import run_tracer_experiment

DIRECTIONS_8 = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

@dataclass
class CFDResult:
    distributions: jax.Array
    solid: jax.Array
    velocity: jax.Array
    block_residuals: np.ndarray
    converged: np.ndarray

@dataclass
class ExperimentResult:
    cfd: CFDResult
    tracer: Any
    @property
    def score(self):
        return float(self.tracer.score)

def _geometry(params, grid):
    return init_geometry(params, grid, CONFIG['gyroid_period'], CONFIG['gyroid_level'], CONFIG['radius'], CONFIG['gyroid_solid_side'], CONFIG['inlet_buffer_cells'], CONFIG['outlet_buffer_cells'])

def _initial_state(params, grid, warm_start=None):
    solid, source_solid, wall_fraction, boundary_field, second_fluid_available = _geometry(params, grid)
    _, distributions, omega, force = init_fluid(solid, CONFIG['inlet_density'], CONFIG['inlet_velocity'], boundary_field, 500)
    if warm_start is not None:
        if 'solid' not in warm_start:
            raise ValueError("warm_start must contain both 'distributions' and 'solid'.")
        distributions = warm_start_new_geometry(
            warm_start['distributions'], warm_start['solid'], solid,
            force=force, inlet_density=CONFIG['inlet_density'], inlet_velocity=CONFIG['inlet_velocity'],
            new_boundary_field=boundary_field, use_heuristic_for_new_fluid=True)
    return distributions, solid, source_solid, wall_fraction, second_fluid_available, jnp.asarray(omega, jnp.float32), force

def _vmap_cfd_block(number_of_steps):
    # Only the expensive CFD time-marching block is vmapped.
    def one(f, solid, source_solid, wall_fraction, second_fluid_available, omega):
        return run_steps(f, solid, source_solid, wall_fraction, second_fluid_available, omega,
                         CONFIG['inlet_velocity'], CONFIG['inlet_density'], number_of_steps=number_of_steps)
    return jax.jit(jax.vmap(one, in_axes=(0,0,0,0,0,0)))

def run_cfd_batch(params_batch, *, warm_start=None, grid=DEFAULT_GRID, number_of_blocks=100,
                  steps_per_block=1000, residual_tolerance=1.1e-4, require_residual_increase=True):
    """Batched CFD. Geometry/warm-start preprocessing is not vmapped.

    Early stopping is checked per simulation after every block; the batch stops
    only when every simulation satisfies the convergence criterion.
    """
    params_batch = list(params_batch)
    if not params_batch:
        raise ValueError('params_batch cannot be empty')
    states = [_initial_state(p, grid, warm_start) for p in params_batch]
    f, solid, source_solid, wall_fraction, second_fluid_available, omega, force = map(jnp.stack, zip(*states))
    interior = ~solid
    interior = interior.at[:, :CONFIG['inlet_buffer_cells']].set(False)
    interior = interior.at[:, -CONFIG['outlet_buffer_cells']:].set(False)
    previous_residual = jnp.full((len(states),), jnp.inf, dtype=jnp.float32)
    converged = jnp.zeros((len(states),), dtype=jnp.bool_)
    history = []
    cfd_block = _vmap_cfd_block(steps_per_block)
    for _ in range(number_of_blocks):
        # run_steps donates its input; preserve the start-of-block state for residuals.
        old_f = jnp.array(f, copy=True)
        f = cfd_block(f, solid, source_solid, wall_fraction, second_fluid_available, omega)
        f.block_until_ready()
        diff = f - old_f
        num = jnp.sum(jnp.where(interior[:,None,...], diff**2, 0.0), axis=(1,2,3,4))
        den = jnp.sum(jnp.where(interior[:,None,...], old_f**2, 0.0), axis=(1,2,3,4))
        residual = jnp.sqrt(num / jnp.maximum(den, 1e-12))
        history.append(np.asarray(jax.device_get(residual)))
        small = residual < residual_tolerance
        converged = small & (residual > previous_residual) if require_residual_increase else small
        if bool(jax.device_get(jnp.all(converged))):
            break
        previous_residual = residual
    _, velocity = jax.vmap(macroscopic_fields)(f, force, solid)
    return CFDResult(f, solid, velocity, np.asarray(history), np.asarray(jax.device_get(converged)))

def run_experiment(params, *, warm_start=None, grid=DEFAULT_GRID, tracer_steps=100_000,
                   pulse_duration=200, diffusivity=0.01, dt=1.0, contact_distance_cells=2.0,
                   score_penalty_weight=0.5, **cfd_kwargs):
    """Run CFD and then the tracer experiment for one geometry."""
    batch = run_cfd_batch([params], warm_start=warm_start, grid=grid, **cfd_kwargs)
    measurement_x = grid.nx - CONFIG['outlet_buffer_cells'] - 2
    tracer = run_tracer_experiment(batch.velocity[0], batch.solid[0], number_of_steps=tracer_steps,
        pulse_duration=pulse_duration, inlet_concentration=1.0, diffusivity=diffusivity, dt=dt,
        measurement_x=measurement_x, contact_distance_cells=contact_distance_cells,
        score_penalty_weight=score_penalty_weight)
    cfd = CFDResult(batch.distributions[0], batch.solid[0], batch.velocity[0], batch.block_residuals[:,0], batch.converged[0])
    return ExperimentResult(cfd, tracer)

def _simplex_params(a, b):
    return {'A': float(a), 'B': float(b), 'C': float(1.0-a-b)}

def grid_search(start_params, n_steps=10, grid=DEFAULT_GRID, step_size=0.1, **kwargs):
    """Greedy cached 8-neighbour search on the non-negative A+B+C=1 simplex."""
    total = float(sum(start_params[k] for k in ('A','B','C')))
    if total <= 0:
        raise ValueError('A+B+C must be positive')
    start = {k: float(start_params[k])/total for k in ('A','B','C')}
    origin_a, origin_b = start['A'], start['B']
    key = (0,0)
    current = run_experiment(start, grid=grid, **kwargs)
    cache = {key: current}
    params_cache = {key: start}
    history = [(start, current.score)]
    cfd_keys = {'number_of_blocks','steps_per_block','residual_tolerance','require_residual_increase'}
    for _ in range(n_steps):
        neighbours = [(key[0]+di, key[1]+dj) for di,dj in DIRECTIONS_8]
        valid = []
        for candidate in neighbours:
            a = origin_a + candidate[0]*step_size
            b = origin_b + candidate[1]*step_size
            c = 1.0-a-b
            if min(a,b,c) >= -1e-12:
                valid.append(candidate)
                params_cache.setdefault(candidate, _simplex_params(max(a,0.0), max(b,0.0)))
        unseen = [candidate for candidate in valid if candidate not in cache]
        if unseen:
            warm = {'distributions': current.cfd.distributions, 'solid': current.cfd.solid}
            batch = run_cfd_batch([params_cache[c] for c in unseen], warm_start=warm, grid=grid,
                                  **{k:v for k,v in kwargs.items() if k in cfd_keys})
            measurement_x = grid.nx - CONFIG['outlet_buffer_cells'] - 2
            for i, candidate in enumerate(unseen):
                tracer = run_tracer_experiment(batch.velocity[i], batch.solid[i],
                    number_of_steps=kwargs.get('tracer_steps',100_000), pulse_duration=kwargs.get('pulse_duration',200),
                    inlet_concentration=1.0, diffusivity=kwargs.get('diffusivity',0.01), dt=kwargs.get('dt',1.0),
                    measurement_x=measurement_x, contact_distance_cells=kwargs.get('contact_distance_cells',2.0),
                    score_penalty_weight=kwargs.get('score_penalty_weight',0.5), print_summary=False)
                cfd = CFDResult(batch.distributions[i], batch.solid[i], batch.velocity[i], batch.block_residuals[:,i], batch.converged[i])
                cache[candidate] = ExperimentResult(cfd, tracer)
        if not valid:
            break
        best = max(valid, key=lambda candidate: cache[candidate].score)
        if cache[best].score <= current.score:
            break
        key, current = best, cache[best]
        history.append((params_cache[key], current.score))
    return {'params': params_cache[key], 'score': current.score, 'result': current, 'history': history, 'cache': cache}
