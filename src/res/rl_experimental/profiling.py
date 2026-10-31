"""Measured environment/inference/training costs; no GPU performance assumptions."""
from __future__ import annotations

import cProfile
import pstats
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .data import PanelData, chronological_split
from .env import EpisodeRange, PortfolioEnv
from .portfolio import PortfolioConstraints
from .training import ExperimentConfig, build_candidate_cache, train_experiment
from .walk_forward import write_json


def profile_env(panel: PanelData, output: Path, steps=512, train_steps=512):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if steps <= 0 or train_steps < 0:
        raise ValueError("steps must be positive and train_steps nonnegative")
    count = min(int(steps), panel.n_steps)
    constraints = PortfolioConstraints()
    start = time.perf_counter()
    cache = build_candidate_cache(panel, constraints.candidate_count)
    result = {'steps': count, 'candidate_cache_seconds': time.perf_counter()-start,
              'panel_array_bytes': sum(v.nbytes for v in vars(panel).values() if isinstance(v, np.ndarray)),
              'environment': {}, 'training': {},
              'memory_note': 'tracemalloc is Python/NumPy-tracked peak; RSS is cumulative process peak, platform units'}
    reference = None
    for optimized in (False, True):
        samples, trajectories = [], []
        for repeat in range(3):
            env = PortfolioEnv(panel, EpisodeRange(0, count), constraints, observation_version=2,
                               record_history=not optimized, cache_observation=optimized,
                               candidate_cache=cache if optimized else None)
            env.reset(seed=7)
            rng = np.random.default_rng(7)
            navs = []
            started = time.perf_counter()
            for _ in range(count):
                action = rng.normal(0, .1, env.action_space.shape).astype(np.float32)
                _, _, done, _, info = env.step(action)
                navs.append((info['nav'], info['turnover'], info['fee_estimate']))
                if done:
                    break
            samples.append(time.perf_counter()-started)
            trajectories.append(navs)
            env.close()
        if reference is None:
            reference = trajectories[0]
        np.testing.assert_allclose(reference, trajectories[0], rtol=1e-7, atol=1e-9)
        # Profile separately so instrumentation does not contaminate timing comparison.
        env = PortfolioEnv(panel, EpisodeRange(0, count), constraints, observation_version=2,
                           record_history=not optimized, cache_observation=optimized,
                           candidate_cache=cache if optimized else None)
        env.reset(seed=7)
        profiler = cProfile.Profile()
        tracemalloc.start()
        profiler.enable()
        for _ in range(count):
            action_shape = env.action_space.shape
            if action_shape is None:
                raise ValueError('continuous action shape is required')
            _, _, done, _, _ = env.step(np.zeros(action_shape, np.float32))
            if done:
                break
        profiler.disable()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        name = 'optimized' if optimized else 'reference'
        profiler.dump_stats(str(output/f'{name}.prof'))
        stats = pstats.Stats(profiler)
        functions = {f'{Path(file).name}:{line}:{fun}': {'calls': data[1], 'self_seconds': data[2], 'cumulative_seconds': data[3]}
                     for (file, line, fun), data in getattr(stats, 'stats').items()
                     if fun in ('_observation', 'select_slots', 'execute_open_close', 'project_action', '__call__', '_base_info')}
        result['environment'][name] = {'median_seconds': float(np.median(samples)), 'seconds': samples,
                                       'steps_per_second': len(reference)/float(np.median(samples)),
                                       'python_peak_bytes': peak, 'functions': functions}
        env.close()
    result['environment']['trajectory_equivalent'] = True
    result['environment']['speedup'] = result['environment']['reference']['median_seconds']/result['environment']['optimized']['median_seconds']
    if train_steps:
        for device in ('cpu', 'cuda'):
            if device == 'cuda' and not torch.cuda.is_available():
                result['training'][device] = {'status': 'unavailable_on_this_host'}
                continue
            cfg = ExperimentConfig(device=device, total_timesteps=train_steps, observation_version=2,
                                   lean_training=True, eval_freq=2560)
            metrics = train_experiment(panel, output/device, cfg, splits=chronological_split(panel.n_steps), evaluate_test=False)
            result['training'][device] = metrics['timing']
            from stable_baselines3 import PPO
            model = PPO.load(output/device/'best_model.zip', device=device)
            env = PortfolioEnv(panel, EpisodeRange(0, count), observation_version=2)
            obs, _ = env.reset()
            inference = {}
            for batch in (1, 4):
                values = {k: np.stack([v]*batch) for k, v in obs.items()}
                for _ in range(3):
                    model.predict(values, deterministic=True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(100):
                    model.predict(values, deterministic=True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                inference[str(batch)] = (time.perf_counter()-started)/100
            result['training'][device]['inference_batch_seconds'] = inference
            env.close()
    result['process_peak_rss_native_units'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(output/'profile.json', result)
    return result
