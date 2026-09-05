"""Single-episode scoring and scheduling, independent of MuJoCo imports."""
from __future__ import annotations
import time
import numpy as np


def one_bool(value):
    values = np.asarray(value)
    if values.size != 1:
        raise ValueError('This evaluator supports exactly one environment per worker')
    return bool(values.reshape(-1)[0])


def rollout(task, policy, horizon, *, trace=None):
    """Score success-ever; retain full-horizon safety and final-state success.

    Exceptions propagate to the outer runner, which marks the planned suite
    incomplete. An exception or missing action is never silently removed.
    """
    if horizon <= 0:
        raise ValueError('horizon must be positive')
    policy.reset()
    observation, _ = task.reset()
    if one_bool(task.judge_success()):
        raise ValueError('Evaluation scene already satisfies success at reset')
    ever = terminal_success = False
    first_success = None
    policy_seconds = task_seconds = 0.0
    started = time.perf_counter()
    for step in range(horizon):
        tick = time.perf_counter()
        action = policy.get_action(observation)
        policy_seconds += time.perf_counter() - tick
        if action is None or not all(np.isfinite(np.asarray(v)).all() for v in action.values()):
            raise ValueError(f'Invalid policy action at step {step}')
        tick = time.perf_counter()
        observation, _, terminal, truncated, _ = task.step(action)
        task_seconds += time.perf_counter() - tick
        terminal_success = one_bool(task.success_cache[-1])
        if terminal_success and first_success is None:
            first_success = step + 1
        ever |= terminal_success
        if trace is not None:
            trace(step, action, observation, terminal_success)
        # Keep numeric bookkeeping, but not hundreds of discarded image frames.
        if task.observation_cache:
            task.observation_cache[-1] = [{}]
        if one_bool(terminal) or one_bool(truncated):
            if step + 1 < horizon:
                raise ValueError(f'Unexpected termination at {step + 1}/{horizon}; full-horizon safety incomplete')
            break
    return {'status': 'complete', 'success': ever, 'terminal_success': terminal_success,
            'first_success_step': first_success, 'steps': step + 1,
            'policy_seconds': policy_seconds, 'task_seconds': task_seconds,
            'rollout_wall_seconds': time.perf_counter() - started}


def summarize(records, planned):
    if planned <= 0 or len(records) > planned:
        raise ValueError('Invalid evaluation denominator')
    completed = [r for r in records if r.get('status') == 'complete']
    complete = len(completed) == planned
    return {'complete': complete, 'planned_episodes': planned, 'completed_episodes': len(completed),
            'successes': sum(bool(r['success']) for r in completed),
            'success_rate': sum(bool(r['success']) for r in completed) / planned if complete else None,
            'collision_free_rate': sum(bool(r['collision_free']) for r in completed) / planned if complete else None,
            'strict_success_rate': sum(bool(r['success'] and r['collision_free']) for r in completed) / planned if complete else None}
