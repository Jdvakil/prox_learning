"""Read-only reconstruction of retained manipulation-result counts, never launches rollouts."""
import collections
import json
import mmap
import os
from pathlib import Path
import re
import argparse

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import h5py

ROOT = Path(__file__).resolve().parents[1]
TOP_FIELDS = ('arm', 'checkpoint_seed', 'task_success', 'collision_free_task_success',
              'episode_id', 'status', 'rollout_id', 'blur_sigma', 'blind_rgb')
SCALAR_PATTERN = re.compile(rb'^  "(' + b'|'.join(k.encode() for k in TOP_FIELDS) + rb')": ([^\n]+)$', re.MULTILINE)


def result_fields(path):
    # Historical results embed hundreds of MB of contact frames. Extract only
    # unique, top-level scalar JSON lines from their verified pretty-print schema.
    # Every extracted value is JSON-decoded; missing or duplicate fields fail.
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        found = collections.defaultdict(list)
        for key, value in SCALAR_PATTERN.findall(data):
            found[key.decode()].append(value)
        row = {}
        for key in TOP_FIELDS:
            values = found[key]
            if key in ('blur_sigma', 'blind_rgb') and not values:
                # Older schemas did not record these later ablation settings.
                # Keep them explicitly unknown; never substitute a numeric zero.
                row[key] = None
                continue
            assert len(values) == 1, (path, key, len(values))
            row[key] = json.loads(values[0].rstrip(b','))
        totals = re.findall(rb'^    "contact_class_totals": (\{.*?^    \})', data, re.MULTILINE | re.DOTALL)
        assert len(totals) == 1, (path, 'contact_class_totals', len(totals))
        row['contact_totals'] = json.loads(totals[0])
    assert row['status'] == 'complete'
    assert type(row['task_success']) is bool and type(row['collision_free_task_success']) is bool
    classes = set(row['contact_totals'])
    assert {'hazard_bar', 'other_environment', 'grasp_target'} <= classes, (path, classes)
    assert classes <= {'hazard_bar', 'other_environment', 'grasp_target', 'clutter', 'mounted_fixture', 'place_receptacle'}, classes
    disallowed = classes - {'grasp_target', 'place_receptacle'}
    row['collision_free'] = all(row['contact_totals'][key] == 0 for key in disallowed)
    assert row['collision_free_task_success'] == (row['task_success'] and row['collision_free'])
    row['hazard_episode'] = row['contact_totals']['hazard_bar'] > 0
    row['h5_success_verified'] = False
    trajectory = path.parent / 'trajectory.h5'
    if trajectory.exists():
        with h5py.File(trajectory, 'r') as handle:
            assert 'traj_0/success' in handle
            assert bool(handle['traj_0/success'][-1]) == row['task_success'], path
        row['h5_success_verified'] = True
    return row


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['checkpoint_seed'], row['blur_sigma'], row['blind_rgb'], row['arm'])].append(row)
    result = []
    for (seed, sigma, blind, arm), values in sorted(groups.items()):
        ids = [r['episode_id'] for r in values]
        assert len(set(ids)) == len(ids)
        result.append({'seed': seed, 'sigma': sigma, 'blind': blind, 'arm': arm, 'n': len(values),
            **{key: sum(int(r[key]) for r in values) for key in
               ('task_success', 'collision_free_task_success', 'collision_free', 'hazard_episode', 'h5_success_verified')}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.output_root:
        assert args.output_root.resolve().is_relative_to(ROOT / 'diagnostics_output/manipulation_eval_inventory_20260906')
        args.output_root.mkdir(parents=True, exist_ok=True)
    studies = {
        'frontend_screen': Path('/root/pact_frontend_screen_artifacts/evaluation_621764f8'),
        'seed_replication_new_seed': Path('/root/pact_seed_replication_artifacts/evaluation_1490160c'),
        'contact_endpoint': Path('/root/pact_contact_endpoint_artifacts/evaluation_v1'),
        'blur_sweep': Path('/root/pact_blur_sweep_artifacts/evaluation_v1'),
        'blind_rgb': Path('/root/pact_blind_rgb_artifacts/evaluation_v1'),
        'place_chunk1': Path('/root/pact_place_chunk1_eval_seed3101'),
        'place_chunk25': Path('/root/pact_place_chunk25_eval_seed3101'),
        'place_chunk100': Path('/root/pact_place_chunk100_eval_seed3101'),
        'place_v109': ROOT / 'diagnostics_output/pact_place_v109_eval',
        'place_v1010': ROOT / 'diagnostics_output/pact_place_v1010_eval_infra_repair_01',
    }
    for study, directory in studies.items():
        rows, errors = [], []
        receipts = {}
        if study in ('place_v109', 'place_v1010'):
            full = json.loads((directory / 'full_run.json').read_text())
            receipts = {str(Path(r['row_dir']) / 'result.json'): r for r in full['results']}
            paths = [Path(p) for p in receipts]
        else:
            paths = sorted(directory.glob('rows/*/result.json')) if (directory / 'rows').exists() else \
                sorted(directory.glob('act/*/result.json')) + sorted(directory.glob('pact/*/result.json'))
        for path in paths:
            if '_pact_zero' in path.parent.name or '_pact_permuted' in path.parent.name:
                continue
            try:
                driver_path = path.parent / 'driver_result.json'
                driver = receipts[str(path)] if str(path) in receipts else json.loads(driver_path.read_text())
                assert driver['returncode'] == 0 and driver['status'] == 'complete', driver_path
                row = result_fields(path)
                if row['arm'] not in ('ACT', 'PACT'):
                    continue
                assert row['rollout_id'] == driver['rollout_id']
                rows.append(row)
            except Exception as error:
                errors.append({'path': str(path), 'error': repr(error)})
        groups = summarize(rows)
        report = {'study': study, 'root': str(directory), 'result_files_found': len(paths),
            'verified_ACT_PACT_rows': len(rows), 'groups': groups, 'errors': errors,
            'verification_scope': 'Per-rollout result fields and actual-exit receipts; collision-free success reconstructed from contact totals. Final H5 success rechecked where trajectory remains. No missing trajectory is claimed as verified.'}
        if args.output_root:
            with (args.output_root / (study + '.json')).open('x') as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write('\n')
        print(json.dumps({**report, 'error_count': len(errors), 'errors': errors[:3]}), flush=True)


if __name__ == '__main__':
    main()
