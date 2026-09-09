"""Read retained physics contact samples and report frame-level safety rates."""
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comparison.json'
METHODS = ('ACT', 'PACT-Finetune', 'PACT-Frozen')
FORBIDDEN = ('hazard_bar', 'other_environment', 'clutter', 'mounted_fixture')


def main():
    source = json.loads(SOURCE.read_text())
    summaries = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for row in source['rows']:
        with h5py.File(Path(row['directory']) / 'telemetry.h5', 'r') as h:
            classes = json.loads(h['contacts'].attrs['class_names'])
            counts = h['contacts/class_entries'][()]
        active = counts > 0
        hazard = active[:, classes.index('hazard_bar')]
        clutter = active[:, classes.index('clutter')]
        forbidden = active[:, [classes.index(c) for c in FORBIDDEN]].any(axis=1)
        assert len(counts) == row['physics_samples'] == 29701
        assert int((hazard | clutter).sum()) == row['union_frames']
        assert {c: int(active[:, i].sum()) for i, c in enumerate(classes)} == row['contact_frames']
        assert bool(not forbidden.any()) == row['collision_free']
        metrics = dict(rollouts=1, physics_samples=len(counts),
                       collision_free_frames=int((~forbidden).sum()),
                       hazard_free_frames=int((~hazard).sum()),
                       hazard_clutter_free_frames=int((~(hazard | clutter)).sum()))
        for block in (str(row['training_seed']), 'pooled'):
            summaries[block][row['method']].update(metrics)
    for block, methods in summaries.items():
        for method, metrics in methods.items():
            expected = source['summaries'][block][method]
            assert metrics['physics_samples'] == expected['physics_samples']
            assert metrics['hazard_clutter_free_frames'] == expected['physics_samples'] - expected['union_frames']
            assert metrics['hazard_free_frames'] == expected['physics_samples'] - expected['contact_frames']['hazard_bar']
            for key in ('collision_free', 'hazard_free', 'hazard_clutter_free'):
                metrics[key + '_percent'] = 100 * metrics[key + '_frames'] / metrics['physics_samples']
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
                  source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  raw_trajectories_checked=len(source['rows']), forbidden_classes=FORBIDDEN,
                  definitions=dict(collision_free='No contact in any forbidden class; target/receptacle contacts excluded.',
                                   hazard_free='No hazard_bar contact; other classes may be present.',
                                   hazard_clutter_free='No hazard_bar or clutter contact; previous frame avoidance metric.',
                                   frame='Audited physics contact sample, nominal 2 ms; not a video frame.'),
                  summaries=summaries)
    (ROOT / 'root_review/frame_avoidance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Frame-level collision and hazard avoidance', '',
             'Recomputed from all 450 retained telemetry files. Each method has 50 rollouts and 1,485,050 physics contact samples per seed; pooled totals are 150 rollouts and 4,455,150 samples per method.', '',
             'Collision-free means no hazard-bar, clutter, other-environment or mounted-fixture contact in that sample. Intentional target/receptacle contact is excluded. Hazard-free only excludes hazard-bar contact. Hazard/clutter-free is the previous frame avoidance metric. These are sample-level rates, not collision-free task success or rendered video frames.', '',
             '| Seed | Method | Collision-free frames | Hazard-free frames | Hazard/clutter-free frames (previous avoidance) |',
             '|---|---|---:|---:|---:|']
    for block in ('3103', '3104', '3105', 'pooled'):
        for method in METHODS:
            s = summaries[block][method]
            values = [f"{s[key + '_frames']:,} ({s[key + '_percent']:.3f}%)"
                      for key in ('collision_free', 'hazard_free', 'hazard_clutter_free')]
            lines.append(f"| {block} | {method} | " + ' | '.join(values) + ' |')
    lines += ['', 'Fine-tuning has the highest pooled collision-free and hazard/clutter-free frame rates. Frozen PACT has the highest hazard-only-free rate. The stricter collision-free metric additionally counts ACT contacts with other environment objects and mounted fixtures; overlapping contacts are counted only once.', '',
              f"Source comparison SHA-256: `{result['source_sha256']}`.", '']
    (ROOT / 'root_review/FRAME_AVOIDANCE.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
