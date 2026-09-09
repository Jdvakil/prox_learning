"""V10.11c collection sampler, original chunk-100 inference, raw telemetry."""
from __future__ import annotations
import argparse
import collections
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from pact_place_v1011c_experiment import *
os.environ.update(environment())
import h5py
import numpy as np
import eval_pact_collision_row as legacy
import eval_pact_place_row as place
import eval_pact_place_v109_row as v109
import molmo_spaces.tasks.pact_place_contact_audit as audit_module
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.tasks.enclosure_reach import (
    PactPlaceCorridorTask, PactPlaceCorridorV1011C33PctTallerPrimitiveSampler)

SLOTS = tuple(collection.ACTIVE_CLUTTER_SLOTS)
CLASSES = tuple(audit_module.CONTACT_CLASSES)
SLOT_RE = re.compile(r'pact_clutter_(\d{2})(?:[/_]|$)')
ACTIVE_ROW = None
OUTPUT = None


def slots_in(value):
    return set(SLOT_RE.findall(str(value))) & set(SLOTS)


class V1011CContactAudit(audit_module.PactPlaceContactAudit):
    def reset(self):
        super().reset()
        self.raw_times = []
        self.raw_steps = []
        self.raw_classes = []
        self.raw_slots = []
        self.pair_lookup = {}
        self.pair_identities = []
        self.raw_pair_counts = []

    def observe(self, env, step):
        sim_time = round(float(env.current_data.time),12)
        if sim_time in self._seen_times:
            return
        original = audit_module.place_environment_contact_pairs

        def capture(environment):
            pairs = original(environment)
            by_class = dict.fromkeys(CLASSES,0)
            by_slot = dict.fromkeys(SLOTS,0)
            pair_counts = collections.Counter()
            for pair in pairs:
                category = audit_module.classify_contact(pair)
                by_class[category] += 1
                names = ' '.join(str(pair[k]) for k in ('geom1','geom2','body1','body2','root1','root2'))
                for slot in slots_in(names):
                    by_slot[slot] += 1
                identity = tuple(str(pair[k]) for k in ('geom1','geom2','body1','body2','root1','root2'))
                if identity not in self.pair_lookup:
                    self.pair_lookup[identity] = len(self.pair_identities)
                    self.pair_identities.append({'names':list(identity),'class':category,'slots':sorted(slots_in(names))})
                pair_counts[self.pair_lookup[identity]] += 1
            frame = len(self.raw_times)
            self.raw_pair_counts.extend((frame,pair_id,count) for pair_id,count in pair_counts.items())
            self.raw_times.append(sim_time)
            self.raw_steps.append(int(step))
            self.raw_classes.append([by_class[c] for c in CLASSES])
            self.raw_slots.append([by_slot[s] for s in SLOTS])
            return pairs

        audit_module.place_environment_contact_pairs = capture
        try:
            super().observe(env,step)
        finally:
            audit_module.place_environment_contact_pairs = original

    def summary(self):
        summary = super().summary()
        classes = np.asarray(self.raw_classes,dtype=np.int32)
        slots = np.asarray(self.raw_slots,dtype=np.int32)
        assert len(classes) == summary['sample_count']
        for i,c in enumerate(CLASSES):
            assert int(classes[:,i].sum()) == summary['contact_class_totals'][c]
            assert int((classes[:,i]>0).sum()) == summary['frames_with_contact'][c]
        summary['per_object'] = {s:{'contact_entries':int(slots[:,i].sum()),
            'contact_frames':int((slots[:,i]>0).sum())} for i,s in enumerate(SLOTS)}
        summary['raw_contact_telemetry'] = 'telemetry.h5:/contacts'
        return summary


class V1011CInferencePolicy(v109.PactPlaceV109InferencePolicy):
    def reset(self):
        super().reset()
        self._contact_audit = V1011CContactAudit()
        self.task._contact_audit_hook = self._contact_audit
        self._initial_written = False
        self._stability_times = set()
        self._stability_raw = []
        self._stability_baseline = None

    def inference_model(self, observation):
        if not self._initial_written:
            initial = OUTPUT / 'initial_observation.h5'
            with h5py.File(initial,'x') as handle:
                def write(prefix,value):
                    if isinstance(value,dict):
                        for key,item in sorted(value.items()):
                            write(prefix+'/'+str(key),item)
                    else:
                        array = np.asarray(value)
                        if array.dtype.kind in 'OU':
                            array = np.bytes_(json.dumps(value,sort_keys=True,default=str))
                        options = {'compression':'gzip'} if array.ndim and array.size else {}
                        handle.create_dataset(prefix,data=array,**options)
                write('observation',observation)
            self._initial_written = True
        return super().inference_model(observation)

    def _v109_update_clutter_stability(self):
        data = self.task.env.current_data
        timestamp = round(float(data.time),12)
        if timestamp in self._stability_times:
            return
        scene = self.task.scene_params
        baselines = scene['pact_clutter_settle']['objects']
        selected = {}
        for baseline in baselines:
            matched = slots_in(baseline['body'])
            if matched:
                assert len(matched) == 1
                slot = next(iter(matched))
                assert slot not in selected
                selected[slot] = baseline
        assert set(selected) == set(SLOTS), f'missing stability baselines: {set(SLOTS)-set(selected)}'
        self._stability_baseline = selected
        measures = []
        for slot in SLOTS:
            baseline = selected[slot]
            body = baseline['body']
            index = int(self.task.env.current_model.body(body).id)
            position = np.asarray(data.xpos[index],dtype=float)
            rotation = np.asarray(data.xmat[index],dtype=float).reshape(3,3)
            reference = np.asarray(baseline['position_m'],dtype=float)
            reference_rotation = np.asarray(baseline['xmat'],dtype=float).reshape(3,3)
            displacement = float(np.linalg.norm(position-reference))
            cosine = np.clip((np.trace(reference_rotation.T @ rotation)-1)/2,-1,1)
            angle = float(np.arccos(cosine))
            assert np.isfinite(displacement) and np.isfinite(angle)
            measures.append([displacement,angle,*position,*rotation.ravel()])
            if (displacement>0.02 or angle>np.deg2rad(25)) and body not in self._v109_stability_bodies:
                self._v109_stability_bodies.add(body)
                self._v109_stability_events.append({'step':int(self._step),'body':body,'slot':slot,
                    'classification':'other_environment','reason':'movable_clutter_toppled_or_displaced',
                    'displacement_m':displacement,'rotation_angle_rad':angle})
        self._stability_times.add(timestamp)
        self._stability_raw.append((timestamp,int(self._step),measures))

    def get_info(self):
        info = super().get_info()
        file = OUTPUT / 'telemetry.h5'
        if not file.exists():
            audit = self._contact_audit
            with h5py.File(file,'x') as h:
                h.attrs['schema_version'] = 'pact_v1011c_raw_telemetry_v1'
                h.attrs['sampler_class'] = collection.SAMPLER_CLASS
                h.attrs['environment_version'] = collection.ENVIRONMENT_VERSION
                h.attrs['row_sha256'] = ACTIVE_ROW['row_sha256']
                for name,value in (
                    ('contacts/sim_time_s',np.asarray(audit.raw_times,dtype=np.float64)),
                    ('contacts/control_step',np.asarray(audit.raw_steps,dtype=np.int32)),
                    ('contacts/class_entries',np.asarray(audit.raw_classes,dtype=np.int32)),
                    ('contacts/slot_entries',np.asarray(audit.raw_slots,dtype=np.int32)),
                    ('contacts/pair_counts',np.asarray(audit.raw_pair_counts,dtype=np.int32).reshape(-1,3)),
                    ('stability/sim_time_s',np.asarray([r[0] for r in self._stability_raw])),
                    ('stability/control_step',np.asarray([r[1] for r in self._stability_raw],dtype=np.int32)),
                    ('stability/measures',np.asarray([r[2] for r in self._stability_raw],dtype=np.float64))):
                    h.create_dataset(name,data=value,compression='gzip',compression_opts=4)
                h['contacts'].attrs['class_names'] = json.dumps(CLASSES)
                h['contacts'].attrs['slot_names'] = json.dumps(SLOTS)
                h['contacts'].attrs['pair_identities'] = json.dumps(audit.pair_identities,sort_keys=True)
                h['stability'].attrs['slot_names'] = json.dumps(SLOTS)
                h['stability'].attrs['measure_names'] = json.dumps(['displacement_m','rotation_angle_rad','x','y','z']+[f'rotation_{i}' for i in range(9)])
                h['stability'].attrs['baseline'] = json.dumps(self._stability_baseline,sort_keys=True)
        info.update({'sampler_class':collection.SAMPLER_CLASS,
            'sampler_module':'molmo_spaces.tasks.enclosure_reach','environment_version':collection.ENVIRONMENT_VERSION,
            'raw_telemetry_sha256':sha256_file(file),
            'initial_observation_sha256':sha256_file(OUTPUT / 'initial_observation.h5'),
            'native_thread_environment':{key:os.environ[key] for key in THREAD_ENV}})
        return info


class V1011CPolicyConfig(place.PactPlacePolicyConfig):
    policy_cls: type = V1011CInferencePolicy
    num_queries: int = 100


class V1011CEvalConfig(place._BaseConfig):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        learned = self.policy_config
        self.task_type = 'pick_and_place'
        self.task_horizon = 900
        self.end_on_success = False
        self.task_config = PickAndPlaceTaskConfig(task_cls=PactPlaceCorridorTask)
        self.policy_config = learned
        self.task_sampler_config.task_sampler_class = PactPlaceCorridorV1011C33PctTallerPrimitiveSampler
        self.task_sampler_config.scene_xml_paths = [v109._active_scene()]*2
        self.robot_config.action_noise_config.enabled = False


def sampler_for(row,default):
    assert row['task_sampler_class'] == collection.SAMPLER_CLASS
    return PactPlaceCorridorV1011C33PctTallerPrimitiveSampler


def manifest_all(path):
    doc = load_eval_manifest(path)
    return {**doc,'rows':doc['rows']+doc['smoke']['rows']}


def publish_h5(**kwargs):
    kwargs['save_videos'] = False
    full, videos = v109._ORIGINAL_PUBLISH_EPISODE(**kwargs)
    assert not videos
    file = Path(full)
    kwargs['result']['trajectory_retention'] = {'mode':'full_h5','full_h5_sha256':sha256_file(file),
        'all_proximity_frames_retained':True,'physics_contact_and_control_stability_retained':True}
    return str(file),[]


def main():
    global ACTIVE_ROW, OUTPUT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--episode-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--h5-only',action='store_true',required=True)
    known,remainder = parser.parse_known_args()
    # Consume only the storage switch; legacy owns the other CLI arguments.
    sys.argv.remove('--h5-only')
    rows = manifest_all(known.manifest)['rows']
    matches = [r for r in rows if r['episode_id']==known.episode_id]
    assert len(matches)==1
    ACTIVE_ROW = matches[0]
    OUTPUT = known.output_dir.resolve()
    assert OUTPUT.is_relative_to(EVAL.resolve())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    scene = ROOT / ACTIVE_ROW['pact_v1011_scene_relative']
    assert sha256_file(scene) == ACTIVE_ROW['pact_v106_scene_sha256']
    module_path = Path(inspect.getfile(PactPlaceCorridorV1011C33PctTallerPrimitiveSampler)).resolve()
    assert module_path == ROOT / 'submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py'
    source = read(WORK / 'source_manifest.json')
    assert sha256_file(module_path) == source['implementation_hashes'][str(module_path.relative_to(ROOT))]
    v109._ACTIVE_SCENE = str(scene)
    v109._ACTIVE_SCENE_SHA256 = ACTIVE_ROW['pact_v106_scene_sha256']
    v109._ACTIVE_OUTPUT_DIR = str(OUTPUT)
    v109.install_v109_bindings()
    place.PactPlaceInferencePolicy = V1011CInferencePolicy
    place.policy_config_factory = lambda **kw:V1011CPolicyConfig(num_queries=100,**kw)
    place.PactPlaceEvalConfig = V1011CEvalConfig
    place.task_sampler_class_for = sampler_for
    place.load_manifest = manifest_all
    place.retry_seed = retry_seed
    legacy._publish_episode = publish_h5
    sys.argv = [sys.argv[0],'--num-queries','100',*sys.argv[1:]]
    return place.main()


if __name__ == '__main__':
    raise SystemExit(main())
