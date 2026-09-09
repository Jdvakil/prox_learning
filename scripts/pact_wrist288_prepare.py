"""Freeze streams, source bindings and the outcome-independent final manifest."""
from __future__ import annotations
import re
import subprocess
import xml.etree.ElementTree as ET
from pact_wrist288_common import *

def historical_seeds():
    pattern=r'"(?:task_seed(?:_u32|_u64)?|seed_u32|seed_u64|episode_seed|seed)"\s*:\s*[0-9]+'
    command=['rg','--no-ignore','--no-heading','--with-filename','-o',pattern,
        '-g','*.json','-g','*.jsonl','-g','!pact_place_v1010_wrist288_s3_v1/**',
        'diagnostics_output','assets','configs']
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    assert result.returncode in (0,1),result.stderr
    seeds=set(); sources={}
    for line in result.stdout.splitlines():
        path,match=line.split(':',1)
        if str(WORK.relative_to(ROOT)) in path: continue
        value=int(re.search(r':\s*([0-9]+)$',match)[1])
        seeds.add(value % 2**32); sources[path]=sources.get(path,0)+1
    assert len(seeds)>1000, f'historical inventory unexpectedly small: {len(seeds)}'
    return {'seed_u32_values':sorted(seeds),'source_match_counts':sources,'search_command':command,
        'scope':'All JSON/JSONL in workspace diagnostics_output, assets and configs; task/episode/selected numeric seed fields.'}

def allocate(inventory):
    used=set(inventory['seed_u32_values']); entries={}; collisions=[]
    keys=[]
    for cell in map(lambda c:cell_key(*c),cells()):
        for ordinal in range(128): keys.append(('collection',cell,ordinal))
        keys.extend((role,cell,0) for role in ('preflight','development','split'))
    for i,c in enumerate(cells()[:4]): keys.append(('smoke',cell_key(*c),i))
    extras=[cells()[0],cells()[4],cells()[8],cells()[9],cells()[13],cells()[17]]
    assert len(set(extras))==6
    assert sorted(c[1] for c in extras)==['left']*3+['right']*3
    assert sorted(c[2] for c in extras)==['center']*2+['neg5']*2+['pos5']*2
    for si,seed in enumerate(SEEDS):
        for c in cells():
            keys.extend((f'final_{seed}',cell_key(*c),i) for i in (0,1))
        keys.extend((f'final_{seed}',cell_key(*c),2) for c in extras[2*si:2*si+2])
    for role,cell,ordinal in keys:
        values=[]
        for retry in range(13):
            collision=0
            while True:
                value=derive(role if retry==0 else 'retry/'+role,cell,ordinal,retry,collision)
                if value['seed_u32'] not in used: break
                collisions.append(value);collision+=1
            used.add(value['seed_u32']);values.append(value)
        entries[f'{role}:{cell}:{ordinal}']=values
    return {'entries':entries,'collisions_rejected':collisions,'extras':[cell_key(*c) for c in extras]}

def build_row(role,cell,ordinal,registry=None):
    import pact_place_v1010_contract as historical
    registry=registry or read(WORK/'seed_registry.json')
    family,side,pose=cell.split('|')
    row=historical.build_row(family,side,pose,ordinal)
    seeds=registry['entries'][f'{role}:{cell}:{ordinal}']
    identity=digest([NAMESPACE,role,cell,ordinal])
    row.update(attempt_id=identity,episode_id=identity,seed_stream=NAMESPACE+'/'+role,
        experiment_namespace=NAMESPACE,stream_role=role,task_seed_u32=seeds[0]['seed_u32'],
        task_seed_u64=seeds[0]['seed_u64'],sampling_seeds=seeds,role_index=ordinal,
        candidate_index=ordinal,schema_version='pact_wrist288_instance_v1',role=role)
    row['row_sha256']=digest({k:v for k,v in row.items() if k!='row_sha256'})
    return row

def input_hashes():
    paths=set()
    for base in (ROOT/'scripts',ROOT/'submodules/act',ROOT/'submodules/molmospaces/molmo_spaces'):
        paths.update(p for p in base.rglob('*.py') if '__pycache__' not in p.parts)
    paths.add(ROOT/'docs/PACT_PLACE_V1010_WRIST288_THREE_SEED_PLAN.md')
    paths.add(ROOT/'tests/test_pact_wrist288.py')
    # Bind XML and all on-disk assets referenced by the certified scenes,
    # including recursively included robot meshes and textures.
    def xml_inputs(path):
        path=Path(os.path.abspath(path))
        if path in paths: return
        assert path.is_relative_to(ROOT),path
        paths.add(path)
        tree=ET.parse(path)
        compiler=tree.getroot().find('compiler')
        for node in tree.iter():
            file=node.get('file')
            if not file: continue
            candidates=[path.parent/file,ROOT/'assets'/file]
            if compiler is not None:
                for name in ('meshdir','texturedir','assetdir'):
                    if compiler.get(name): candidates.append(path.parent/compiler.get(name)/file)
            candidates=[Path(os.path.abspath(p)) for p in candidates]
            found=next((p for p in candidates if p.is_file()),None)
            if found is None: raise RuntimeError(f'unresolved XML asset: {path}: {file}')
            if found.suffix=='.xml': xml_inputs(found)
            else: paths.add(found)
    for entry in SCENE_BY_POSE.values():
        assert sha(ROOT/entry['relative'])==entry['sha256']
        xml_inputs(ROOT/entry['relative'])
    # Clutter meshes are loaded dynamically from the V9.5 frozen palette.
    from pact_place_v1010_contract import v95_row_payload
    from pact_place_v9_contract import PALETTE_PATH
    from molmo_spaces.utils.lazy_loading_utils import install_uid
    paths.add(PALETTE_PATH)
    payload=v95_row_payload(cells()[0][0],cells()[0][1])
    palette_uids={str(o['uid']) for o in payload['pact_clutter_palette']}|{'Cup_10'}
    for uid in sorted(palette_uids):xml_inputs(Path(install_uid(uid)))
    xml_inputs(ROOT/'assets/robots/franka_skin/model_hybrid.xml')
    paths.update(p for p in (ROOT/'assets/robots/franka_skin').rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def main():
    os.environ.update(environment())
    if (WORK/'config.json').exists():
        check_bindings(); print('Existing freeze verified'); return
    molmo=ROOT/'submodules/molmospaces'
    head=subprocess.check_output(['git','-C',str(molmo),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(molmo),'branch','--show-current'],text=True).strip()
    assert head=='70dedc07f34ed7f8335aed7f694ddef7ef823d3d'
    assert branch=='experiment/pact-vs-act-remediation-v2'
    assert not subprocess.check_output(['git','-C',str(molmo),'status','--porcelain'],text=True).strip()
    assert sha(ENCODER_PATH)==ENCODER_SHA256
    inventory=historical_seeds(); freeze(WORK/'historical_seeds.json',inventory)
    registry=allocate(inventory); freeze(WORK/'seed_registry.json',registry)
    manifests={}
    for role in ('preflight','smoke','development',*(f'final_{s}' for s in SEEDS)):
        rows=[]
        for key in registry['entries']:
            stream,cell,ordinal=key.split(':')
            if stream==role: rows.append(build_row(role,cell,int(ordinal),registry))
        doc={**empty_authorization(),'namespace':NAMESPACE,'role':role,'sensor_names':list(CANONICAL_SENSOR_NAMES),
            'rows':rows,'smoke':{'rows':[]}}
        doc['manifest_sha256']=digest(doc)
        freeze(WORK/f'manifests/{role}.json',doc)
        manifests[role]={'path':f'manifests/{role}.json','sha256':sha(WORK/f'manifests/{role}.json'),'count':len(rows)}
    config={**empty_authorization(),'namespace':NAMESPACE,'is_phase0_pass':False,
        'molmospaces_head':head,'molmospaces_branch':branch,'sampler_class':SAMPLER_CLASS,
        'active_clutter_slots':list(ACTIVE_CLUTTER_SLOTS),'scene_by_pose':SCENE_BY_POSE,
        'sensor_names':list(CANONICAL_SENSOR_NAMES),'sensor_order_sha256':SENSOR_ORDER_SHA256,
        'encoder_path':str(ENCODER_PATH),'encoder_sha256':ENCODER_SHA256,
        'file_hashes':input_hashes(),'historical_seed_inventory_sha256':sha(WORK/'historical_seeds.json'),
        'seed_registry_sha256':sha(WORK/'seed_registry.json'),'manifests':manifests,
        'collection':{'episodes':288,'per_cell':12,'max_inflight_per_cell':1,'max_attempts_per_cell_before_pause':128},
        'split':{'train':240,'validation':48,'validation_per_cell':2,'ranking':'SHA256(namespace, split seed derivation, accepted episode identity)'},
        'supervision':'obs[t] -> commanded_action[t+1]; require genuine move command and nonempty joint_pos at action row; no fixed tail count',
        'training':{'seeds':list(SEEDS),'updates':60000,'batch_size':8,'epochs':2000,'lr':1e-5,'lr_backbone':1e-5,
            'chunk':100,'kl':10,'hidden':512,'feedforward':3200,'enc_layers':7,'dec_layers':7,'heads':8,
            'rgb_shape':[240,320,3],'camera':'wrist_camera','state_dim':9,'action_dim':8,'loader_workers':4,
            'resume_every_updates':3000,'restart_at_updates':30000,'padding':'max(635, converted_T_max + 8)'},
        'development':{'histories':[100,10],'pairs':24,'selection':['combined_success','combined_collision_free_success','negative_combined_hazard_frames','prefer_100'],
            'gate_pact_successes':13,'gate_pact_advantage':3},
        'evaluation':{'horizon':900,'query_every_step':True,'weight':'exp(-0.01 * age)','gripper_threshold':127.5,
            'end_on_success':False,'action_noise':False,'smoke_pairs':4,'final_pairs_per_seed':50,
            'rgb_max_abs_delta':2,'rgb_max_changed_fraction':0.001,'non_rgb':'exact shapes, dtypes, bytes and configuration'},
        'resources':{'rollout_workers':[14,12,10],'vram_max':0.85,'ram_max':0.80,'pid_max':0.75,'disk_reserve_gib':10,
            'pressure_samples':3,'sample_seconds':60,'trainer_concurrency_min_gain':1.20},
        'thread_environment':THREAD_ENV,'safe_stop_utc':'2026-09-07T23:00:00+00:00','deadline_utc':'2026-09-08T00:00:00+00:00',
        'output_root':str(WORK)}
    config['config_sha256']=digest(config);freeze(WORK/'config.json',config)
    print(json.dumps({'config_sha256':config['config_sha256'],'historical_seeds':len(inventory['seed_u32_values']),
        'new_seeds':sum(map(len,registry['entries'].values())),'input_files':len(config['file_hashes']),'manifests':manifests},indent=2))

if __name__=='__main__': main()
