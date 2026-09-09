"""Cross-model pairing: provenance is separate; all physical fields compare."""
from pathlib import Path
import h5py
import numpy as np
from pact_v1010b_contract import ROOT, B, read, freeze, sha, inside
from pact_wrist288_analysis import rgb_difference
from pact_v1010b_storage import names, same_value


def audit_initial_group(rows,destination):
    destination=inside(destination); assert len(rows)>=2
    reference=rows[0]; comparisons=[]
    for row in rows[1:]:
        for key in ('scene_id','physical_row_digest','task_seed'):
            assert reference[key]==row[key], f'physical identity differs: {key}'
        pa=Path(reference['directory'])/'initial_observation.h5'
        pb=Path(row['directory'])/'initial_observation.h5'
        detail={'reference':str(pa),'comparison':str(pb),'reference_sha256':sha(pa),'comparison_sha256':sha(pb),
            'exact_non_rgb':[],'violations':[],'rgb':None}
        with h5py.File(pa) as a,h5py.File(pb) as b:
            na,nb=names(a),names(b)
            if set(a.attrs)!=set(b.attrs) or any(not same_value(a.attrs[k],b.attrs[k]) for k in set(a.attrs)&set(b.attrs)):
                detail['violations'].append('root attributes')
            if na!=nb: detail['violations'].append('dataset/group names')
            for name in sorted(set(na)&set(nb)):
                av,bv=a[name],b[name]
                if set(av.attrs)!=set(bv.attrs) or any(not same_value(av.attrs[k],bv.attrs[k]) for k in set(av.attrs)&set(bv.attrs)):
                    detail['violations'].append('attributes/'+name)
                if not isinstance(av,h5py.Dataset): continue
                if av.shape!=bv.shape or av.dtype!=bv.dtype:
                    detail['violations'].append('shape/dtype/'+name);continue
                if name=='observation/wrist_camera':
                    result,delta=rgb_difference(av[()],bv[()]);detail['rgb']=result
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(destination.parent/(destination.stem+f'_{len(comparisons):02d}.npz'),
                        first=av[()],second=bv[()],absolute_difference=delta)
                    if not result['passed']: detail['violations'].append('RGB tolerance')
                elif not same_value(av[()],bv[()]): detail['violations'].append(name)
                else: detail['exact_non_rgb'].append(name)
        assert detail['rgb'] is not None and any(n.startswith('physics/') for n in detail['exact_non_rgb'])
        detail['passed']=not detail['violations'];comparisons.append(detail)
    doc={'schema':'pact_v1010b_grasp_v1','scene_id':reference['scene_id'],
        'provenance':[r.get('provenance',{}) for r in rows], 'comparisons':comparisons,
        'passed':all(c['passed'] for c in comparisons)}
    freeze(destination,doc)
    assert doc['passed'], f'initial group mismatch: {destination}'
    return doc
