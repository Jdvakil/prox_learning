"""Measure lossless storage on six outcome-independent retained examples."""
from audit import *

def main():
    rows=read(OUT/'rollout_audit.json')['rows']; reports=[]
    for seed in (3103,3104,3105):
        for arm in ('ACT','PACT'):
            r=min((r for r in rows if r['seed']==seed and r['arm']==arm),key=lambda r:hashlib.sha256(('v1010b/storage/'+r['episode_id']).encode()).hexdigest())
            source=ROOT/r['directory']/'trajectory.h5'; destination=OUT/'storage_probe_temporary.h5'
            start=time.time(); count=0
            with h5py.File(source,'r') as a,h5py.File(destination,'x') as b:
                def visit(key,x):
                    nonlocal count
                    if isinstance(x,h5py.Group): y=b.require_group(key)
                    else:
                        kwargs={'compression':'gzip','compression_opts':4,'shuffle':True} if x.shape and all(x.shape) and x.dtype.kind!='O' else {}
                        y=b.create_dataset(key,data=x[()],dtype=x.dtype,**kwargs); count+=1
                    for k,v in x.attrs.items(): y.attrs[k]=v
                for k,v in a.attrs.items(): b.attrs[k]=v
                a.visititems(visit)
            with h5py.File(source,'r') as a,h5py.File(destination,'r') as b:
                def check(key,x):
                    y=b[key];assert set(x.attrs)==set(y.attrs)
                    for k in x.attrs: assert np.array_equal(x.attrs[k],y.attrs[k])
                    if isinstance(x,h5py.Dataset):
                        assert x.dtype==y.dtype and x.shape==y.shape and np.asarray(x[()]).tobytes()==np.asarray(y[()]).tobytes()
                a.visititems(check)
                for k in a.attrs:assert np.array_equal(a.attrs[k],b.attrs[k])
            extra=sum((ROOT/r['directory']/f).stat().st_size for f in ('telemetry.h5','actions.npz','initial_observation.h5'))
            reports.append(dict(seed=seed,arm=arm,episode_id=r['episode_id'],source_sha256=sha(source),source_bytes=source.stat().st_size,
                repacked_bytes=destination.stat().st_size,total_standard_artifact_bytes=destination.stat().st_size+extra,
                datasets_exact=count,elapsed_s=time.time()-start))
            destination.unlink()
    write('storage_probe_six.json',dict(rule='min sha256(v1010b/storage/+episode_id) per arm/seed; outcomes unused',rows=reports,
        mean_standard_MiB=float(np.mean([r['total_standard_artifact_bytes'] for r in reports])/2**20),
        max_standard_MiB=max(r['total_standard_artifact_bytes'] for r in reports)/2**20,
        source_hashes=HASHES,codec='gzip4+shuffle numeric arrays; strings/scalars unchanged; all attributes preserved'))
    print(json.dumps(reports,indent=2))

if __name__=='__main__': main()
