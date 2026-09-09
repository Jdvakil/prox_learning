"""Transactional, verified lossless storage; only new-run files are writable."""
from __future__ import annotations
import os
from pathlib import Path
import h5py
import numpy as np
from pact_v1010b_contract import B, inside, sha, freeze


def names(h):
    out={}
    h.visititems(lambda n,o:out.update({n:type(o).__name__}))
    return out


def same_value(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype: return False
    if a.dtype.kind=='O': return bool(np.array_equal(a,b))
    return a.tobytes()==b.tobytes()


def verify_h5_equal(source,destination):
    with h5py.File(source,'r') as a,h5py.File(destination,'r') as b:
        na,nb=names(a),names(b); assert na==nb and na
        count=0
        for name in ['',*na]:
            av,bv=a[name] if name else a,b[name] if name else b
            assert set(av.attrs)==set(bv.attrs), f'attribute names: {name}'
            for key in av.attrs: assert same_value(av.attrs[key],bv.attrs[key]),f'attribute: {name}/{key}'
            if isinstance(av,h5py.Dataset):
                assert av.dtype==bv.dtype and av.shape==bv.shape,name
                assert same_value(av[()],bv[()]),f'dataset: {name}'
                count+=1
    return count


def repack(source):
    source=inside(source); assert source.name=='trajectory.h5'
    destination=source.with_name('trajectory.repacking.h5')
    receipt=source.parent/'storage_provenance.json'
    assert not receipt.exists(), 'storage publication is immutable'
    assert not destination.exists(), 'partial repack must be preserved under an invalid attempt'
    before=sha(source); before_bytes=source.stat().st_size
    with h5py.File(source,'r') as a,h5py.File(destination,'x') as b:
        for key,value in a.attrs.items(): b.attrs[key]=value
        def copy(name,obj):
            if isinstance(obj,h5py.Group): target=b.require_group(name)
            else:
                value=obj[()]
                options={'compression':'gzip','compression_opts':4,'shuffle':True} if obj.ndim and obj.size and obj.dtype.kind not in 'OSU' else {}
                target=b.create_dataset(name,data=value,dtype=obj.dtype,**options)
            for key,value in obj.attrs.items(): target.attrs[key]=value
        a.visititems(copy); b.flush()
    count=verify_h5_equal(source,destination)
    assert sha(source)==before, 'source changed during repack'
    doc={'source_path':str(source),'original_sha256':before,'original_bytes':before_bytes,
        'final_sha256':sha(destination),'final_bytes':destination.stat().st_size,'datasets_exact':count,
        'all_attributes_exact':True,'codec':'gzip4+shuffle on nonempty numeric arrays; strings/scalars unchanged',
        'new_run_only':True,'source_replaced_only_after_exact_verification':True}
    # Freeze a recoverable transaction intent before the atomic rename.
    freeze(source.parent/'storage_verified_intent.json',doc)
    os.replace(destination,source)
    assert sha(source)==doc['final_sha256']
    freeze(receipt,doc)
    return doc
