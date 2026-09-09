"""One changed decision in the original loader: the supervised query start."""
from __future__ import annotations
import collections
import hashlib
import inspect
import textwrap
from pathlib import Path
from pact_v1010b_contract import NAMESPACE, append, read, digest
import numpy as np
import torch
import utils


def choose_start(T,close,episode_id,epoch,draw_index,p,*,training_seed=0):
    assert T>0 and 0<=close<T and p in (0.,.25)
    uniform=int(np.random.choice(T))  # Always consume exactly the original draw.
    if p==0:return uniform
    identity=[NAMESPACE,'acquisition_window',int(training_seed),int(epoch),str(episode_id),int(draw_index)]
    rng=np.random.default_rng(int(digest(identity),16))
    return int(rng.integers(max(0,close-30),min(T-1,close+30)+1)) if rng.random()<p else uniform


class AcquisitionWindowDataset(utils.EpisodicDataset):
    @classmethod
    def wrap(cls,dataset,labels,training_seed,p,stream_dir=None,batch_size=8,num_workers=4):
        obj=cls.__new__(cls);obj.__dict__.update(dataset.__dict__)
        obj.labels={int(r['act_episode_index']):r for r in labels}
        assert set(obj.episode_ids)==set(obj.labels)
        obj.training_seed=training_seed;obj.p=p;obj.stream_dir=Path(stream_dir) if stream_dir else None
        obj.batch_size=batch_size;obj.num_workers=num_workers;obj.set_epoch(0)
        return obj

    def set_epoch(self,epoch):
        self.epoch=int(epoch);self.draw_indices=collections.Counter();self.local_draw=0

    def _choose_start(self,T,episode_index):
        label=self.labels[int(episode_index)];assert T==label['T']
        draw=self.draw_indices[episode_index];self.draw_indices[episode_index]+=1
        start=choose_start(T,label['close'],label['episode_id'],self.epoch,draw,self.p,training_seed=self.training_seed)
        worker=torch.utils.data.get_worker_info();wid=worker.id if worker else 0
        workers=worker.num_workers if worker else 1
        position=(self.local_draw//self.batch_size)*workers*self.batch_size+wid*self.batch_size+self.local_draw%self.batch_size
        self.local_draw+=1
        if self.stream_dir:
            append(self.stream_dir/f'epoch_{self.epoch:04d}_worker_{wid}.jsonl',{
                'epoch':self.epoch,'stream_index':position,'episode_id':label['episode_id'],
                'episode_index':int(episode_index),'draw_index':draw,'start':start})
        return start

# Preserve every preprocessing, alignment, normalization and padding statement.
_source=textwrap.dedent(inspect.getsource(utils.EpisodicDataset.__getitem__))
assert _source.count('start_ts = np.random.choice(episode_len)')==1
_source=_source.replace('start_ts = np.random.choice(episode_len)','start_ts = self._choose_start(episode_len, episode_id)')
_globals=dict(utils.__dict__)
exec(compile(_source,__file__,'exec'),_globals)
AcquisitionWindowDataset.__getitem__=_globals['__getitem__']


def stream_summary(directory,start_epoch=2000,end_epoch=2100):
    from pact_v1010b_contract import lines
    rows=[r for p in Path(directory).glob('epoch_*_worker_*.jsonl') for r in lines(p)
        if start_epoch<=r['epoch']<end_epoch]
    rows.sort(key=lambda r:(r['epoch'],r['stream_index']))
    expected=(end_epoch-start_epoch)*240
    assert len(rows)==expected
    assert len({(r['epoch'],r['stream_index']) for r in rows})==expected
    for epoch in range(start_epoch,end_epoch):
        selected=[r for r in rows if r['epoch']==epoch]
        assert [r['stream_index'] for r in selected]==list(range(240))
        assert len({r['episode_id'] for r in selected})==240 and all(r['draw_index']==0 for r in selected)
    return {'sample_count':len(rows),'start_epoch':start_epoch,'end_epoch':end_epoch,
        'sample_stream_sha256':digest([(r['epoch'],r['episode_id'],r['start']) for r in rows]),
        'full_rows_sha256':digest(rows)}
