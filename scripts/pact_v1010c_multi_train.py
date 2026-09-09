"""Seed-parameterized joint fine-tuning with original policy math and main's encoder path."""
from __future__ import annotations
import argparse
import signal
import time
from pact_v1010c_multi_core import *
from fixed_split_data import seed_epoch
from utils import set_seed
import imitate_episodes as original


def validate_optimizer(optimizer, step):
    steps = {int(s['step']) for s in optimizer.state.values() if 'step' in s}
    assert steps == {step}, (steps, step)
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[-1]['lr'] == 1e-5
    return len(optimizer.state)


def resource_sample():
    import pynvml
    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory = pynvml.nvmlDeviceGetMemoryInfo(device, version=pynvml.nvmlMemory_v2)
    cg = Path('/sys/fs/cgroup')
    def scalar(name):
        value = (cg / name).read_text().strip()
        return None if value == 'max' else int(value)
    values = {k:scalar(k) for k in ('memory.current','memory.max','pids.current','pids.max')}
    events = dict(line.split() for line in (cg/'memory.events').read_text().splitlines())
    import shutil
    return {'utc':now(), 'ram_fraction':values['memory.current']/values['memory.max'],
            'pid_fraction':values['pids.current']/values['pids.max'],
            'vram_fraction':memory.used/memory.total, 'oom_kill':int(events['oom_kill']),
            'disk_free_bytes':shutil.disk_usage(C).free}


def train(stop_updates):
    assert stop_updates in (300, 60000)
    contract = read(C/'contract.json')
    assert contract['sha256'] == digest({k:v for k,v in contract.items() if k!='sha256'})
    for path, expected in contract['code_hashes'].items():
        assert sha(ROOT/path)==expected, path
    initial = read(C/'initialization.json')
    assert sha(initial['path']) == initial['sha256']
    directory = C/'checkpoint'
    directory.mkdir(exist_ok=True)
    resume = directory/'resume_bundle.ckpt'
    baseline_resource = resource_sample()
    resources = C/'training_resources.jsonl'
    append(resources, baseline_resource)
    assert baseline_resource['disk_free_bytes'] > 13*2**30
    train_loader, val_loader, stats, stats_meta = make_loaders()
    from pact_v1010c_multi_capacity import CapacityLoader
    train_loader, val_loader = CapacityLoader(train_loader), CapacityLoader(val_loader)
    encoder = make_encoder()
    initial_encoder = {k:v.detach().cpu().clone() for k,v in encoder.inner.state_dict().items()}
    set_seed(TRAINING_SEED)
    policy = make_policy()
    optimizer = add_main_encoder_group(policy.configure_optimizers(), encoder)
    bridge = PolicyCallAdapter(policy)
    global_step = 0
    start_epoch = 0
    if resume.exists():
        bundle = torch.load(resume, map_location='cuda', weights_only=False)
        assert bundle['contract_sha256'] == contract['sha256']
        policy.load_state_dict(bundle['model_state'], strict=True)
        encoder.load_state_dict(bundle['encoder_state'], strict=True)
        optimizer.load_state_dict(bundle['optimizer_state'])
        global_step = int(bundle['global_step'])
        start_epoch = int(bundle['epoch'])+1
        assert global_step == start_epoch*30 and 0<global_step<stop_updates
        validate_optimizer(optimizer, global_step)
        original._restore_rng_state(bundle['rng_state'])
        old = lines(directory/'epoch_log.jsonl')
        extra = [r for r in old if r['global_step']>global_step]
        if extra:
            freeze(directory/f'uncommitted_epochs_{time.time_ns()}.json', extra)
            (directory/'epoch_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old if r['global_step']<=global_step))
        del bundle
    else:
        assert not (directory/'epoch_log.jsonl').exists()
        freeze(directory/'initialization_receipt.json', {'utc':now(), 'encoder_source':initial,
            'policy_seed':TRAINING_SEED, 'policy_initialization':f'fresh after setting seed{TRAINING_SEED}',
            'policy_projection_shape':list(policy.model.input_proj_proximity.weight.shape),
            'encoder_initial_tensors_equal_checkpoint':True, 'optimizer_groups':len(optimizer.param_groups)})
        (directory/'dataset_stats.pkl').write_bytes((W/'dataset_stats.pkl').read_bytes())
        freeze(directory/'run_manifest.json', {'schema':SCHEMA, 'seed':TRAINING_SEED,
            'policy_config':policy_config(), 'contract_sha256':contract['sha256'],
            'stats_sha256':sha(directory/'dataset_stats.pkl'), 'encoder_source_sha256':initial['sha256']})
    start_step = global_step
    stop_requested = False
    stop_reason = None
    def request_stop(*unused):
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = 'signal'
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    sampled = started
    pressure = 0
    gradient_check = None
    for epoch in range(start_epoch, 2000):
        seed_epoch(train_loader, val_loader, TRAINING_SEED, epoch)
        policy.eval()
        encoder.eval()
        with torch.inference_mode():
            vals = [forward_pass(batch, bridge, encoder) for batch in val_loader]
            val = original.compute_dict_mean(vals)
        policy.train()
        encoder.train()
        optimizer.zero_grad()
        epoch_values = []
        for batch in train_loader:
            output = forward_pass(batch, bridge, encoder)
            loss = output['loss']
            assert torch.isfinite(loss).all(), 'nonfinite loss'
            loss.backward()
            if gradient_check is None:
                nonzero = [n for n,p in encoder.named_parameters()
                           if p.grad is not None and torch.count_nonzero(p.grad).item()>0]
                assert any('conv_stem' in n for n in nonzero)
                assert any('transformer' in n for n in nonzero)
                assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
                assert all(p.grad is None for n,p in encoder.named_parameters()
                           if any(k in n for k in ('embedding_head','auxiliary_head','reconstruction_head')))
                gradient_check = {'stem_and_transformer_nonzero':True,
                                  'nonzero_parameters':nonzero, 'unused_pretraining_heads_have_no_gradient':True}
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            epoch_values.append(original.detach_dict(output))
        mean = original.compute_dict_mean(epoch_values)
        record = {'utc':now(), 'epoch':epoch, 'global_step':global_step,
                  'train':{k:float(v) for k,v in mean.items()},
                  'validation':{k:float(v) for k,v in val.items()},
                  'elapsed_s':time.monotonic()-started}
        append(directory/'epoch_log.jsonl', record)
        atomic(directory/'progress.json', record)
        print(json.dumps(record), flush=True)
        if time.monotonic()-sampled >= 60:
            sample = resource_sample()
            append(resources, sample)
            over = sample['ram_fraction']>.80 or sample['pid_fraction']>.75 or sample['vram_fraction']>.85
            pressure = pressure+1 if over else 0
            if pressure>=3 or sample['oom_kill']>baseline_resource['oom_kill'] or sample['disk_free_bytes']<10*2**30:
                stop_requested, stop_reason = True, 'resource guard'
            from datetime import datetime
            if time.time() >= datetime.fromisoformat(contract['hard_deadline_utc']).timestamp()-3600:
                stop_requested, stop_reason = True, 'deadline reserve'
            sampled = time.monotonic()
        if global_step%300==0 or global_step==stop_updates or stop_requested:
            states = validate_optimizer(optimizer, global_step)
            safe_torch_save({'schema':SCHEMA, 'contract_sha256':contract['sha256'],
                'epoch':epoch, 'global_step':global_step, 'model_state':policy.state_dict(),
                'encoder_state':encoder.state_dict(), 'optimizer_state':optimizer.state_dict(),
                'rng_state':original._rng_state()}, resume)
            atomic(directory/'checkpoint_progress.json', {'utc':now(), 'global_step':global_step,
                'resume_sha256':sha(resume), 'optimizer_states':states})
        if global_step==stop_updates or stop_requested:
            break
    assert global_step == (epoch+1)*30
    changed = [k for k,v in encoder.inner.state_dict().items()
               if not torch.equal(v.detach().cpu(), initial_encoder[k])]
    assert any(k.startswith('conv_stem') for k in changed)
    assert any(k.startswith('transformer') for k in changed)
    assert not any(any(part in k for part in ('embedding_head','auxiliary_head','reconstruction_head')) for k in changed)
    receipt = {'utc':now(), 'global_step':global_step, 'started_step':start_step,
               'optimizer_calls':global_step-start_step, 'elapsed_s':time.monotonic()-started,
               'stop_requested':stop_requested, 'stop_reason':stop_reason,
               'encoder_changed_parameters':changed, 'gradient_check':gradient_check,
               'policy_projection_shape':list(policy.model.input_proj_proximity.weight.shape),
               'upstream_byte_hashes_verified':True, 'stats_sha256':sha(directory/'dataset_stats.pkl')}
    if not stop_requested:
        assert global_step == stop_updates
        receipt['pair'] = save_pair(policy, encoder, directory, global_step)
        # Strict round-trip of both models; restore caller RNG after checking.
        rng = original._rng_state()
        again_policy, again_encoder, _ = load_pair(directory, expected_step=global_step)
        assert all(torch.equal(v, again_policy.state_dict()[k]) for k,v in policy.state_dict().items())
        assert all(torch.equal(v, again_encoder.state_dict()[k]) for k,v in encoder.state_dict().items())
        original._restore_rng_state(rng)
        receipt['strict_pair_reload_exact'] = True
        del again_policy, again_encoder
    freeze(directory/f'process_{start_step}_{global_step}.json', receipt)
    if global_step == 60000 and not stop_requested:
        freeze(directory/'completed.json', receipt)
    print(json.dumps(receipt), flush=True)
    if stop_requested:
        raise SystemExit(f'training paused: {stop_reason}')
    return receipt


if __name__ == '__main__':
    os.environ.update(environment())
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-updates', type=int, required=True)
    train(parser.parse_args().stop_updates)
