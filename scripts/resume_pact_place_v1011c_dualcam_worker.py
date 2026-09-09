"""Checkpoint-boundary process renewal; original optimizer/sampling math retained."""
from pact_place_v1011c_dualcam import *
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import train_pact_place_v1011c_dualcam_worker as worker


def main():
    target = int(sys.argv[sys.argv.index('--max_steps')+1])
    original_freeze = worker.freeze
    def receipt(path,value):
        path = Path(path)
        if path.name == 'checkpoint_storage.json':
            path = path.with_name(f'checkpoint_storage_resume_to_{target}.json')
        if path.parent.name == 'progress':
            value = {**value,'resumed_segment_target':target,
                'cgroup_pids_current':int(Path('/sys/fs/cgroup/pids.current').read_text()),
                'cgroup_pids_events':Path('/sys/fs/cgroup/pids.events').read_text(),
                'process_threads':next(int(s.split()[1]) for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))}
        if path.name in {f'policy_step_{step}.json' for step in MILESTONES}:
            step = value['global_step']
            manifest = path.parent/f'policy_step_{step}_run_manifest.json'
            assert not manifest.exists()
            os.link(path.parent/'run_manifest.json',manifest)
            assert sha256_file(manifest) == value['run_manifest_sha256']
        original_freeze(path,value)
    worker.freeze = receipt
    worker.main()


if __name__ == '__main__':
    main()
