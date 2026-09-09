"""Cover converted .hdf5 and completed training checkpoints in cache release."""
from __future__ import annotations
import argparse
from pact_wrist288_common import *
import pact_wrist280_final50 as followup

RECOVERY = WORK / 'amendments/final_cache_recovery_20260907'
capacity = followup.capacity
prior_candidates = capacity.cache_candidates


def extra_cache_candidates():
    root = WORK.resolve()
    receipt = WORK / 'processes/conversion/exit_receipt.json'
    if receipt.exists() and read(receipt)['returncode'] == 0:
        for path in (WORK / 'converted').glob('*.hdf5'):
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                yield path
    for directory in (WORK / 'checkpoints').iterdir():
        progress = directory / 'progress.json'
        if not progress.exists() or read(progress)['updates'] != 60000:
            continue
        # The evaluation checkpoint stays cached. These files are no longer
        # needed for training; fadvise preserves their on-disk contents.
        for name in ('policy_best.ckpt', 'policy_last.ckpt', 'policy_update_30000.ckpt', 'resume_bundle.ckpt'):
            path = directory / name
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                yield path


def candidates():
    yield from prior_candidates()
    yield from extra_cache_candidates()


def install():
    contract = read(RECOVERY / 'contract.json')
    assert contract['sha256'] == digest({k: v for k, v in contract.items() if k != 'sha256'})
    assert sha(followup.AMEND / 'contract.json') == contract['followup_contract_file_sha256']
    for path, expected in contract['file_hashes'].items():
        assert sha(ROOT / path) == expected, path
    for path in (WORK / 'checkpoints').glob('*/progress.json'):
        assert read(path)['updates'] == 60000
    installed = followup.install()
    capacity.cache_candidates = candidates
    analysis = installed[3]
    prior_report = analysis.report

    def report(verdict, reason, rows=None):
        prior_report(verdict, reason, rows)
        with (WORK / 'EVAL.md').open('a') as stream:
            stream.write('\nFinal evaluation cache recovery: include converted `.hdf5` files '
                         'and unused checkpoint files from fully completed training in scoped '
                         'file-cache release. Inference checkpoints stay cached. All disk contents, '
                         'memory thresholds, selected instances and completed results are preserved. '
                         'See `amendments/final_cache_recovery_20260907/contract.json`.\n')

    analysis.report = report
    return installed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('supervisor', 'monitor'), required=True)
    args = parser.parse_args()
    os.environ.update(environment())
    installed = install()
    if args.mode == 'monitor':
        installed[2].main()
    else:
        followup.supervise(installed)
