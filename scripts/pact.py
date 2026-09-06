#!/usr/bin/env python3
"""One entry point for dataset-bound PACT training and simulation evaluation."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from pact_workflow import ROOT, check_dataset_files, file_digest, load_contract, prepare_contract, resolve, write_json
from pact_checkpoint import paired_encoder_checkpoint

DEFAULT_ENCODER = 'experiments_output/default/surface_encoder_train/pact_place_corridor_v5/pact_surface_embedding_encoder_v1.pt'


def profiles():
    return json.loads((ROOT / 'configs/pact_datasets.json').read_text())['datasets']


def manifest_path(dataset):
    return ROOT / 'assets/pact_experiments' / dataset / 'experiment.json'


def runtime_setup(profile):
    """Export pinned local git objects; never switch the user's submodule/worktree."""
    dest = resolve(profile['runtime_dir'])
    marker = dest / 'runtime.json'
    revision = subprocess.check_output(['git', '-C', str(ROOT / 'submodules/molmospaces'),
                                        'rev-parse', profile['molmospaces_revision']], text=True).strip()
    if marker.exists():
        verify_runtime(profile)
        return dest
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f'Incomplete runtime already exists: {dest}; use a fresh runtime_dir')
    blob = subprocess.check_output(['git', '-C', str(ROOT / 'submodules/molmospaces'),
                                    'archive', revision, 'molmo_spaces'])
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        # Only regular files/directories; no symlink traversal or executable archive hooks.
        for member in archive.getmembers():
            target = dest / member.name
            if not target.resolve().is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive path')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            else:
                raise ValueError(f'Unsupported archive entry: {member.name}')
    if profile['adapter'] == 'v1011d':
        helper = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                          '7d1e25ee:scripts/pact_place_v9_contract.py'])
        (dest / 'pact_place_v9_contract.py').write_bytes(helper)
    if profile.get('scene_blob'):
        scene = dest / 'molmo_spaces/data_generation/custom_scenes' / profile['scene_filename']
        scene.write_bytes(subprocess.check_output([
            'git', '-C', str(ROOT / 'submodules/molmospaces'), 'cat-file', 'blob', profile['scene_blob']]))
        if file_digest(scene) != profile['scene_sha256']:
            raise ValueError('Archived scene differs from recorded collection hash')
    hashes = {str(p.relative_to(dest)): file_digest(p) for p in dest.rglob('*') if p.is_file()}
    write_json(marker, {'revision': revision, 'files': hashes})
    return dest


def verify_runtime(profile):
    dest = resolve(profile['runtime_dir'])
    marker = json.loads((dest / 'runtime.json').read_text())
    if not marker['revision'].startswith(profile['molmospaces_revision']):
        raise ValueError('Runtime revision differs from experiment')
    for rel, expected in marker['files'].items():
        if file_digest(dest / rel) != expected:
            raise ValueError(f'Pinned runtime was modified: {dest / rel}')
    return dest


def evaluation_python(profile):
    local = ROOT / 'assets/pact_env' / profile['adapter'] / 'bin/python'
    return str(local) if local.exists() else sys.executable


def setup_environment(profile):
    destination = ROOT / 'assets/pact_env' / profile['adapter']
    if not (destination / 'bin/python').exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(destination)], check=True)
    # Overlay only the collection's simulator packages. Reuse the installed
    # PyTorch/CUDA stack without modifying the user's training environment.
    subprocess.run([str(destination / 'bin/python'), '-m', 'pip', 'install', '--no-deps',
                    'mujoco==3.5.0', 'mujoco-warp==3.5.0', 'mujoco-mjx==3.5.0',
                    'warp-lang==1.11.1'], check=True)
    preflight(profile)


def preflight(profile):
    runtime = verify_runtime(profile)
    compatibility = runtime / 'molmo_spaces/data_generation/runtime_compat.py'
    if compatibility.exists():
        check = subprocess.run([evaluation_python(profile), '-c',
            "import runpy,sys; c=runpy.run_path(sys.argv[1]); issues=c['check_runtime'](); "
            "print(c['format_report'](issues)); sys.exit(bool(issues))", str(compatibility)],
            text=True, capture_output=True)
        if check.returncode:
            raise ValueError(check.stdout + check.stderr +
                             '\nRun scripts/pact.py setup DATASET --env using your training Python. See README section 4.20.')
    print(f"Runtime files verified: {profile['molmospaces_revision']}")


def run_directory(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('--run must be a simple unique name')
    return ROOT / 'runs/pact' / name


def binding(run):
    directory = run_directory(run)
    pointer = directory / 'legacy_checkpoint.json'
    checkpoint = resolve(json.loads(pointer.read_text())['checkpoint_dir']) if pointer.exists() else directory
    contract = load_contract(directory / 'experiment.json')
    return directory, checkpoint, contract


def launch(command, dry_run=False, env=None):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT / 'submodules/act', env=env, check=True)


def train_command(args, contract):
    encoder = getattr(args, 'encoder_checkpoint', None)
    encoder_lr = getattr(args, 'encoder_lr', None)
    if args.arm != 'readout' and (encoder or encoder_lr is not None):
        raise ValueError('Encoder options require --arm readout')
    command = [sys.executable, str(ROOT / 'submodules/act/imitate_episodes.py'),
               '--experiment_manifest', str(manifest_path(args.dataset)),
               '--run_dir', str(run_directory(args.run)), '--ckpt_dir', str(ROOT / 'runs/pact'),
               '--task_name', args.dataset, '--policy_class', 'ACT', '--batch_size', str(args.batch_size),
               '--seed', str(args.seed), '--num_epochs', str(args.epochs), '--lr', str(args.lr),
               '--chunk_size', str(contract['profile']['chunk_size']), '--kl_weight', '10',
               '--hidden_dim', '512', '--dim_feedforward', '3200', '--wandb_run_name', args.run]
    if args.arm == 'raw':
        command += ['--use_proximity', '--prox_feature', 'raw', '--prox_layout', 'per_sensor', '--prox_pool', 'min']
    elif args.arm == 'readout':
        encoder = resolve(encoder or DEFAULT_ENCODER)
        if not encoder.is_file():
            raise ValueError(f'Missing pretrained surface encoder: {encoder}; pass --encoder-checkpoint PATH')
        command += ['--use_proximity', '--prox_feature', 'surface_embedding',
                    '--prox_layout', 'per_sensor', '--prox_pool', 'min', '--prox_tokens_per_sensor', '1',
                    '--prox_encoder_ckpt', str(encoder), '--finetune_prox_encoder', '--prox_policy_tap', 'readout']
        if encoder_lr is not None:
            if not 0 < encoder_lr < float('inf'):
                raise ValueError('encoder-lr must be finite and positive')
            command += ['--prox_encoder_lr', str(encoder_lr)]
    return command


def convert_command(dataset, profile):
    return [sys.executable, '-m', 'scripts.convert_pact_place_to_act',
            '--src', str(resolve(profile['raw_dir'])), '--dst', str(resolve(profile['data_dir'])),
            '--with_proximity', '--prox_pool', 'min', '--image_h', '240', '--image_w', '320',
            '--task_name', dataset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    conversion = sub.add_parser('convert', help='Convert clean demonstrations for both ACT and PACT')
    conversion.add_argument('dataset', choices=profiles())
    conversion.add_argument('--dry-run', action='store_true')
    for verb in ('prepare', 'setup'):
        p = sub.add_parser(verb)
        p.add_argument('dataset', choices=profiles())
        if verb == 'setup':
            p.add_argument('--env', action='store_true', help='Install pinned simulator packages in an isolated local environment')
    train = sub.add_parser('train')
    train.add_argument('dataset', choices=profiles())
    train.add_argument('--run', required=True)
    train.add_argument('--arm', choices=('readout', 'raw', 'act'), default='readout',
                       help='readout (default): jointly finetune the surface encoder with ACT; raw/act are baselines')
    train.add_argument('--encoder-checkpoint', help=f'Readout initialization; default: {DEFAULT_ENCODER}')
    train.add_argument('--encoder-lr', type=float, help='Readout encoder learning rate; defaults to --lr')
    train.add_argument('--epochs', type=int, default=2000)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--lr', type=float, default=1e-5)
    train.add_argument('--dry-run', action='store_true')
    adopt = sub.add_parser('adopt', help='Bind an existing checkpoint without changing its weights or saved normalization')
    adopt.add_argument('dataset', choices=profiles())
    adopt.add_argument('--checkpoint', required=True)
    adopt.add_argument('--run', required=True)
    offline = sub.add_parser('offline', help='Fast action prediction diagnostic using the run split')
    offline.add_argument('--run', required=True)
    offline.add_argument('--split', choices=('train', 'val'), default='val')
    offline.add_argument('--limit', type=int, default=8)
    offline.add_argument('--dry-run', action='store_true')
    check = sub.add_parser('check', help='Verify run data, runtime pin and dependency versions without launching a rollout')
    check.add_argument('--run', required=True)
    for verb in ('eval', 'verify'):
        p = sub.add_parser(verb)
        p.add_argument('--run', required=True)
        p.add_argument('--checkpoint-name', default='policy_best.ckpt')
        p.add_argument('--dry-run', action='store_true')
        if verb == 'eval':
            p.add_argument('--suite', choices=('smoke', 'dev', 'test'), default='smoke')
            p.add_argument('--reference', action='store_true', help='Render every step for a diagnostic comparison')
    args = parser.parse_args()
    if args.command == 'convert':
        profile = profiles()[args.dataset]
        destination = resolve(profile['data_dir'])
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f'Refusing to overwrite converted data: {destination}')
        command = convert_command(args.dataset, profile)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.command == 'list':
        for name, p in profiles().items():
            variant = p.get('dataset_environment_version', p['environment_version'])
            print(f"{name}: {variant} | {p['data_dir']} | cameras={p['camera_names']}")
        return
    if args.command == 'setup':
        print(runtime_setup(profiles()[args.dataset]))
        if args.env:
            setup_environment(profiles()[args.dataset])
        return
    if args.command == 'prepare':
        contract = prepare_contract(args.dataset, profiles()[args.dataset])
        path = manifest_path(args.dataset)
        if path.exists() and load_contract(path)['sha256'] != contract['sha256']:
            raise ValueError('Prepared contract differs; use a new dataset profile name to preserve existing suites')
        write_json(path, contract)
        print(f"{path}: train={len(contract['split']['train'])}, val={len(contract['split']['val'])}, "
              f"dev={len(contract['evaluation']['dev'])}, test={len(contract['evaluation']['test'])}")
        return
    if args.command in ('train', 'adopt'):
        contract = load_contract(manifest_path(args.dataset))
        check_dataset_files(contract)
        dest = run_directory(args.run)
        if dest.exists() and any(dest.iterdir()):
            raise ValueError(f'Run name already in use: {dest}')
        if args.command == 'train':
            if args.epochs <= 0 or args.batch_size <= 0:
                raise ValueError('epochs and batch-size must be positive')
            launch(train_command(args, contract), args.dry_run)
        else:
            checkpoint = resolve(args.checkpoint)
            for name in ('policy_best.ckpt', 'dataset_stats.pkl'):
                if not (checkpoint / name).is_file():
                    raise ValueError(f'Missing {checkpoint / name}')
            # Explicit binding is an assertion of provenance, not a claim the old
            # weights were trained using the new grouped split.
            write_json(dest / 'experiment.json', contract)
            write_json(dest / 'legacy_checkpoint.json', {
                'checkpoint_dir': str(checkpoint), 'split': 'legacy_unknown_or_leaky',
                'normalization': 'preserved_original', 'dataset_binding': 'user_declared',
            })
            print(f'Bound {dest}; original weights/stats preserved. Legacy validation is not held out under the new split.')
        return
    directory, checkpoint, contract = binding(args.run)
    if args.command == 'check':
        check_dataset_files(contract)
        preflight(contract['profile'])
        for filename in ('policy_best.ckpt', 'dataset_stats.pkl'):
            if not (checkpoint / filename).is_file():
                raise ValueError(f'Missing {checkpoint / filename}')
        prox = checkpoint / 'prox_config.json'
        if prox.exists():
            paired_encoder_checkpoint(checkpoint, json.loads(prox.read_text()))
        print('Run data and checkpoint files verified. Live judge/rollout verification remains separate.')
        return
    if args.command == 'offline':
        if args.limit <= 0:
            raise ValueError('limit must be positive')
        command = [sys.executable, str(ROOT / 'submodules/act/eval_train_set.py'),
                   '--ckpt_dir', str(checkpoint), '--data_dir', str(resolve(contract['profile']['data_dir'])),
                   '--split', args.split, '--limit_episodes', str(args.limit),
                   '--output', str(directory / f'offline_{args.split}.json')]
        if not (directory / 'legacy_checkpoint.json').exists():
            ids_path = directory / f'{args.split}_episode_ids.json'
            if not args.dry_run:
                write_json(ids_path, contract['split'][args.split])
            command += ['--episode_ids', str(ids_path)]
        else:
            print('Legacy checkpoint: seed-1 split diagnostic; validation may contain normalization/scene leakage.')
        launch(command, args.dry_run)
        return
    if not args.dry_run:
        preflight(contract['profile'])
    command = [evaluation_python(contract['profile']), str(ROOT / 'submodules/act/eval_pact.py'),
               '--run-dir', str(directory), '--checkpoint-dir', str(checkpoint),
               '--checkpoint-name', args.checkpoint_name]
    if args.command == 'verify':
        command += ['--verify']
    else:
        command += ['--suite', args.suite]
        if args.reference:
            command += ['--reference']
    launch(command, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'PACT command failed (exit {error.returncode}); see the child output and report above.', file=sys.stderr)
        raise SystemExit(error.returncode) from None
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
