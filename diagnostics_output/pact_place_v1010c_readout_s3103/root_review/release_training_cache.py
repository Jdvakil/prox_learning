"""Evict only clean cached training H5 pages after training; never edit the files."""
import datetime
import hashlib
import json
import os
from pathlib import Path

C = Path(__file__).resolve().parents[1]
ROOT = C.parents[1]
W = ROOT / 'diagnostics_output/pact_place_v1010_wrist288_s3_v1'


def memory():
    cg = Path('/sys/fs/cgroup')
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    return {'current_bytes': int((cg / 'memory.current').read_text()),
            'max_bytes': int((cg / 'memory.max').read_text()),
            'file_bytes': int(stats['file']), 'anon_bytes': int(stats['anon'])}


def release():
    assert json.loads((C / 'checkpoint/completed.json').read_text())['global_step'] == 60000
    manifest_path = W / 'conversion_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    before = memory()
    files, total_bytes = [], 0
    for row in manifest['episodes']:
        path = (W / 'converted' / row['act_file']).resolve()
        assert path.is_relative_to((W / 'converted').resolve())
        info = path.stat()
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        after = path.stat()
        assert (info.st_size, info.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        files.append(str(path))
        total_bytes += info.st_size
    assert len(files) == 280
    record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'operation': 'POSIX_FADV_DONTNEED on read-only descriptors for the completed training dataset',
        'training_complete_at': 60000, 'files': 280, 'file_bytes_advised': total_bytes,
        'before': before, 'after': memory(), 'data_size_and_mtime_unchanged': True,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (C / 'root_review/training_cache_release.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return record


if __name__ == '__main__':
    print(json.dumps(release(), indent=2))
