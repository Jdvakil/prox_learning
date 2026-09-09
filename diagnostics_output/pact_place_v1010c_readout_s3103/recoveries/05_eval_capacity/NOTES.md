# Evaluation capacity after training completes

Training completed60000 committed updates and the final policy/encoder passed strict paired reload. The first eight readout workers started correctly, but RAM reached81.6% for three consecutive minute samples. The existing80% guard stopped new launches and is draining those eight workers. This capacity decision was made before their task outcomes were available.

At23:12UTC cgroup memory consisted of about50GiB anonymous memory and88GiB file cache. A scoped POSIX_FADV_DONTNEED advisory on read-only descriptors for the280 completed-training H5 files reduced current memory by14.9GB. No dataset bytes, sizes or modification times changed. The operation and before/after memory counters are recorded in root_review/training_cache_release.jsonl. It did not clear global caches or affect unrelated files.

After the first pool closes, the launcher requires all eight completed rows to have observed successful exits and passed initial matching. It retains them and continues only the42 pending rows, using six workers for more RAM headroom. Full source/data verification is followed by release of the now-unused training file cache, since hashing the training files warms it again. All original RAM/VRAM/PID/disk limits and the original48h deadline remain unchanged.

This changes evaluation capacity and unused page-cache residency only. Training, paired weights, all50 scene identities, physics, horizon, decoder, metrics and matching tolerances remain unchanged. No completed rollout is repeated or replaced.
