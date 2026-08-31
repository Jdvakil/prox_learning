# Superseded V10.9 evaluation manifest, attempt 01

`eval_manifest_01_superseded_self_hash_defect.json` and its audit are retained,
not deleted, and are **not** the operative manifest.

Defect: `manifest_sha256` was computed over the document excluding only
`manifest_sha256`. The immutable writer then appended its own `payload_sha256`
to the file, so the stored self-hash could never validate on reload and
`load_manifest` refused it. No rollout was ever run against this manifest.

Fix: the self-hash is now computed over the document excluding both
`manifest_sha256` and `payload_sha256`. The operative manifest is
`eval_manifest.json`.
