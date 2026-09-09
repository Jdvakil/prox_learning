# Superseded V10.9 split manifest, attempt 01

Retained, not deleted. **Not** operative. No training run ever consumed it.

Defect: written through `write_immutable_create_only`, which appends its own
`payload_sha256` key to the document *after* `split_manifest_sha256` was
computed. `fixed_split_data.load_split_manifest` recomputes the self-hash over
every key except `split_manifest_sha256`, so the injected key made the manifest
permanently unloadable:

    DataContractError: split manifest self-hash mismatch:
      recomputed f66b953d54fdbaab4ca12192f15ecc20ee8af2f5549a06e36a2db4169cd7b752
      != stored  52acfd76598916b68a48deb4cd868ed0627a850e650c259cc727077d534155db

Caught by loading it through the real training loader before launching training.

Fix: the operative `split_manifest.json` is written as text containing exactly
the document that was hashed, and the builder now round-trips it through
`fixed_split_data.load_split_manifest` and asserts 113/28 before returning. The
training loader itself is **not** modified — that would have broken the
requirement that no training/model/loader source differs from the V5 commit.
