# PACT front-end screen dataset-hash amendment

This amendment was frozen before encoder training, policy training, schedule
construction, or any screen rollout. No screen endpoint existed when it was
written. It changes only how the raw-data copy is verified.

The preregistered re-materializer initially required each copied H5 to equal
the original converter's recorded `act_semantic_sha256`. Episode 0 stopped on
that check. Audit showed the original `_semantic_sha256` calls
`ndarray.tobytes()` on HDF5 variable-length string datasets, whose NumPy dtype
is `object`. For object arrays, `tobytes()` serializes process-local pointer
values rather than string content. Rehashing the same unchanged source H5 in
three fresh Python processes produced three different hashes:

- `1e648344558e044be6b116080220d94060932bdefccdcf54e80bf73202321821`
- `025f66711a5b66b6b539e4b59ff72a02f9670a0cb66fd0498b5835ec03fda5eb`
- `b07311ff4a102ec6d321d08e48e6587f90bfd184161069d3a780a322d63b039f`

Therefore the legacy recorded value cannot be regenerated and is not a valid
content hash.

The amended verifier hashes numeric arrays by value and hashes
variable-length strings by UTF-8 content with an eight-byte length prefix. It
computes a raw view of each preserved encoded source while excluding only the
legacy derived token datasets:

- `observations/proximity_positions`
- `observations/proximity_valid`
- `observations/proximity_valid_probability`
- `observations/proximity_embeddings`

The token-free destination must equal that deterministic source raw-view hash
exactly. The copy still retains action, qpos/qvel, wrist RGB, raw proximity,
ordered sensor names, intrinsics, extrinsics, and provenance. The original
recorded legacy hash remains in the new manifest as an audit field but is not
used as evidence of equality.

The amended re-materializer SHA-256 is
`20b9a3c7fc0f80416b614320923e8dab2b21311da42183cc567db52d651f54a0`.
No other preregistered front-end, policy, environment, schedule, endpoint,
analysis, or decision-rule field changes.
