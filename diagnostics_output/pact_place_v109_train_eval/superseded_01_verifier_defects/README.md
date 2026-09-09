# Superseded V10.9 embedding artifacts, attempt 01

Retained, not deleted. **Not** operative. Both files were produced by a first
verification pass that reported two failures; both failures were defects in the
*verifier*, not in the encoded data. The corrected pass supersedes them.

## Defect 1 — wrong key for the encoder schema

The check read `payload["schema"]`. The frozen checkpoint declares its schema
under `payload["schema_version"]`, whose value is
`pact_surface_embedding_encoder_v1` as required. The corrected pass reads
`schema_version` and additionally asserts the checkpoint's own
`sensor_order_sha256` equals the canonical
`2198e29b796ce63f43d8b0db50a92da7d4429895f8571f7d87b655bc265c8fe1`, which it
does — the encoder was trained against this exact sensor order.

## Defect 2 — a non-reproducible semantic hash inherited from the V5 converter

`_semantic_sha256` hashes each dataset with
`np.ascontiguousarray(...).tobytes()`. For a variable-length UTF-8 string
dataset numpy returns an **object array**, and `.tobytes()` serialises the
*pointers*, not the strings. Reading the same unmodified file twice in one
process yields two different digests:

    tobytes hash run A: daa5c117f3f613d07a309cbe9a4fe654...
    tobytes hash run B: 96c085ab8b2ca327c219efa60d9e0a16...
    contents equal:     True

`observations/proximity_sensor_names` is such a dataset, so every
`act_semantic_sha256` and the `converted_tree_semantic_sha256` recorded at
conversion time are **not reproducible** and cannot serve as integrity anchors.
The first pass compared against them and reported all 141 files as changed.

Nothing downstream depended on them: training verifies `act_file_sha256` and
`converted_tree_file_sha256`, which are raw-file byte digests and are
deterministic.

The corrected pass drops the semantic hash as an anchor and proves preservation
the strong way instead — re-extracting `action`, `qpos`, `qvel`, wrist RGB, raw
proximity and the sensor extrinsics/intrinsics from the V10.8 **source** HDF5
and MP4 for all 141 episodes and comparing them element-wise against the
encoded files.
