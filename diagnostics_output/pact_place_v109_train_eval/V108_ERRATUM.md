# V10.8 erratum

Create-only. `diagnostics_output/pact_place_v108_collection/closeout.json` is
**not** modified. Written by V10.9; every correction is re-derived from
`ledger.jsonl` and the retained row files, not copied from the close-out.

V10.8's status is unchanged: an exploratory owner-override collection, stopped
early by owner instruction at 141 of 152 target successes, quotas unmet, not a
Phase-0 pass. V10.7's Phase-0 gate remains failed at 8/24 and permanently closed.

- ledger.jsonl SHA-256 `ca4adea083d4fd0f25eb2e0dfd39b910c36f877ad1d76309beabc563a63038f6` (353 records)
- closeout.json raw SHA-256 `a7601c2ca6bb1c81b07ff7ade20590504c3774cdc815b3a2ac04d5b423adcce0`
- closeout.json payload SHA-256 `1777ed44b64bd9501f3c4a2bad9a3d0f4427ad339971da7b05f9bb42021be941`
- source manifest payload SHA-256 `884d417d378972677251453bd61f13704c056969aca8e84b6b85d7febe050f0c` (141 rows)
- erratum payload SHA-256 `3d4ee5679b0326de35c329e62a5f8fe2eebb9a4ed83bcb93b6542157b47fa8c6`

## E1 — pendant involvement  ·  FALSE

**Previously:** no pendant involvement anywhere in the collection

**Correction:** Zero *accepted* rows contacted the pendant -- that part stands. But one rejected attempt did contact it, so the stronger claim is false.

```json
{
  "accepted_rows_with_pendant_contact": 0,
  "attempts_with_any_pendant_involvement": 1
}
```

## E2 — the one pendant-contact attempt  ·  OMITTED

**Previously:** not recorded in the close-out narrative

**Correction:** Rejected attempt 1a756c9304311cdc07091641e59af8da16b6098550aa2fc9ce9d1c0c99cb6ae8 recorded 42 mounted_fixture contacts, 1 pendant-contact frame, and zero clearance. It is the only such row in all 353 attempts and it was correctly rejected.

```json
{
  "accepted": false,
  "attempt_id": "1a756c9304311cdc07091641e59af8da16b6098550aa2fc9ce9d1c0c99cb6ae8",
  "attempt_index": 10,
  "cell": "F0_target_side_stagger|right|neg5",
  "clutter_contacts": 412,
  "clutter_stability_events": 1,
  "defects": [
    "contact:clutter=412",
    "contact:mounted_fixture=42",
    "clutter_stability_events=1",
    "pendant_contact"
  ],
  "episode_steps": 431,
  "min_lobe_stem_clearance_m": 0.0,
  "min_pendant_clearance_m": 0.0,
  "mounted_fixture_contacts": 42,
  "pendant_contact_frames": 1,
  "result_sha256": "1ce78636e31e0185a92b1a88c68394f1c98c1921487111f0c4fce34046b672b7",
  "row_sha256": "d5e64e431509bcab1a0b78fbd2125fa461abf0a32e29d8db4f077d1781256259",
  "task_seed_u32": 1876755499
}
```

## E3 — worker deaths and infrastructure events  ·  FALSE AND MISLEADING

**Previously:** closeout.json reports infrastructure_halts: 0

**Correction:** There were 8 BrokenProcessPool worker deaths, caused by terminating the process pool on the owner's stop instruction while a batch was in flight. They are owner-stop-induced worker terminations -- not zero infrastructure events, and not spontaneous data corruption. None advanced a scientific seed stream and none entered the 353 scientific ledger rows. They must not be conflated with the 12 sampling_failure ledger rows, which are a different, scientific outcome.

```json
{
  "closeout_infrastructure_halts": 0,
  "excluded_from_scientific_attempts": true,
  "halted_for_repair_attempt_id": "417388c39bd665e97bab108c8a9dbaf248176a03667041c30ff8f50e155aa8de",
  "halted_row_replaced": false,
  "halted_scientific_stream_advanced": false,
  "ledger_rows_with_status_sampling_failure": 12,
  "ledger_rows_with_status_worker_died": 0,
  "n_infrastructure_defects": 8,
  "recorded_worker_deaths": 8,
  "retries_authorized_by_contract": false,
  "runner_recorded_stop_reason": "infrastructure_or_schema_defect",
  "sampling_failure_cells": [
    "F2_outer_panel_stagger|left|pos5",
    "F3_aperture_side_stagger|left|pos5",
    "F3_aperture_side_stagger|right|center",
    "F3_aperture_side_stagger|right|neg5",
    "F3_aperture_side_stagger|right|pos5"
  ],
  "true_stop_reason": "owner_instructed_early_stop",
  "worker_death_cells": [
    {
      "attempt_index": 29,
      "cell": "F2_outer_panel_stagger|left|center"
    },
    {
      "attempt_index": 23,
      "cell": "F3_aperture_side_stagger|left|pos5"
    },
    {
      "attempt_index": 16,
      "cell": "F3_aperture_side_stagger|right|center"
    },
    {
      "attempt_index": 30,
      "cell": "F2_outer_panel_stagger|left|center"
    },
    {
      "attempt_index": 24,
      "cell": "F3_aperture_side_stagger|left|pos5"
    },
    {
      "attempt_index": 17,
      "cell": "F3_aperture_side_stagger|right|center"
    },
    {
      "attempt_index": 31,
      "cell": "F2_outer_panel_stagger|left|center"
    },
    {
      "attempt_index": 25,
      "cell": "F3_aperture_side_stagger|left|pos5"
    }
  ]
}
```

## E4 — quota accounting  ·  AMBIGUOUS, OVERSTATES BALANCE

**Previously:** cells_at_quota: 19

**Correction:** Exactly 17 cells equal quota and 2 exceed it, so 19 meet or exceed quota. 5 cells are short. Reporting '19 at quota' without the split reads as 19 cells having met their target exactly.

```json
{
  "cells_at_or_over_quota": 19,
  "cells_exactly_at_quota": 17,
  "cells_over_quota": 2,
  "cells_short": 5,
  "closeout_cells_at_quota_field": 19,
  "over_quota_detail": {
    "F0_target_side_stagger|right|center": 1,
    "F2_outer_panel_stagger|left|neg5": 1
  },
  "short_detail": {
    "F2_outer_panel_stagger|left|center": 1,
    "F3_aperture_side_stagger|left|pos5": 1,
    "F3_aperture_side_stagger|right|center": 1,
    "F3_aperture_side_stagger|right|neg5": 5,
    "F3_aperture_side_stagger|right|pos5": 5
  }
}
```

## E5 — splittability of the smallest cells  ·  OVERSTATED

**Previously:** the F3 right-side cells cannot be represented in both train and validation

**Correction:** Only F3_aperture_side_stagger|right|neg5, with one episode, is mathematically unable to appear in both. F3_aperture_side_stagger|right|pos5, with two, splits 1/1.

```json
{
  "F3_aperture_side_stagger|right|neg5": 1,
  "F3_aperture_side_stagger|right|pos5": 2
}
```

## E6 — encoder-health provenance  ·  UNREPRESENTATIVE SAMPLE

**Previously:** encoder health was reported as if it characterised the collection

**Correction:** Those statistics came from 120 windows drawn from a single real episode, not the full corpus. The exact corpus is 71,511 HDF5 timesteps and 2,860,440 sensor windows. Corpus-wide statistics are computed during V10.9 embedding generation and supersede them.

```json
{
  "corpus_sensor_windows": 2860440,
  "corpus_timesteps": 71511,
  "previously_reported_episodes": 1,
  "previously_reported_windows": 120,
  "sensors_per_episode": 40
}
```

## E7 — corpus dimensions  ·  CORRECT BUT EASILY MISREAD

**Previously:** episode_steps min 355, max 626 (control steps)

**Correction:** Those are control steps. HDF5 T = episode_steps + 1, so T ranges 356 to 627 and sums to 71,511. Anything sized against the HDF5 arrays (episode_horizon, padding, window counts) must use T.

```json
{
  "closeout_episode_steps_max": 626,
  "closeout_episode_steps_min": 355,
  "hdf5_t_max": 627,
  "hdf5_t_min": 356,
  "hdf5_t_sum": 71511
}
```
