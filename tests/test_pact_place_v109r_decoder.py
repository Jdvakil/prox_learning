"""Behavioural tests for the V10.9R event-based action-chunk decoder.

Each test states a property the repair must have, and is written so that
reverting to the averaging decoder would fail it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "submodules" / "act", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v109r_decoder import (  # noqa: E402
    ARM_SLEW_LIMIT_L2,
    CHUNK_SIZE,
    GRIPPER_THRESHOLD,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_RELEASED,
    EventChunkDecoder,
    LegacyAverageDecoder,
    weighted_median,
)


def chunk(gripper, arm=0.0, size=CHUNK_SIZE):
    """A (size, 8) chunk. ``gripper`` may be a scalar or a per-row sequence."""
    out = np.full((size, 8), float(arm), dtype=np.float32)
    out[:, 7] = gripper
    return out


def open_chunk(size=CHUNK_SIZE):
    return chunk(0.0, size=size)


def close_at(offset, size=CHUNK_SIZE, arm=0.0):
    g = np.zeros(size, dtype=np.float32)
    g[offset:] = 255.0
    return chunk(g, arm=arm, size=size)


# --------------------------------------------------------------------- #
# 1. raw-chunk recording
# --------------------------------------------------------------------- #
def test_every_raw_chunk_is_retained_with_start_step_and_hash():
    d = EventChunkDecoder()
    for step in range(5):
        d.observe(step, open_chunk())
    raw = d.telemetry.raw_chunks
    assert len(raw) == 5
    assert [r["start_step"] for r in raw] == [0, 1, 2, 3, 4]
    assert all(len(r["sha256"]) == 64 for r in raw)
    assert all(r["shape"] == [CHUNK_SIZE, 8] for r in raw)


def test_raw_chunk_hash_is_content_addressed():
    d = EventChunkDecoder()
    d.observe(0, open_chunk())
    d.observe(1, open_chunk())
    d.observe(2, close_at(0))
    hashes = [r["sha256"] for r in d.telemetry.raw_chunks]
    assert hashes[0] == hashes[1], "identical chunks must hash identically"
    assert hashes[2] != hashes[0], "a different chunk must hash differently"


def test_raw_chunks_are_recorded_before_aggregation_not_after():
    """The retained chunk is the raw prediction, not the decoded action."""
    d = EventChunkDecoder()
    d.observe(0, close_at(50))
    assert d.telemetry.raw_chunks[0]["phase_at_emission"] == STATE_OPEN
    assert d.state == STATE_OPEN  # a crossing 50 steps out must not fire now


# --------------------------------------------------------------------- #
# 2. absolute event timing
# --------------------------------------------------------------------- #
def test_event_offset_is_converted_to_absolute_episode_time():
    d = EventChunkDecoder()
    d.observe(0, close_at(10))       # predicts closure at absolute step 10
    for step in range(1, 10):
        d.observe(step, close_at(max(0, 10 - step)))
        assert d.state == STATE_OPEN, f"fired early at {step}"
    d.observe(10, close_at(0))
    assert d.state == STATE_CLOSED


def test_a_stale_chunk_keeps_its_original_absolute_time():
    """A chunk emitted at step 0 predicting offset 20 means step 20 forever."""
    d = EventChunkDecoder()
    d.observe(0, close_at(20))
    for step in range(1, 20):
        d.observe(step, open_chunk())
    assert d.state == STATE_OPEN
    ev = d._decode_event(19)
    assert ev["event_time"] == 20.0


# --------------------------------------------------------------------- #
# 3. missing votes
# --------------------------------------------------------------------- #
def test_no_votes_means_no_transition():
    d = EventChunkDecoder()
    for step in range(30):
        d.observe(step, open_chunk())
    assert d.state == STATE_OPEN
    assert d.telemetry.events == []


def test_a_single_vote_among_many_abstainers_does_not_fire():
    d = EventChunkDecoder()
    for step in range(20):
        d.observe(step, open_chunk())
    d.observe(20, close_at(0))     # one lone chunk says close now
    assert d.state == STATE_OPEN, "a single vote must not carry the ensemble"


def test_decoder_handles_a_chunk_shorter_than_the_horizon():
    d = EventChunkDecoder()
    d.observe(0, close_at(0, size=3))
    assert d.state == STATE_CLOSED


# --------------------------------------------------------------------- #
# 4. weighted consensus
# --------------------------------------------------------------------- #
def test_weighted_median_is_deterministic_and_handles_ties():
    v = np.array([5.0, 1.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])
    assert weighted_median(v, w) == 3.0
    assert weighted_median(v, w) == weighted_median(v, w)
    assert weighted_median(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == 2.0


def test_weighted_median_follows_the_weight_not_the_count():
    v = np.array([1.0, 100.0])
    assert weighted_median(v, np.array([9.0, 1.0])) == 1.0
    assert weighted_median(v, np.array([1.0, 9.0])) == 100.0


def test_majority_of_live_weight_is_required():
    d = EventChunkDecoder(consensus_weight_fraction=0.9)
    for step in range(10):
        d.observe(step, open_chunk())
    d.observe(10, close_at(0))
    assert d.state == STATE_OPEN
    lenient = EventChunkDecoder(consensus_weight_fraction=0.0)
    for step in range(10):
        lenient.observe(step, open_chunk())
    lenient.observe(10, close_at(0))
    assert lenient.state == STATE_CLOSED


def test_empty_weighted_median_is_refused():
    with pytest.raises(ValueError):
        weighted_median(np.array([]), np.array([]))


# --------------------------------------------------------------------- #
# 5. close and release transitions
# --------------------------------------------------------------------- #
def test_close_then_release_is_decoded_as_two_separate_events():
    d = EventChunkDecoder()
    for step in range(6):
        d.observe(step, close_at(0))
    assert d.state == STATE_CLOSED
    for step in range(6, 20):
        d.observe(step, open_chunk())
    assert d.state == STATE_RELEASED
    kinds = [e["to_state"] for e in d.telemetry.events]
    assert kinds == [STATE_CLOSED, STATE_RELEASED]


def test_closure_does_not_latch_permanently():
    d = EventChunkDecoder()
    for step in range(6):
        d.observe(step, close_at(0))
    assert d.state == STATE_CLOSED
    for step in range(6, 30):
        d.observe(step, open_chunk())
    assert d.state == STATE_RELEASED, "closure must not latch"


def test_release_requires_its_own_decoded_crossing():
    """Staying closed forever yields no release, and that is reported."""
    d = EventChunkDecoder()
    for step in range(40):
        d.observe(step, close_at(0))
    assert d.state == STATE_CLOSED
    assert d.summary()["closure_without_release"] is True
    assert d.summary()["release_step"] is None


def test_released_is_terminal():
    d = EventChunkDecoder()
    for step in range(6):
        d.observe(step, close_at(0))
    for step in range(6, 20):
        d.observe(step, open_chunk())
    assert d.state == STATE_RELEASED
    for step in range(20, 40):
        d.observe(step, close_at(0))
    assert d.state == STATE_RELEASED, "a released gripper must not re-close"


def test_gripper_output_is_driven_by_state_not_by_an_average():
    d = EventChunkDecoder()
    out = [d.observe(s, close_at(0))[7] for s in range(6)]
    assert out[0] == 255.0, "a unanimous close-now must command close immediately"
    assert all(v == 255.0 for v in out)


# --------------------------------------------------------------------- #
# 6. phase filtering
# --------------------------------------------------------------------- #
def test_pregrasp_arm_plans_are_not_averaged_with_postgrasp_plans():
    d = EventChunkDecoder()
    pregrasp = chunk(0.0, arm=0.0)        # open plan, arm at 0
    postgrasp = chunk(255.0, arm=1.0)     # closed plan, arm at 1
    d.observe(0, pregrasp)
    d.observe(1, pregrasp)
    action = d.observe(2, postgrasp)
    if d.state == STATE_CLOSED:
        assert np.allclose(action[:7], 1.0), \
            "once closed, only postgrasp chunks may contribute to the arm"


def test_incompatible_chunks_are_discarded_and_counted():
    d = EventChunkDecoder()
    for step in range(4):
        d.observe(step, chunk(255.0, arm=1.0))
    assert d.state == STATE_CLOSED
    d.observe(4, chunk(0.0, arm=0.0))
    assert d.telemetry.discarded_incompatible > 0


def test_falls_back_to_the_freshest_plan_when_none_are_compatible():
    d = EventChunkDecoder()
    for step in range(4):
        d.observe(step, chunk(255.0, arm=1.0))
    assert d.state == STATE_CLOSED
    for step in range(4, 12):
        d.observe(step, chunk(0.0, arm=0.0))
    assert d.state == STATE_RELEASED
    assert np.all(np.isfinite(d.observe(12, chunk(0.0, arm=0.0))))


# --------------------------------------------------------------------- #
# 7. buffer reset
# --------------------------------------------------------------------- #
def test_reset_clears_state_buffer_and_telemetry():
    d = EventChunkDecoder()
    for step in range(6):
        d.observe(step, close_at(0))
    assert d.state == STATE_CLOSED
    d.reset()
    assert d.state == STATE_OPEN
    assert d._records == []
    assert d.telemetry.raw_chunks == []
    assert d.telemetry.events == []
    assert d._previous_arm is None


def test_reset_makes_a_second_episode_independent_of_the_first():
    d = EventChunkDecoder()
    first = [d.observe(s, close_at(0))[7] for s in range(5)]
    d.reset()
    second = [d.observe(s, close_at(0))[7] for s in range(5)]
    assert first == second


def test_expired_chunks_leave_the_buffer():
    d = EventChunkDecoder()
    d.observe(0, open_chunk(size=5))
    for step in range(1, 12):
        d.observe(step, open_chunk(size=5))
    assert all(step - r.start < len(r.chunk) for r in d._records for step in [11])


# --------------------------------------------------------------------- #
# 8. deterministic output
# --------------------------------------------------------------------- #
def test_identical_input_yields_bit_identical_output():
    rng = np.random.default_rng(20260830)
    chunks = [rng.normal(size=(CHUNK_SIZE, 8)).astype(np.float32) * 50 for _ in range(30)]

    def run():
        d = EventChunkDecoder()
        return np.stack([d.observe(i, c) for i, c in enumerate(chunks)])

    a, b = run(), run()
    assert np.array_equal(a, b)


def test_decoder_does_not_mutate_the_caller_s_chunk():
    d = EventChunkDecoder()
    c = close_at(0)
    before = c.copy()
    d.observe(0, c)
    assert np.array_equal(c, before)


# --------------------------------------------------------------------- #
# 9. restoration on exceptions
# --------------------------------------------------------------------- #
def test_a_malformed_chunk_is_refused_without_corrupting_state():
    d = EventChunkDecoder()
    for step in range(4):
        d.observe(step, close_at(0))
    state, records, events = d.state, len(d._records), len(d.telemetry.events)
    for bad in (np.zeros((CHUNK_SIZE, 7), dtype=np.float32),
                np.zeros(CHUNK_SIZE, dtype=np.float32),
                np.zeros((2, CHUNK_SIZE, 8), dtype=np.float32)):
        with pytest.raises(ValueError):
            d.observe(4, bad)
    assert d.state == state
    assert len(d._records) == records
    assert len(d.telemetry.events) == events
    assert np.all(np.isfinite(d.observe(4, close_at(0))))


def test_state_survives_a_refused_chunk_and_continues_correctly():
    d = EventChunkDecoder()
    for step in range(6):
        d.observe(step, close_at(0))
    with pytest.raises(ValueError):
        d.observe(6, np.zeros((CHUNK_SIZE, 3), dtype=np.float32))
    for step in range(6, 24):
        d.observe(step, open_chunk())
    assert d.state == STATE_RELEASED


# --------------------------------------------------------------------- #
# 10. action continuity
# --------------------------------------------------------------------- #
def test_a_phase_change_cannot_produce_a_step_larger_than_training_ever_saw():
    d = EventChunkDecoder()
    for step in range(4):
        d.observe(step, chunk(0.0, arm=0.0))
    previous = d.observe(4, chunk(0.0, arm=0.0))[:7]
    for step in range(5, 12):
        action = d.observe(step, chunk(255.0, arm=1000.0))
        assert np.linalg.norm(action[:7] - previous) <= ARM_SLEW_LIMIT_L2 + 1e-5
        previous = action[:7]


def test_slew_limit_preserves_direction():
    d = EventChunkDecoder()
    d.observe(0, chunk(0.0, arm=0.0))
    first = d.observe(1, chunk(0.0, arm=0.0))[:7]
    second = d.observe(2, chunk(0.0, arm=500.0))[:7]
    delta = second - first
    assert np.linalg.norm(delta) <= ARM_SLEW_LIMIT_L2 + 1e-5
    assert np.all(delta >= 0), "clamping must not reverse the commanded direction"


def test_normal_motion_is_never_clamped():
    """A trajectory within the training envelope must pass through untouched."""
    rng = np.random.default_rng(7)
    d = EventChunkDecoder()
    arm = 0.0
    for step in range(60):
        arm += rng.normal(scale=0.001)
        d.observe(step, chunk(0.0, arm=arm))
    assert d.telemetry.slew_clamped_steps == 0
    assert d.telemetry.max_arm_step_delta < ARM_SLEW_LIMIT_L2


def test_max_arm_step_delta_is_reported():
    d = EventChunkDecoder()
    for step in range(5):
        d.observe(step, chunk(0.0, arm=0.0))
    assert "max_arm_step_delta" in d.summary()


# --------------------------------------------------------------------- #
# the defect itself: the legacy decoder must actually be late
# --------------------------------------------------------------------- #
def ramp_crossing_at(offset, size=CHUNK_SIZE, width=40, arm=0.0):
    """A chunk whose gripper ramps smoothly and crosses threshold at ``offset``.

    Real gripper channels are continuous, not binary. The ramp is what makes the
    two decoders genuinely different: the crossing time is a property of the
    curve, but the *value* at any given row depends on how gradual the ramp is.
    """
    g = np.clip((np.arange(size) - offset) / width + 0.5, 0.0, 1.0) * 255.0
    return chunk(g.astype(np.float32), arm=arm, size=size)


def test_unanimous_chunks_make_the_two_decoders_agree():
    """Recorded so the repair is not credited with more than it does.

    When every live chunk asserts the same crossing time, averaging the values
    and taking the median of the times fire at the same step, whatever the ramp
    shape. The event decoder's advantage is not in the unanimous case.
    """
    for width in (1, 10, 40, 80):
        legacy, event = LegacyAverageDecoder(), EventChunkDecoder()
        for step in range(60):
            c = ramp_crossing_at(20 - step, width=width)
            legacy.observe(step, c)
            event.observe(step, c)
        assert event.summary()["close_step"] == 20
        assert legacy.summary()["close_step"] == 20


def test_a_single_extreme_chunk_cannot_force_a_premature_closure():
    """Where the event decoder is genuinely safer.

    Gripper outputs are unnormalized regression, not probabilities, so one
    chunk can predict a wildly out-of-range value. That single outlier drags the
    average over threshold and closes the gripper. A weighted majority of
    crossing *times* is unmoved by it -- which is exactly the "premature
    closure" this task lists as a stop condition.
    """
    legacy, event = LegacyAverageDecoder(), EventChunkDecoder()
    for step in range(10):
        legacy.observe(step, open_chunk())
        event.observe(step, open_chunk())
    outlier = chunk(2000.0)
    legacy_action = legacy.observe(10, outlier)
    event_action = event.observe(10, outlier)
    assert legacy_action[7] == 255.0, "the averaging decoder is fooled by the outlier"
    assert event_action[7] == 0.0, "the event decoder must not be"
    assert event.state == STATE_OPEN


def test_the_event_decoder_makes_each_transition_at_most_once():
    """The legacy decoder re-thresholds every step and can reopen; the state
    machine cannot go backwards."""
    event = EventChunkDecoder()
    for step in range(6):
        event.observe(step, close_at(0))
    assert event.state == STATE_CLOSED
    for step in range(6, 26):
        event.observe(step, open_chunk())
    assert event.state == STATE_RELEASED
    for step in range(26, 60):
        event.observe(step, close_at(0) if step % 2 else open_chunk())
    to_states = [e["to_state"] for e in event.telemetry.events]
    assert to_states == [STATE_CLOSED, STATE_RELEASED]

    legacy = LegacyAverageDecoder()
    grips = []
    for step in range(60):
        grips.append(legacy.observe(step, close_at(0) if step % 2 else open_chunk())[7])
    assert len(set(grips)) > 1, "the legacy decoder is free to flip repeatedly"


def test_both_decoders_agree_when_every_chunk_is_unanimous():
    legacy, event = LegacyAverageDecoder(), EventChunkDecoder()
    out_l, out_e = [], []
    for step in range(20):
        c = chunk(255.0, arm=0.25)
        out_l.append(legacy.observe(step, c))
        out_e.append(event.observe(step, c))
    assert np.allclose(np.stack(out_l)[:, 7], np.stack(out_e)[:, 7])
    assert np.allclose(np.stack(out_l)[:, :7], np.stack(out_e)[:, :7], atol=1e-5)
