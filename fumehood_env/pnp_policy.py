"""Placement planning that survives a short, cluttered fume-hood.

Upstream's ``PickAndPlacePlannerPolicy._get_placement_poses`` builds exactly one
preplace pose — grasp orientation, tray-centre XY, tray-top + ``place_z_offset``
— and raises if IK misses it. Inside a hood that fails on essentially every
episode: the hover height runs into the hood ceiling on short variants, and the
tray centre is a fixed XY that the arm may not reach with the grasp orientation
it was handed.

The fallback idiom here is upstream's own (``PickAndPlaceNextToPlannerPolicy``
lines 174-184 retry a lowered preplace before giving up); this generalises it to
a small search over hover height and in-tray offset. Every candidate still drops
the object *on* the tray: the tray is 0.18 m square, so offsets stay under 0.05 m.
"""
from __future__ import annotations

import logging

import numpy as np

from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)


class FumehoodPickAndPlacePlannerPolicy(PickAndPlacePlannerPolicy):
    """Same placement semantics as upstream, but searched instead of asserted."""

    HOVER_HEIGHTS = (0.07, 0.05, 0.035, 0.02)      # above the tray surface
    TRAY_OFFSETS = (0.0, -0.04, 0.04)              # along x, mouth-ward first
    TRAY_HALF_Z = 0.008                            # from the scene generator

    def _get_placement_poses(self, grasp_pose_world, pickup_obj, place_receptacle):
        data = self.task.env.current_data
        # body_aabb keeps only non-colliding geoms, and the tray inherits the
        # hood's contype/conaffinity - so it reports an empty box at the body
        # origin. The tray is a mocap body of known thickness; use that.
        receptacle_top_z = float(place_receptacle.position[2]) + self.TRAY_HALF_Z

        obj_center, obj_size = body_aabb(data.model, data, pickup_obj.object_id)
        obj_bottom_z = obj_center[2] - obj_size[2] / 2
        clearance = max(grasp_pose_world[2, 3] - obj_bottom_z, 0.0)
        # keeps the object (not the wrist) centred over the drop point
        ee_bias = grasp_pose_world[:3, 3] - pickup_obj.position

        candidates, stacked = [], []
        for dx in self.TRAY_OFFSETS:
            for hover in self.HOVER_HEIGHTS:
                preplace = grasp_pose_world.copy()
                preplace[:2, 3] = place_receptacle.position[:2]
                preplace[0, 3] += dx
                preplace[2, 3] = receptacle_top_z + clearance + hover
                preplace[:3, 3] += ee_bias

                place = preplace.copy()
                place[2, 3] = receptacle_top_z + clearance
                candidates.append((dx, hover, preplace, place))
                stacked += [preplace, place]

        cap = self.policy_config.grasp_feasibility_batch_size
        stacked = np.stack(stacked)
        mask = np.concatenate([
            np.atleast_1d(self.check_feasible_ik(stacked[i:i + cap]))
            for i in range(0, len(stacked), cap)
        ]).reshape(len(candidates), 2).all(axis=1)

        if not mask.any():
            raise ValueError(
                f"IK failed for every placement candidate ({len(candidates)} tried) "
                f"over tray at {np.round(place_receptacle.position, 3)}"
            )

        dx, hover, preplace, place = candidates[int(np.argmax(mask))]
        log.info("[FumehoodPnP] placement dx=%.3f hover=%.3f feasible=%d/%d",
                 dx, hover, int(mask.sum()), len(candidates))

        postplace = place.copy()
        postplace[:3, 3] -= self.policy_config.end_z_offset * postplace[:3, 2]
        return preplace, place, postplace


class FumehoodPickAndPlacePlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = FumehoodPickAndPlacePlannerPolicy
