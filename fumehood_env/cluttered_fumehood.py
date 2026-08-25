"""Cluttered, size-varying fume-hood pick and pick-and-place samplers.

Extends the invisible-obstacle pick with what the base sampler lacks:

* clutter standing on the bench, placed clear of the approach corridor so it is
  something the skin grazes rather than something that blocks the grasp;
* hood-shell obstacle boxes read back off the compiled model — the base sampler
  hardcodes the default hood's walls into the planner's obstacle list, which is
  wrong for every size variant;
* a reach depth that puts the object well inside the hood, so the whole arm has
  to enter rather than the gripper reaching just past the mouth;
* a pick-AND-place variant that sets the object down on the scene's tray.

Pairs with the fumehood_v*.xml scene variants emitted by gen_fumehood_variants.py.
"""
from __future__ import annotations

import mujoco
import numpy as np

from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.enclosure_reach import (
    SHELF_TOP_Z,
    TUBE_X0,
    InvisibleObstacleFumehoodPickSampler,
)

T_WALL = 0.012
PARK = (-3.0, 3.0, -2.0)

# (half_x, half_y, half_z) per clutter body, in the order the scene generator
# cycles its shapes.
CLUTTER_HALF = [
    (0.030, 0.030, 0.075),
    (0.032, 0.032, 0.055),
    (0.045, 0.045, 0.040),
    (0.022, 0.022, 0.090),
]


class ClutteredFumehoodPickSampler(InvisibleObstacleFumehoodPickSampler):
    N_CLUTTER = 12               # bodies present in the scene XML
    CLUTTER_N_RANGE = (0, 9)     # how many are actually placed, per episode
    CORRIDOR_R = 0.14            # keep-out half-width around the approach line
    OBJ_KEEPOUT = 0.12           # keep-out radius around the object itself
    REACH_SPAN = (0.06, 0.34)    # object depth past the mouth, metres
    # The tray is a mocap body, so it is posed per episode rather than left at
    # whatever the scene XML baked in. Upstream's placement pose is the grasp
    # orientation translated to the tray centre, which only solves if the tray
    # sits well inside the arm's workspace: keep it shallow (near the mouth) and
    # clear of the approach corridor.
    PLACE_PAD_NAME = "place_tray"
    TRAY_DEPTH = 0.10            # metres past the aperture plane
    TRAY_HALF_Z = 0.008          # tray geom half-height, from the scene generator

    # Half-extents the base sampler hardcodes for the default hood shell. They
    # are wrong for every size variant, so they are dropped and rebuilt from
    # the compiled model.
    _STALE_SHELL_HALVES = {(0.4, 0.012, 0.4), (0.012, 0.46, 0.4)}

    # ---------------- geometry read back from the compiled scene ----------------
    def _hood_dims(self, m):
        half_w = abs(float(m.geom("hood_side_l").pos[1])) - T_WALL
        depth = float(m.geom("hood_back").pos[0]) - T_WALL - TUBE_X0
        height = float(m.geom("hood_top").pos[2]) - 0.015 - SHELF_TOP_Z
        return half_w, depth, height

    @staticmethod
    def _shell_boxes(m):
        out = []
        for name in ("hood_side_l", "hood_side_r", "hood_back", "hood_top"):
            g = m.geom(name)
            out.append([list(map(float, g.pos)), list(map(float, g.size))])
        return out

    # ---------------- theta ----------------
    def _draw_theta(self):
        th = super()._draw_theta()
        th["reach_frac"] = float(np.random.uniform(0.0, 1.0))
        th["n_clutter"] = int(np.random.randint(self.CLUTTER_N_RANGE[0],
                                                self.CLUTTER_N_RANGE[1] + 1))
        th["clutter_uv"] = [[float(np.random.uniform(0.12, 0.92)),
                             float(np.random.uniform(-0.88, 0.88))]
                            for _ in range(self.N_CLUTTER)]
        return th

    def _obj_rest(self):
        x, y, z = super()._obj_rest()
        th = getattr(self, "_theta", None) or {}
        lo, hi = self.REACH_SPAN
        x = TUBE_X0 + lo + float(th.get("reach_frac", 0.4)) * (hi - lo)
        rest = (float(x), float(y), float(z))
        th["obj_rest"] = list(rest)   # remembered so clutter keeps clear of it
        return rest

    # ---------------- scene application ----------------
    def _park(self, env, i):
        self._mocap_set(env, f"cl_{i}", [PARK[0] - 0.2 * i, PARK[1], PARK[2]])

    def _tray_xy(self, half_w):
        return TUBE_X0 + self.TRAY_DEPTH, -max(min(half_w - 0.11, 0.16), 0.10)

    def _apply_theta(self, env, th):
        super()._apply_theta(env, th)
        m, d = env.current_model, env.current_data
        half_w, depth, height = self._hood_dims(m)
        th["hood_dims"] = [float(half_w), float(depth), float(height)]
        z0 = SHELF_TOP_Z

        obj = th.get("obj_rest") or [TUBE_X0 + 0.20, 0.0, z0]
        ox, oy = float(obj[0]), float(obj[1])

        placed, clutter = 0, []
        want = int(th.get("n_clutter", 0))
        for i in range(self.N_CLUTTER):
            hx, hy, hz = CLUTTER_HALF[i % len(CLUTTER_HALF)]
            if placed >= want:
                self._park(env, i)
                continue
            u, v = th["clutter_uv"][i]
            x = TUBE_X0 + 0.05 + u * max(depth - 0.10, 0.05)
            y = v * max(half_w - 0.06, 0.05)
            # The arm sweeps a corridor along y = oy from the mouth to the
            # object; anything inside it would block the grasp rather than be
            # skimmed by the skin.
            in_corridor = abs(y - oy) < self.CORRIDOR_R and TUBE_X0 - 0.05 <= x <= ox + 0.05
            near_obj = float(np.hypot(x - ox, y - oy)) < self.OBJ_KEEPOUT
            near_bar = False
            if th.get("protrusion_present") and th.get("protr_center"):
                pc = th["protr_center"]
                near_bar = float(np.hypot(x - pc[0], y - pc[1])) < 0.08
            # keep the place tray clear too (front-right corner in every scene),
            # otherwise pick-and-place episodes can start with clutter on the pad
            tray_x, tray_y = self._tray_xy(half_w)
            near_tray = float(np.hypot(x - tray_x, y - tray_y)) < 0.16
            if in_corridor or near_obj or near_bar or near_tray:
                self._park(env, i)
                continue
            pos = [float(x), float(y), float(z0 + hz)]
            self._mocap_set(env, f"cl_{i}", pos)
            clutter.append([pos, [hx, hy, hz]])
            placed += 1

        tray_x, tray_y = self._tray_xy(half_w)
        self._mocap_set(env, self.PLACE_PAD_NAME,
                        [tray_x, tray_y, z0 + self.TRAY_HALF_Z])
        th["tray_xy"] = [float(tray_x), float(tray_y)]

        th["n_clutter_placed"] = placed
        th["clutter_aabbs"] = clutter

        # Replace the base sampler's hardcoded hood shell with this scene's
        # real walls, then hand the planner the clutter as well.
        kept = [b for b in th.get("obstacle_aabbs", [])
                if tuple(round(float(v), 4) for v in b[1]) not in self._STALE_SHELL_HALVES]
        th["obstacle_aabbs"] = self._shell_boxes(m) + kept + clutter

        mujoco.mj_forward(m, d)


class ClutteredFumehoodPickAndPlaceSampler(
    ClutteredFumehoodPickSampler, PickAndPlaceTaskSampler
):
    """Fume-hood pick-AND-place: reach in, grasp, set the object on the tray.

    Both parents descend from PickTaskSampler, so the MRO runs the enclosure
    logic (hood posing, clutter, obstacle bar) and then reaches
    PickAndPlaceTaskSampler, which builds a PickAndPlaceTask. Explicit
    delegation the way FridgePickAndPlaceTaskSampler does it is not an option
    here: the enclosure methods call super(), which requires the instance to be
    of their own class.

    The receptacle hooks are neutralised — PickAndPlaceTaskSampler normally
    spawns receptacles from the object database and needs annotated surfaces to
    place them, while this scene already carries a `place_tray` body.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.place_receptacle_name = self.PLACE_PAD_NAME
        self._receptacle_names = [self.PLACE_PAD_NAME]

    def _seed_referral_expressions(self) -> None:
        """This MRO bypasses PickTaskSampler._sample_task, which is where the
        pick chain normally fills referral_expressions — downstream task
        descriptions then KeyError on 'pickup_obj_name'. Seed the keys with the
        enclosure task's fixed object (the red cup) before any task exists."""
        tc = self.config.task_config
        if tc.referral_expressions is None:
            tc.referral_expressions = {}
        tc.referral_expressions.setdefault("pickup_obj_name", "red cup")
        tc.referral_expressions.setdefault("pickup_name", "red cup")
        tc.referral_expressions.setdefault("place_name", "tray")

    def _settle_injected_object(self, env) -> None:
        self._seed_referral_expressions()
        super()._settle_injected_object(env)

    def _add_receptacles_to_scene(self, spec) -> None:
        return None   # the tray is part of the scene XML

    def _get_place_target_candidates(self, env, pickup_obj_name, supporting_geom_id):
        self.place_receptacle_name = self.PLACE_PAD_NAME
        return [self.PLACE_PAD_NAME]

    def _prepare_place_target(self, env, place_target_name, pickup_obj_name,
                              pickup_obj_pos, supporting_geom_id) -> bool:
        self.place_receptacle_name = self.PLACE_PAD_NAME
        om = env.object_managers[env.current_batch_index]
        return om.get_object_by_name(self.PLACE_PAD_NAME) is not None

    def _generate_referral_expressions(self, env, object_name, context_objects):
        if object_name == self.PLACE_PAD_NAME:
            priority = [(1.0, 1.0, "blue tray")]
            return priority, priority
        return super()._generate_referral_expressions(env, object_name, context_objects)

    def _sample_task(self, env):
        self.place_receptacle_name = self.PLACE_PAD_NAME
        self._seed_referral_expressions()
        return super()._sample_task(env)

    def _sample_and_place_robot(self, env) -> None:
        self._seed_referral_expressions()
        super()._sample_and_place_robot(env)
        tc = self.config.task_config
        tc.place_receptacle_name = self.PLACE_PAD_NAME
        tc.place_target_name = self.PLACE_PAD_NAME
