"""Behavioral tests for V10.6: asymmetric static pendant on V9.5 real clutter.

Every expectation is recomputed from source, from live MuJoCo state, or from
file bytes. The preregistered admission rule is exercised directly, including
the cases it must refuse.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_v106_contract import (  # noqa: E402
    INTRUSION_SIDES,
    N_EVALUATIONS_PER_BUNDLE,
    N_REVIEW_ROWS,
    N_REVIEW_VIDEOS,
    PHASE0_MIN_CLEAN,
    POOL_MIN_CLEAN,
    POOL_MIN_CLEAN_PER_POSE,
    POOL_MIN_CLEAN_PER_SIDE,
    POOL_MIN_CLEAN_PER_SIDE_POSE,
    admit_candidate,
    build_specification_contract,
    phase0_rows,
    pool_eligibility,
    review_rows,
    streams_are_disjoint,
)
from pact_place_v106_geometry import (  # noqa: E402
    ALL_GEOMS_V106,
    CLEARANCE_FLOOR_M,
    FALLBACK_ABSOLUTE_MIN_CLEARANCE_M,
    FALLBACK_MIN_FRACTION_GE_FLOOR,
    FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR,
    LATTICE_R_NEG_M,
    LATTICE_R_POS_M,
    LATTICE_X_M,
    PENDANT_BODY_V106,
    POSE_IDS,
    POSE_OFFSETS_M,
    build_assembly,
    lattice_candidates,
    scene_xml_text,
)


class AsymmetricGeometryTest(unittest.TestCase):
    def test_registered_lattice(self):
        self.assertEqual(LATTICE_X_M, (0.800,))
        self.assertEqual(LATTICE_R_NEG_M, (0.325, 0.330, 0.335))
        self.assertEqual(LATTICE_R_POS_M, (0.295, 0.300, 0.305))
        self.assertEqual(len(lattice_candidates()), 9)
        self.assertEqual(len(lattice_candidates()) * len(POSE_IDS), 27)

    def test_lobes_take_independent_radii(self):
        a = build_assembly(0.800, 0.335, 0.295, 0.0, pose_id="center")
        by = {i["name"]: i for i in a["components"]}
        self.assertAlmostEqual(by["lobe_0"]["center_m"][1], -0.335, places=9)
        self.assertAlmostEqual(by["lobe_1"]["center_m"][1], +0.295, places=9)
        self.assertTrue(a["asymmetric"])

    def test_crossbar_spans_the_actual_asymmetric_stem_endpoints(self):
        for rn in LATTICE_R_NEG_M:
            for rp in LATTICE_R_POS_M:
                for pose in POSE_IDS:
                    a = build_assembly(0.800, rn, rp, POSE_OFFSETS_M[pose],
                                       pose_id=pose)
                    by = {i["name"]: i for i in a["components"]}
                    bar = by["crossbar"]
                    lo = bar["center_m"][1] - bar["half_m"][1]
                    hi = bar["center_m"][1] + bar["half_m"][1]
                    for slot in (0, 1):
                        stem = by[f"stem_{slot}"]
                        s_lo = stem["center_m"][1] - stem["half_m"][1]
                        s_hi = stem["center_m"][1] + stem["half_m"][1]
                        self.assertLessEqual(lo, s_lo + 1e-9)
                        self.assertGreaterEqual(hi, s_hi - 1e-9)
                    self.assertAlmostEqual(
                        lo, by["stem_0"]["center_m"][1] - 0.006, places=9
                    )
                    self.assertAlmostEqual(
                        hi, by["stem_1"]["center_m"][1] + 0.006, places=9
                    )

    def test_crossbar_is_not_centred_on_d_when_asymmetric(self):
        a = build_assembly(0.800, 0.335, 0.295, 0.0, pose_id="center")
        bar = next(i for i in a["components"] if i["name"] == "crossbar")
        self.assertNotAlmostEqual(bar["center_m"][1], 0.0, places=6)

    def test_a_disconnected_crossbar_would_be_refused(self):
        """Guard the connectivity assertion itself."""
        import pact_place_v106_geometry as geom

        original = geom.component_specs

        def broken(x, rn, rp, d):
            specs = list(original(x, rn, rp, d))
            bar = dict(specs[4])
            bar["half_m"] = (bar["half_m"][0], 0.05, bar["half_m"][2])
            specs[4] = bar
            return tuple(specs)

        geom.component_specs = broken
        try:
            with self.assertRaises(ValueError):
                geom.build_assembly(0.800, 0.335, 0.295, 0.0, pose_id="center")
        finally:
            geom.component_specs = original

    def test_height_and_shape_are_frozen(self):
        for x, rn, rp in lattice_candidates():
            a = build_assembly(x, rn, rp, 0.0, pose_id="center")
            self.assertAlmostEqual(a["lobe_bottom_z_m"], 0.98, places=9)
            self.assertAlmostEqual(a["lobe_top_z_m"], 1.04, places=9)
            self.assertAlmostEqual(a["crossbar_top_z_m"], 1.515, places=9)

    def test_scene_has_no_joint_freejoint_mocap_or_actuator(self):
        text = scene_xml_text(build_assembly(0.800, 0.330, 0.300, 0.0,
                                             pose_id="center"))
        body = text[text.index(f'<body name="{PENDANT_BODY_V106}"'):]
        body = body[: body.index("</body>")]
        for forbidden in ("<joint", "<freejoint", "mocap=", "<actuator"):
            self.assertNotIn(forbidden, body)

    def test_five_geoms_visible_equals_collision(self):
        a = build_assembly(0.800, 0.330, 0.300, 0.0, pose_id="center")
        self.assertEqual(len(a["components"]), 5)
        text = scene_xml_text(a)
        for geom in ALL_GEOMS_V106:
            self.assertEqual(text.count(f'name="{geom}"'), 1)

    def test_no_runtime_model_writes_in_v106_sources(self):
        import ast

        for name in ("scripts/pact_place_v106_geometry.py",
                     "scripts/pact_place_v106_contract.py",
                     "scripts/run_pact_place_v106_siting.py"):
            tree = ast.parse((ROOT / name).read_text())
            for node in ast.walk(tree):
                targets = (
                    list(node.targets) if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
                )
                for target in targets:
                    inner = target
                    while isinstance(inner, ast.Subscript):
                        inner = inner.value
                    if isinstance(inner, ast.Attribute):
                        self.assertNotIn(
                            inner.attr,
                            ("geom_pos", "geom_size", "geom_aabb", "geom_rbound",
                             "bvh_aabb", "body_pos", "mocap_pos"),
                            f"{name} writes {inner.attr}",
                        )


class CompiledV106SceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

    def _compile(self, assembly):
        import shutil

        import mujoco

        scenes = ROOT / (
            "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes"
        )
        scratch = Path(tempfile.mkdtemp(prefix="v106_scene_"))
        for name in ("pact_place_corridor_v3.xml", "pact_place_corridor_v5.xml"):
            shutil.copyfile(scenes / name, scratch / name)
        path = scratch / "probe.xml"
        path.write_text(scene_xml_text(assembly))
        try:
            return mujoco.MjModel.from_xml_path(str(path))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_compiled_static_with_enclosing_bounds(self):
        a = build_assembly(0.800, 0.335, 0.295, -0.005, pose_id="neg5")
        model = self._compile(a)
        body_id = int(model.body(PENDANT_BODY_V106).id)
        self.assertEqual(int(model.body_dofnum[body_id]), 0)
        self.assertEqual(int(model.body_jntnum[body_id]), 0)
        self.assertLess(int(model.body_mocapid[body_id]), 0)
        for item in a["components"]:
            gid = int(model.geom(item["geom"]).id)
            half = np.asarray(item["half_m"], dtype=float)
            self.assertTrue(np.allclose(model.geom_size[gid], half, atol=1e-9))
            self.assertTrue(np.allclose(
                model.geom_pos[gid], np.asarray(item["center_m"]), atol=1e-9))
            aabb = np.asarray(model.geom_aabb[gid], dtype=float)
            self.assertTrue(np.all(aabb[3:] >= half - 1e-12))
            self.assertGreaterEqual(
                float(model.geom_rbound[gid]),
                float(np.linalg.norm(half)) - 1e-9,
            )
            self.assertNotEqual(
                (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])),
                (0, 0),
            )


class AdmissionRuleTest(unittest.TestCase):
    def _stats(self, **over):
        base = {
            "x_m": 0.8, "r_neg_m": 0.33, "r_pos_m": 0.30,
            "n_evaluations": 294,
            "absolute_min_clearance_m": 0.016,
            "n_below_floor": 0, "n_contacts": 0,
            "fraction_ge_floor": 1.0,
            "band_evaluations_by_group": {
                f"{p}|{s}": 5 for p in POSE_IDS for s in INTRUSION_SIDES
            },
            "evaluations_ge_floor_by_group": {
                f"{p}|{s}": {"n": 49, "n_ge_floor": 49}
                for p in POSE_IDS for s in INTRUSION_SIDES
            },
            "direction_band_witnesses": {
                "left": ["loaded_outbound"], "right": ["loaded_outbound"],
            },
            "n_window_below_floor": 0, "n_initial_below_floor": 0,
        }
        base.update(over)
        return base

    def test_universal_clearance_is_admitted_and_labelled(self):
        d = admit_candidate(self._stats())
        self.assertTrue(d["universal_clearance"])
        self.assertTrue(d["admitted"])
        self.assertEqual(d["admission_basis"], "universal_clearance")

    def test_inbound_risk_is_not_required(self):
        d = admit_candidate(self._stats())
        self.assertFalse(d["inbound_risk_required"])
        self.assertTrue(d["fallback_checks"]["loaded_outbound_risk_on_both_sides"])

    def test_fallback_admits_a_near_miss(self):
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.011, n_below_floor=20,
            fraction_ge_floor=(294 - 20) / 294,
            evaluations_ge_floor_by_group={
                f"{p}|{s}": {"n": 49, "n_ge_floor": 45}
                for p in POSE_IDS for s in INTRUSION_SIDES
            },
        ))
        self.assertFalse(d["universal_clearance"])
        self.assertTrue(d["admitted"])
        self.assertEqual(d["admission_basis"], "preregistered_fallback")

    def test_any_contact_refuses(self):
        d = admit_candidate(self._stats(n_contacts=1))
        self.assertFalse(d["admitted"])
        self.assertFalse(d["fallback_checks"]["zero_exact_contacts"])

    def test_below_ten_millimetres_refuses(self):
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.009, n_below_floor=10,
            fraction_ge_floor=(294 - 10) / 294))
        self.assertFalse(d["admitted"])
        self.assertFalse(d["fallback_checks"]["absolute_min_at_least_10mm"])

    def test_below_ninety_percent_refuses(self):
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.011, n_below_floor=40,
            fraction_ge_floor=(294 - 40) / 294))
        self.assertFalse(d["admitted"])
        self.assertFalse(
            d["fallback_checks"]["at_least_90pct_evaluations_ge_15mm"])

    def test_one_group_below_eighty_percent_refuses(self):
        groups = {
            f"{p}|{s}": {"n": 49, "n_ge_floor": 49}
            for p in POSE_IDS for s in INTRUSION_SIDES
        }
        groups["center|left"] = {"n": 49, "n_ge_floor": 38}   # 77.6%
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.011, n_below_floor=11,
            fraction_ge_floor=(294 - 11) / 294,
            evaluations_ge_floor_by_group=groups))
        self.assertFalse(d["admitted"])
        self.assertFalse(
            d["fallback_checks"]["every_group_at_least_80pct_ge_15mm"])

    def test_window_or_initial_violation_refuses(self):
        for field in ("n_window_below_floor", "n_initial_below_floor"):
            with self.subTest(field=field):
                d = admit_candidate(self._stats(
                    absolute_min_clearance_m=0.011, n_below_floor=5,
                    fraction_ge_floor=(294 - 5) / 294, **{field: 1}))
                self.assertFalse(d["admitted"])

    def test_missing_band_witness_in_one_group_refuses(self):
        band = {f"{p}|{s}": 5 for p in POSE_IDS for s in INTRUSION_SIDES}
        band["pos5|right"] = 0
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.011, n_below_floor=5,
            fraction_ge_floor=(294 - 5) / 294,
            band_evaluations_by_group=band))
        self.assertFalse(d["admitted"])

    def test_missing_loaded_outbound_on_one_side_refuses(self):
        d = admit_candidate(self._stats(
            absolute_min_clearance_m=0.011, n_below_floor=5,
            fraction_ge_floor=(294 - 5) / 294,
            direction_band_witnesses={"left": ["loaded_outbound"],
                                      "right": ["inbound"]}))
        self.assertFalse(d["admitted"])

    def test_inbound_only_does_not_rescue_a_missing_loaded_outbound(self):
        d = admit_candidate(self._stats(
            direction_band_witnesses={"left": ["inbound"],
                                      "right": ["inbound"]}))
        self.assertFalse(
            d["fallback_checks"]["loaded_outbound_risk_on_both_sides"])

    def test_fallback_thresholds_are_the_registered_values(self):
        self.assertEqual(FALLBACK_ABSOLUTE_MIN_CLEARANCE_M, 0.010)
        self.assertEqual(FALLBACK_MIN_FRACTION_GE_FLOOR, 0.90)
        self.assertEqual(FALLBACK_MIN_GROUP_FRACTION_GE_FLOOR, 0.80)
        self.assertEqual(CLEARANCE_FLOOR_M, 0.015)

    def test_evaluations_per_bundle_identity(self):
        self.assertEqual(N_EVALUATIONS_PER_BUNDLE, 98 * 3)
        self.assertEqual(N_EVALUATIONS_PER_BUNDLE, 294)


class PoolFloorTest(unittest.TestCase):
    def _pool(self, clean_flags):
        rows = review_rows()
        results = [
            {"role_index": r["role_index"], "v106_clean_success": bool(c)}
            for r, c in zip(rows, clean_flags)
        ]
        return rows, results

    def test_floors_are_the_scaled_values(self):
        self.assertEqual((POOL_MIN_CLEAN, N_REVIEW_ROWS), (32, 48))
        self.assertEqual(POOL_MIN_CLEAN_PER_SIDE, 14)
        self.assertEqual(POOL_MIN_CLEAN_PER_POSE, 8)
        self.assertEqual(POOL_MIN_CLEAN_PER_SIDE_POSE, 4)
        self.assertEqual(PHASE0_MIN_CLEAN, 16)

    def test_all_clean_passes(self):
        rows, results = self._pool([True] * 48)
        report = pool_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 48)
        self.assertTrue(report["pool_passed"])

    def test_thirty_one_clean_fails(self):
        rows, results = self._pool([True] * 31 + [False] * 17)
        report = pool_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 31)
        self.assertFalse(report["pool_passed"])

    def test_thirty_two_clean_can_still_fail_a_balance_floor(self):
        rows = review_rows()
        flags = [r["intrusion_side"] == "right" for r in rows]  # 24 right clean
        added = 0
        for i, r in enumerate(rows):
            if not flags[i] and added < 8:
                flags[i] = True
                added += 1
        _, results = self._pool(flags)
        report = pool_eligibility(rows, results)
        self.assertEqual(report["clean_successes"], 32)
        self.assertFalse(report["pool_passed"])
        self.assertTrue(report["limiting_predicates"])

    def test_pool_authorizations_default_false(self):
        rows, results = self._pool([True] * 48)
        report = pool_eligibility(rows, results)
        for key in ("authorizes_phase0", "authorizes_collection",
                    "authorizes_training", "authorizes_evaluation",
                    "phase0_passed", "human_approval_present"):
            self.assertFalse(report[key], key)


class ManifestTest(unittest.TestCase):
    def test_phase0_and_review_shapes_and_balance(self):
        g, r = phase0_rows(), review_rows()
        self.assertEqual((len(g), len(r)), (24, 48))
        for side in INTRUSION_SIDES:
            self.assertEqual(sum(1 for x in g if x["intrusion_side"] == side), 12)
            self.assertEqual(sum(1 for x in r if x["intrusion_side"] == side), 24)
        for pose in POSE_IDS:
            self.assertEqual(sum(1 for x in g if x["pose_id"] == pose), 8)
            self.assertEqual(sum(1 for x in r if x["pose_id"] == pose), 16)

    def test_streams_disjoint_from_each_other_and_from_v105(self):
        from pact_place_v105_contract import (
            phase0_rows as v105_gate,
            review_rows as v105_review,
        )

        report = streams_are_disjoint(review_rows(), phase0_rows())
        self.assertTrue(report["disjoint"])
        v106_ids = {x["episode_id"] for x in review_rows() + phase0_rows()}
        v105_ids = {x["episode_id"] for x in v105_review() + v105_gate()}
        self.assertEqual(v106_ids & v105_ids, set())

    def test_contract_records_global_not_per_family_placement(self):
        lineage = build_specification_contract()["lineage"]
        self.assertFalse(lineage["per_family_placement"])
        self.assertTrue(lineage["global_asymmetric_placement"])
        self.assertFalse(lineage["uses_v95_low_wall"])

    def test_contract_declares_it_modifies_no_prior_artifact(self):
        contract = build_specification_contract()
        self.assertTrue(contract["does_not_modify_v104_or_v105_artifacts"])
        self.assertTrue(contract["v105_narrative_treated_as_untrusted"])
        self.assertFalse(contract["lattice"]["may_be_extended_after_results"])

    def test_contract_records_the_band_field_definition(self):
        accounting = build_specification_contract()["evaluation_accounting"]
        self.assertEqual(accounting["n_evaluations_per_bundle"], 294)
        self.assertIn("evaluations", accounting["band_evaluations_by_group_definition"])

    def test_six_videos_registered(self):
        self.assertEqual(N_REVIEW_VIDEOS, 6)


class V105ArtifactsUntouchedTest(unittest.TestCase):
    SEALED = {
        "diagnostics_output/pact_place_v105_reconstruction/reconstruction.json":
            "71bcb635e038fbe158c1bf926562b4b6cc3ae8a7e181dab227728af017e149f1",
        "diagnostics_output/pact_place_v105_siting/siting.json":
            "56f5d6ba2e35c1f76ee5945fbd2976c10ac93553e68cd23b41b2e229d76fb6b4",
    }

    def test_v105_artifacts_are_byte_identical(self):
        import hashlib

        for relative, expected in self.SEALED.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative
            )

    def test_v106_writes_only_to_v106_roots(self):
        from pact_place_v106_contract import (
            CAUSAL_ROOT, CERT_ROOT, PHASE0_ROOT, POOL_ROOT, REVIEW_ROOT,
            SITING_ROOT,
        )

        for root in (SITING_ROOT, CERT_ROOT, CAUSAL_ROOT, POOL_ROOT,
                     REVIEW_ROOT, PHASE0_ROOT):
            self.assertIn("v106", root)


if __name__ == "__main__":
    unittest.main()
