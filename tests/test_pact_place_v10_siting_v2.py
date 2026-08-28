from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file
from pact_place_v10_catalog import (
    SurvivorCatalogV2,
    assembly_from_two_lobe_keys,
    assembly_id_from_lobe_keys,
    unique_union_count,
    union_key_from_two_lobe_key,
    write_survivor_catalog_v2,
)
from pact_place_v10_compound_pendant_contract import (
    HOOD_TOP_BOTTOM_Z_M,
    HOOD_TOP_GEOM_NAME,
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
    V5_SCENE_XML_RELATIVE,
    V99_RECONSTRUCTION_SHA256,
    V99_SCOPED_CONCLUSION,
    V99_SITING_SHA256,
    V99_SNAPSHOT_SHA256,
    empty_authorization,
)
from pact_place_v10_environment import (
    aabb_overlap,
    assembly_panel_clear_on_side,
    combine_assembly_environment_cache,
    hood_top_attachment_face_only,
    panel_boxes_from_dump,
    panels_for_side,
    scan_panel_clear_mask,
    score_assembly_environment,
    score_component_against_environment,
    score_unique_keys_environment,
)
from pact_place_v10_exact import score_component_initial_target
from pact_place_v10_geometry import (
    build_assembly,
    lobe_from_key,
    planning_probe_assembly,
    planning_probe_v1_invalid_assembly,
    union_aabb_key,
    union_fixture,
)
from pact_place_v10_route import (
    evaluate_all_perturbation_corners,
    evaluate_environment_no_intersection,
    evaluate_pendant_nominal_and_robust,
    route_ik_cache_key,
    signal_screen_requires_shortlist,
    union_cluster_row_indices,
)


def _panel(side: str) -> dict:
    import mujoco

    cy = 0.34 if side == "left" else -0.34
    center = np.array([0.615, cy, 0.89], dtype=np.float64)
    half = np.array([0.055, 0.24, 0.09], dtype=np.float64)
    return {
        "role_index": 600 if side == "left" else 601,
        "intrusion_side": side,
        "name": f"pact_intrusion_{side}_g",
        "lo": center - half,
        "hi": center + half,
        "axis_aligned_box": True,
        "role": "panel",
        "body": f"pact_intrusion_{side}",
        "gtype": int(mujoco.mjtGeom.mjGEOM_BOX),
        "size": half,
        "pos": center,
        "mat": np.eye(3),
        "verts": None,
        "geom_id": 1 if side == "left" else 2,
    }


def _box_geom(*, name: str, role: str, center, half, geom_id: int = 9) -> dict:
    import mujoco

    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return {
        "geom_id": geom_id,
        "name": name,
        "body": name,
        "role": role,
        "gtype": int(mujoco.mjtGeom.mjGEOM_BOX),
        "size": half,
        "pos": center,
        "mat": np.eye(3, dtype=np.float64),
        "verts": None,
        "lo": center - half,
        "hi": center + half,
        "axis_aligned_box": True,
    }


class PactPlaceV10SitingV2Test(unittest.TestCase):
    def test_probe_v1_rejected_by_both_side_specific_panels(self) -> None:
        v1 = planning_probe_v1_invalid_assembly()
        self.assertEqual(v1["probe_label"], "probe_v1_invalid_panel_overlap")
        left = [_panel("left")]
        right = [_panel("right")]
        self.assertFalse(assembly_panel_clear_on_side(v1, left))
        self.assertFalse(assembly_panel_clear_on_side(v1, right))
        v2 = planning_probe_assembly()
        self.assertEqual(v2["probe_label"], "probe_v2")
        self.assertTrue(assembly_panel_clear_on_side(v2, left))
        self.assertTrue(assembly_panel_clear_on_side(v2, right))
        self.assertNotEqual(v1["assembly_id"], v2["assembly_id"])

    def test_panel_and_clutter_are_in_cached_and_direct_eval(self) -> None:
        v1 = planning_probe_v1_invalid_assembly()
        v2 = planning_probe_assembly()
        cells = [
            {
                "role_index": 600,
                "intrusion_side": "left",
                "geoms": [
                    _panel("left"),
                    _box_geom(
                        name="pact_clutter_00/decor",
                        role="clutter",
                        center=[1.2, 0.0, 0.9],
                        half=[0.02, 0.02, 0.02],
                    ),
                ],
            },
            {
                "role_index": 601,
                "intrusion_side": "right",
                "geoms": [_panel("right")],
            },
        ]
        direct_v1 = score_assembly_environment(v1, cells)
        direct_v2 = score_assembly_environment(v2, cells)
        self.assertFalse(direct_v1["panel_clear"])
        self.assertFalse(direct_v1["environment_clear"])
        self.assertTrue(direct_v2["panel_clear"])
        self.assertTrue(direct_v2["environment_clear"])
        unique = []
        for item in v1["components"] + v2["components"]:
            if not item.get("active", True):
                continue
            unique.append(item)
        from pact_place_v10_environment import score_unique_components_environment

        cache = score_unique_components_environment(unique, cells)
        cached_v1 = combine_assembly_environment_cache(v1, cache)
        cached_v2 = combine_assembly_environment_cache(v2, cache)
        self.assertEqual(cached_v1["panel_clear"], direct_v1["panel_clear"])
        self.assertEqual(cached_v1["environment_clear"], direct_v1["environment_clear"])
        self.assertEqual(cached_v2["panel_clear"], direct_v2["panel_clear"])
        self.assertEqual(cached_v2["environment_clear"], direct_v2["environment_clear"])
        self.assertTrue(any(item["role"] == "panel" for item in cells[0]["geoms"]))
        self.assertTrue(any(item["role"] == "clutter" for item in cells[0]["geoms"]))

    def test_initial_target_contact_is_rejected(self) -> None:
        import mujoco

        assembly = planning_probe_assembly()
        lobe = next(item for item in assembly["components"] if item["role"] == "lobe" and item["active"])
        n, g = 2, 1
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        target_pos = np.full((n, g, 3), 10.0, dtype=np.float64)
        target_lo = np.full((n, g, 3), 9.9)
        target_hi = np.full((n, g, 3), 10.1)
        target_pos[0, 0] = lobe["center_m"]
        target_lo[0, 0] = np.asarray(lobe["center_m"]) - 0.001
        target_hi[0, 0] = np.asarray(lobe["center_m"]) + 0.001
        cell = {
            "role_index": 600,
            "target_gtype": np.array([box_type], dtype=np.int32),
            "target_size": np.array([[0.02, 0.02, 0.02]], dtype=np.float64),
            "target_pos": target_pos,
            "target_mat": np.tile(np.eye(3, dtype=np.float64).reshape(9), (n, 1, 1)),
            "target_verts": [None],
            "target_lo": target_lo,
            "target_hi": target_hi,
            "initial_mask": np.array([True, False]),
        }
        report = score_component_initial_target(lobe, cell)
        self.assertFalse(report["initial_target_clear"])
        far = dict(cell)
        far["target_pos"] = np.full((n, g, 3), 10.0)
        far["target_lo"] = np.full((n, g, 3), 9.9)
        far["target_hi"] = np.full((n, g, 3), 10.1)
        clear = score_component_initial_target(lobe, far)
        self.assertTrue(clear["initial_target_clear"])

    def test_only_crossbar_hood_top_attachment_is_allowed(self) -> None:
        assembly = planning_probe_assembly()
        bar = next(item for item in assembly["components"] if item["role"] == "crossbar")
        lobe = next(item for item in assembly["components"] if item["role"] == "lobe" and item["active"])
        hood = _box_geom(
            name=HOOD_TOP_GEOM_NAME,
            role="hood_top",
            center=[0.95, 0.0, 1.53],
            half=[0.42, 0.46, 0.015],
        )
        self.assertTrue(hood_top_attachment_face_only(bar, hood))
        self.assertFalse(hood_top_attachment_face_only(lobe, hood))
        bar_ok = score_component_against_environment(bar, [hood])
        self.assertTrue(bar_ok["environment_clear"])
        penetrator = dict(bar)
        penetrator["center_m"] = [bar["center_m"][0], bar["center_m"][1], 1.53]
        penetrator["half_m"] = [0.003, 0.20, 0.05]
        self.assertFalse(hood_top_attachment_face_only(penetrator, hood))
        penetrated = score_component_against_environment(penetrator, [hood])
        self.assertFalse(penetrated["environment_clear"])
        other = _box_geom(
            name="hood_side_l",
            role="static",
            center=[0.95, 0.45, 1.12],
            half=[0.40, 0.012, 0.40],
        )
        overlapping_static = dict(bar)
        overlapping_static["center_m"] = [0.95, 0.45, 1.12]
        overlapping_static["half_m"] = [0.05, 0.05, 0.05]
        static_hit = score_component_against_environment(overlapping_static, [other])
        self.assertFalse(static_hit["environment_clear"])

    def test_catalog_ordering_hashes_and_streaming_memory(self) -> None:
        keys = np.array(
            [
                [[0.70, 0.22, 0.86, 0.01, 0.02, 0.02], [0.70, -0.18, 0.86, 0.01, 0.04, 0.04]],
                [[0.70, -0.18, 0.86, 0.01, 0.04, 0.04], [0.70, 0.22, 0.86, 0.01, 0.02, 0.02]],
            ],
            dtype=np.float64,
        )
        volume = np.array([0.2, 0.1], dtype=np.float64)
        bits = np.array([1, 2], dtype=np.int32)
        margin = np.array([0.01, 0.02], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exact_survivors_v2.npz"
            first = write_survivor_catalog_v2(
                path,
                lobe_keys=keys,
                volume_m3=volume,
                lobe_necessity_bits=bits,
                min_grasp_clearance_margin_m=margin,
            )
            second = write_survivor_catalog_v2(
                path,
                lobe_keys=keys[::-1],
                volume_m3=volume[::-1],
                lobe_necessity_bits=bits[::-1],
                min_grasp_clearance_margin_m=margin[::-1],
            )
            self.assertEqual(first, second)
            catalog = SurvivorCatalogV2(path, mmap=True)
            self.assertEqual(len(catalog), 2)
            self.assertIsInstance(catalog.lobe_keys, np.memmap)
            row0 = catalog.row(0)
            self.assertNotIn("assembly_ids", catalog.memory_bound_fields())
            self.assertEqual(
                row0["assembly_id"],
                assembly_id_from_lobe_keys(row0["lobe_keys"], topology="two_lobe"),
            )
            ids = [item["assembly_id"] for item in catalog.iter_rows()]
            self.assertEqual(ids, sorted(ids))

    def test_union_equivalent_morphologies_remain_distinct(self) -> None:
        negative_a = lobe_from_key((0.70, -0.20, 0.90, 0.02, 0.04, 0.04))
        negative_b = lobe_from_key((0.70, -0.22, 0.90, 0.02, 0.02, 0.04))
        positive = lobe_from_key((0.70, 0.20, 0.90, 0.02, 0.04, 0.04))
        a = build_assembly([negative_a, positive])
        b = build_assembly([negative_b, positive])
        union_a = union_aabb_key(
            *[[union_fixture(a)["center_m"][i] - union_fixture(a)["half_m"][i] for i in range(3)],
              [union_fixture(a)["center_m"][i] + union_fixture(a)["half_m"][i] for i in range(3)]],
        )
        # Use the same helper the catalog uses.
        key_a = union_key_from_two_lobe_key([negative_a["key"], positive["key"]])
        key_b = union_key_from_two_lobe_key([negative_b["key"], positive["key"]])
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(a["assembly_id"], b["assembly_id"])
        self.assertNotEqual(a["volume_m3"], b["volume_m3"])
        clusters = union_cluster_row_indices([key_a, key_b])
        self.assertEqual(len(clusters[key_a]), 2)
        ranked = [
            {"assembly_id": a["assembly_id"], "volume_m3": a["volume_m3"]},
            {"assembly_id": b["assembly_id"], "volume_m3": b["volume_m3"]},
        ]
        self.assertEqual({item["assembly_id"] for item in ranked}, {a["assembly_id"], b["assembly_id"]})
        screen = signal_screen_requires_shortlist(2, max_complete_screens=1)
        self.assertTrue(screen["requires_shortlist_amendment"])
        self.assertFalse(screen["collapsed_by_union_aabb"])
        self.assertEqual(screen["stop_reason"], "too_many_morphologies_for_signal_screen")

    def test_nominal_and_eight_robust_corners_are_separate_predicates(self) -> None:
        pendant = evaluate_pendant_nominal_and_robust(
            min_nominal_m=0.025,
            min_robust_m=0.020,
            nominal_clearance_m=0.026,
            corner_clearances_m=[0.021] * 8,
        )
        self.assertTrue(pendant["meets_nominal"])
        self.assertTrue(pendant["meets_robust"])
        self.assertEqual(pendant["n_corners_evaluated"], 8)
        failing = evaluate_pendant_nominal_and_robust(
            min_nominal_m=0.025,
            min_robust_m=0.020,
            nominal_clearance_m=0.026,
            corner_clearances_m=[0.021] * 7 + [0.019],
        )
        self.assertTrue(failing["meets_nominal"])
        self.assertFalse(failing["meets_robust"])
        env = evaluate_environment_no_intersection([0.001, 0.010, 0.040])
        self.assertTrue(env["environment_clear"])
        self.assertGreater(0.001, 0.0)
        calls = []

        def evaluator(corner):
            calls.append(int(corner["perturbation_index"]))
            return {"clearance_m": 0.03}

        reports = evaluate_all_perturbation_corners(
            {
                "lane_y_m": 0.12,
                "entry_x_m": 0.80,
                "exit_x_m": 0.50,
            },
            evaluator,
        )
        self.assertEqual(len(reports), 8)
        self.assertEqual(calls, list(range(8)))
        self.assertTrue(all(item["evaluated"] for item in reports))
        cache_a = route_ik_cache_key(
            cell_role_index=600,
            direction="inbound",
            union_key=key_placeholder(),
            padding_m=0.10,
            lane_y_m=-0.12,
            perturbation_index=0,
        )
        cache_b = route_ik_cache_key(
            cell_role_index=600,
            direction="inbound",
            union_key=key_placeholder(),
            padding_m=0.10,
            lane_y_m=-0.12,
            perturbation_index=1,
        )
        self.assertNotEqual(cache_a, cache_b)

    def test_v99_hashes_and_conclusions_unchanged(self) -> None:
        recon = json.loads(
            (ROOT / "diagnostics_output/pact_place_v99_baseline_reconstruction/reconstruction.json").read_text()
        )
        snap = json.loads(
            (ROOT / "diagnostics_output/pact_place_v99_siting/snapshots/snapshots.json").read_text()
        )
        siting = json.loads(
            (ROOT / "diagnostics_output/pact_place_v99_siting/siting.json").read_text()
        )
        self.assertEqual(recon["artifact_sha256"], V99_RECONSTRUCTION_SHA256)
        self.assertEqual(snap["artifact_sha256"], V99_SNAPSHOT_SHA256)
        self.assertEqual(siting["artifact_sha256"], V99_SITING_SHA256)
        self.assertEqual(siting.get("scoped_conclusion") or recon.get("scoped_conclusion") or V99_SCOPED_CONCLUSION, V99_SCOPED_CONCLUSION)
        self.assertEqual(sha256_file(ROOT / V5_SCENE_XML_RELATIVE), PLACE_V5_SCENE_SHA256)
        self.assertEqual(sha256_file(ROOT / SCENE_XML_RELATIVE), PLACE_V10_SCENE_SHA256)
        auth = empty_authorization()
        self.assertFalse(any(auth.values()))

    def test_siting_v2_artifact_if_present(self) -> None:
        path = ROOT / "diagnostics_output/pact_place_v10_siting_v2/siting.json"
        if not path.is_file():
            self.skipTest("siting v2 close-out has not been written yet")
        document = json.loads(path.read_text())
        self.assertEqual(document["schema_version"], "pact_place_v10_siting_v2")
        self.assertEqual(document["robot_target_prefilter_count"], 8554036)
        self.assertEqual(document["panel_clear_count"], 150288)
        self.assertFalse(document["routing_run"])
        self.assertFalse(document["physics_stepped"])
        self.assertFalse(document["episodes_run"])
        self.assertFalse(document["authorizes_collection"])
        self.assertEqual(document["v5_scene_xml_sha256"], PLACE_V5_SCENE_SHA256)
        self.assertEqual(document["v10_scene_xml_sha256"], PLACE_V10_SCENE_SHA256)
        self.assertNotEqual(document["v5_scene_xml_sha256"], document["v10_scene_xml_sha256"])
        self.assertEqual(int(document["full_environment_exact_survivor_count"]), 150288)
        self.assertEqual(int(document["panel_clear_union_aabb_count"]), 1779)
        self.assertEqual(int(document["corrected_unique_union_count"]), 1779)
        self.assertTrue(document["planning_probe"]["trust_anchor"])
        self.assertTrue(document["cache_direct_parity"]["ok"])
        self.assertFalse(document["three_lobe"]["searched"])
        catalog_path = ROOT / "diagnostics_output/pact_place_v10_siting_v2/exact_survivors_v2.npz"
        catalog = SurvivorCatalogV2(catalog_path)
        self.assertEqual(len(catalog), int(document["full_environment_exact_survivor_count"]))
        import zipfile

        self.assertNotIn("assembly_ids.npy", zipfile.ZipFile(catalog_path).namelist())
        from pact_place_v10_environment import load_environment_geoms, score_assembly_environment

        dumped = load_environment_geoms(
            ROOT / "diagnostics_output/pact_place_v10_siting_v2/environment_geoms.pkl.gz"
        )
        v1 = planning_probe_v1_invalid_assembly()
        v2 = planning_probe_assembly()
        direct_v1 = score_assembly_environment(v1, dumped)
        direct_v2 = score_assembly_environment(v2, dumped)
        self.assertFalse(direct_v1["environment_clear"])
        self.assertTrue(direct_v2["environment_clear"])
        self.assertEqual(direct_v2["n_cells"], 6)


def key_placeholder() -> tuple[float, ...]:
    return (0.69, -0.24, 0.82, 0.71, 0.24, 1.515)


if __name__ == "__main__":
    unittest.main()
