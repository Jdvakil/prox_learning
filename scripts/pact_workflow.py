"""Dataset contracts, grouped splits and deterministic evaluation rows for PACT.

No simulator imports. Paths in contracts are repository-relative for portable runs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pact_experiment_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("unsupported experiment contract")
    expected = contract.get("sha256")
    payload = {k: v for k, v in contract.items() if k != "sha256"}
    if expected != digest(payload):
        raise ValueError("experiment contract hash mismatch; prepare a new contract")
    validate_split(contract["episodes"], contract["split"])
    for relative, expected in contract.get('adapter_files', {}).items():
        if file_digest(resolve(relative)) != expected:
            raise ValueError(f'Adapter input changed since prepare: {relative}; use a new dataset profile')
    return contract


def groups_for(episodes):
    # Union repeated final worlds even when different retry/request seeds produced them.
    parent = list(range(len(episodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, ep in enumerate(episodes):
        for key in ("selected_seed", "layout_sha256", "trajectory_sha256"):
            value = ep.get(key)
            if value in (None, ""):
                continue
            identity = (key, str(value))
            if identity in seen:
                parent[find(i)] = find(seen[identity])
            else:
                seen[identity] = i
    groups = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[find(i)].append(ep["id"])
    return list(groups.values())


def grouped_split(episodes, seed=2026090401, val_fraction=0.2):
    by_id = {e["id"]: e for e in episodes}
    totals = Counter(e["cell"] for e in episodes)
    groups = groups_for(episodes)
    groups.sort(key=lambda ids: digest([seed, sorted(ids)]))
    counts = Counter()
    val = []
    target = max(len(totals), round(len(episodes) * val_fraction))
    while groups:
        candidates = []
        for i, group in enumerate(groups):
            added = Counter(by_id[j]["cell"] for j in group)
            if any(counts[c] + n >= totals[c] for c, n in added.items()):
                continue  # Every category must retain training examples.
            coverage = sum(counts[c] == 0 for c in added)
            improvement = sum(
                (counts[c] - totals[c] * val_fraction) ** 2
                - (counts[c] + n - totals[c] * val_fraction) ** 2
                for c, n in added.items()
            )
            candidates.append((coverage, improvement, -i, i, added))
        if not candidates:
            break
        coverage, _, _, index, added = max(candidates)
        if len(val) >= target and coverage == 0:
            break
        val.extend(groups.pop(index))
        counts.update(added)
    if set(counts) != set(totals):
        raise ValueError("cannot cover every cell in validation while keeping repeated worlds grouped")
    val = sorted(val)
    split = {"seed": seed, "train": sorted(set(by_id) - set(val)), "val": val,
             "normalization": "train_only", "group_keys": ["selected_seed", "layout_sha256", "trajectory_sha256"]}
    validate_split(episodes, split)
    return split


def validate_split(episodes, split):
    ids = [e["id"] for e in episodes]
    train, val = split["train"], split["val"]
    if len(ids) != len(set(ids)) or not train or not val:
        raise ValueError("empty partition or duplicate episode IDs")
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("duplicate split IDs")
    if set(train) & set(val) or set(train) | set(val) != set(ids):
        raise ValueError("split must partition all episodes exactly once")
    for group in groups_for(episodes):
        if set(group) & set(train) and set(group) & set(val):
            raise ValueError(f"repeated scene crosses train/validation: {group}")


def inventory(profile):
    import h5py
    meta = json.loads((resolve(profile["data_dir"]) / "convert_meta.json").read_text())
    if meta.get("camera_names") != profile["camera_names"]:
        raise ValueError("conversion camera order differs from dataset registry")
    if meta.get("prox_pool") != "min":
        raise ValueError("this adapter requires converted min-pooled proximity")
    episodes, templates = [], {}
    raw_root = resolve(profile["raw_dir"]) / "rows"
    if profile.get('dataset_environment_version'):
        manifest = json.loads((raw_root.parent / 'manifest.json').read_text())
        if manifest['environment_version'] != profile['dataset_environment_version']:
            raise ValueError('Raw dataset variant differs from profile')
        if profile['adapter'] == 'v12' and not manifest.get('standing_kitchen_extras'):
            raise ValueError('v12 requires the recorded kitchen overlay')
    for i in range(int(meta["num_episodes"])):
        converted = resolve(profile["data_dir"]) / f"episode_{i}.hdf5"
        with h5py.File(converted) as f:
            row_name = str(f.attrs["source_row"])
            length = int(f["action"].shape[0])
            if f["action"].shape != (length, 8) or f["observations/qpos"].shape != (length, 9):
                raise ValueError(f"state/action shape mismatch: {converted}")
            if f["observations/proximity"].shape != (length, 40, 8, 8):
                raise ValueError(f"proximity shape mismatch: {converted}")
            for camera in profile["camera_names"]:
                if f[f"observations/images/{camera}"].shape != (length, meta["image_h"], meta["image_w"], 3):
                    raise ValueError(f"image shape mismatch: {converted} {camera}")
        row_dir = raw_root / row_name
        result = json.loads((row_dir / "result.json").read_text())
        with h5py.File(row_dir / "trajectory.h5") as f:
            blob = f["traj_0/obs_scene"][()]
            scene = json.loads(blob.decode() if isinstance(blob, bytes) else blob)
        params = scene.get("scene_params", {})
        if params.get("pact_place_environment_version") != profile["environment_version"]:
            raise ValueError(f"environment mismatch in {row_dir}")
        if int(scene["policy_dt_ms"]) != profile["policy_dt_ms"]:
            raise ValueError(f"control period mismatch in {row_dir}")
        side = result.get("intrusion_side") or params.get("pact_intrusion_side")
        family = result.get("family_id", "hallway")
        pose = result.get("pose_id", "center")
        if profile.get('supported_poses') and pose not in profile['supported_poses']:
            raise ValueError(f'Unregistered pose {pose} in {row_dir}')
        if profile.get('scene_sha256') and params.get('pact_v106_scene_sha256') != profile['scene_sha256']:
            raise ValueError(f'Collection scene hash differs from profile: {row_dir}')
        cell = "|".join((family, side, pose))
        selected = result.get("selected_seed") or {}
        if isinstance(selected, dict):
            selected = selected.get("seed_u32")
        episode = {"id": i, "source_row": row_name, "cell": cell, "length": length,
                   "selected_seed": selected,
                   "requested_seed": result.get("task_seed_u32"),
                   "layout_sha256": params.get("pact_v1011_layout_sha256"),
                   "trajectory_sha256": result.get("trajectory_h5_sha256"),
                   "converted_size": converted.stat().st_size,
                   "converted_mtime_ns": converted.stat().st_mtime_ns}
        episodes.append(episode)
        if cell not in templates:
            templates[cell] = {"family_id": family, "intrusion_side": side, "pose_id": pose,
                               "scene_params": params}
    return meta, episodes, templates


def evaluation_row(profile, template, role, index, forbidden_seeds):
    family, side, pose = template["family_id"], template["intrusion_side"], template["pose_id"]
    nonce = 0
    while True:
        seed = int(digest(["pact_eval_v1", profile["environment_version"], role, index, nonce])[:8], 16)
        if seed not in forbidden_seeds:
            forbidden_seeds.add(seed)
            break
        nonce += 1
    row = {"role": role, "role_index": index, "family_id": family, "intrusion_side": side,
           "pose_id": pose, "task_seed_u32": seed, "environment_version": profile["environment_version"],
           "scene_template_house_index": 1, "panel_x_jitter_m": 0.0, "panel_face_jitter_m": 0.0}
    if profile["adapter"] == "v1011d":
        params = template["scene_params"]
        layout = deepcopy(params["pact_clutter_layout"])
        # All six centers are redrawn by V1011D. Retain the observed shape/route
        # contract, never substitute V1010 household objects or final training poses.
        for key in ("base_slot_randomization", "near_target_placements", "target_rest_m",
                    "pact_v1011_layout_sha256", "pact_v1011_identity_sha256"):
            layout.pop(key, None)
        row.update({"pact_clutter_palette": deepcopy(params["pact_clutter_palette"]),
                    "pact_clutter_layout": layout,
                    "clutter_x_jitter_m": layout.get("applied_clutter_x_jitter_m", {}),
                    "clutter_y_jitter_m": layout.get("applied_clutter_y_jitter_m", {})})
        for key in ("pact_v106_scene_sha256", "pact_v106_x_m", "pact_v106_r_neg_m", "pact_v106_r_pos_m"):
            row[key] = params[key]
    elif profile['adapter'] == 'v12':
        from pact_v12_adapter import row_payload
        row.update(row_payload(profile, template))
    else:
        # Same hallway jitter support as the established frozen evaluation protocol.
        import random
        rng = random.Random(seed)
        row["panel_x_jitter_m"] = rng.uniform(-0.015, 0.015)
        row["panel_face_jitter_m"] = rng.uniform(-0.005, 0.005)
    row["sha256"] = digest(row)
    return row


def prepare_contract(dataset, profile):
    meta, episodes, templates = inventory(profile)
    split = grouped_split(episodes)
    cells = sorted(templates)
    # Spread a small development budget over families, sides and poses; full test
    # covers every category twice. Smoke is a prefix of development, never test.
    dev_cells = cells[::max(1, len(cells) // 8)][:8]
    if len(cells) == 24:
        dev_cells = [cells[i] for i in (0, 4, 7, 11, 14, 15, 19, 23)]
    forbidden = {e[k] for e in episodes for k in ("selected_seed", "requested_seed") if e.get(k) is not None}
    dev = [evaluation_row(profile, templates[c], "dev", i, forbidden) for i, c in enumerate(dev_cells)]
    test_cells = cells * profile.get('test_repeats_per_cell', 2 if len(cells) > 2 else 24)
    final = [evaluation_row(profile, templates[c], "test", i, forbidden) for i, c in enumerate(test_cells)]
    contract = {"schema_version": SCHEMA, "dataset": dataset, "profile": profile,
                "image_h": meta["image_h"], "image_w": meta["image_w"],
                "prox_pool": meta["prox_pool"], "episodes": episodes, "split": split,
                "evaluation": {"dev": dev, "test": final},
                "success_definition": "ever_success", "collision_window": "full_horizon",
                "validation_status": "runtime_verification_required"}
    if profile['adapter'] == 'v12':
        contract['adapter_files'] = {name: file_digest(resolve(name)) for name in (
            profile['clutter_config'], 'scripts/pact_v12_adapter.py',
            'scripts/render_pact_place_v12_clutter.py', 'scripts/pact_place_v12_contract.py')}
        contract['generalization_scope'] = 'new episode seeds within observed center-pose layout families; static layouts shared'
    contract["sha256"] = digest(contract)
    return contract


def check_dataset_files(contract):
    for ep in contract["episodes"]:
        path = resolve(contract["profile"]["data_dir"]) / f"episode_{ep['id']}.hdf5"
        stat = path.stat()
        if stat.st_size != ep["converted_size"] or stat.st_mtime_ns != ep["converted_mtime_ns"]:
            raise ValueError(f"converted dataset changed since prepare: {path}; prepare again")
