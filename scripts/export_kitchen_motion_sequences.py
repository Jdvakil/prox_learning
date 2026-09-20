#!/usr/bin/env python3
"""Text-free, contact-checked kinematic sequences using original experiment assets.

These are illustrative IK paths, not learned-policy or dynamics rollouts.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

from export_kitchen_collision_pair import ROOT, ROW, SCENE, scene, contacts
from package_whole_body_figure import P, A, R, REL, CT, child, xml
import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

OUT = ROOT/'images/whole_body/kitchen_motion_sequences'
PATHS = {
    'collision': [[-.14, -.07, .12], [-.075, -.07, .12], [-.035, -.07, .12],
                  [.01, -.07, .12], [.037, -.07, .12], [.047, -.07, .12]],
    'avoidance': [[-.14, -.07, .12], [-.075, -.07, .12], [-.035, .005, .12],
                  [.01, .09, .12], [.07, .18, .12], [.14, .18, .12]],
}
VIEWS = {'side_view': -55, 'opposite_side': 55}
CAMERA = dict(lookat=[.32, 0, .89], distance=1.65, elevation=-20, fovy=42)


def deck(description=None):
    with ZipFile(ROOT/'images/whole_body/whole_body_canva.pptx') as z:
        files = {n: z.read(n) for n in z.namelist()
                 if not n.startswith(('ppt/slides/', 'ppt/notesSlides/', 'ppt/notesMasters/', 'ppt/media/'))}
    pres = ET.fromstring(files['ppt/presentation.xml'])
    slides = pres.find(f'{{{P}}}sldIdLst'); slides.clear()
    notes = pres.find(f'{{{P}}}notesMasterIdLst')
    if notes is not None: pres.remove(notes)
    size = pres.find(f'{{{P}}}sldSz')
    scale = 4000
    size.set('cx', str(2952*scale)); size.set('cy', str(744*scale)); size.set('type', 'custom')
    rels = ET.fromstring(files['ppt/_rels/presentation.xml.rels'])
    for rel in list(rels):
        if rel.get('Type', '').endswith(('/slide', '/notesMaster')): rels.remove(rel)
    types = ET.fromstring(files['[Content_Types].xml'])
    for item in list(types):
        if item.get('PartName', '').startswith(('/ppt/slides/', '/ppt/notesSlides/', '/ppt/notesMasters/')):
            types.remove(item)
    for slide_i, view in enumerate(VIEWS, 1):
        sld = ET.Element(f'{{{P}}}sld')
        tree = child(child(sld, P, 'cSld'), P, 'spTree')
        nv = child(tree, P, 'nvGrpSpPr'); child(nv, P, 'cNvPr', id=1, name='')
        child(nv, P, 'cNvGrpSpPr'); child(nv, P, 'nvPr'); child(tree, P, 'grpSpPr')
        sr = ET.Element(f'{{{REL}}}Relationships')
        child(sr, REL, 'Relationship', Id='rId1', Type=R+'/slideLayout', Target='../slideLayouts/slideLayout7.xml')
        for row, label in enumerate(PATHS):
            for frame in range(6):
                idx = 2+row*6+frame
                file = OUT/view/label/f'frame_{frame+1:02d}.png'
                media = f'{view}_{label}_{frame+1:02d}.png'
                files['ppt/media/'+media] = file.read_bytes()
                child(sr, REL, 'Relationship', Id=f'rId{idx}', Type=R+'/image', Target='../media/'+media)
                pic = child(tree, P, 'pic'); nv = child(pic, P, 'nvPicPr')
                child(nv, P, 'cNvPr', id=idx, name=str(file.relative_to(OUT)))
                child(nv, P, 'cNvPicPr'); child(nv, P, 'nvPr')
                fill = child(pic, P, 'blipFill'); blip = child(fill, A, 'blip')
                blip.set(f'{{{R}}}embed', f'rId{idx}')
                child(child(fill, A, 'stretch'), A, 'fillRect')
                props = child(pic, P, 'spPr'); xf = child(props, A, 'xfrm')
                child(xf, A, 'off', x=(6+492*frame)*scale, y=(6+372*row)*scale)
                child(xf, A, 'ext', cx=480*scale, cy=360*scale)
                child(child(props, A, 'prstGeom', prst='rect'), A, 'avLst')
        child(child(sld, P, 'clrMapOvr'), A, 'masterClrMapping')
        files[f'ppt/slides/slide{slide_i}.xml'] = xml(sld)
        files[f'ppt/slides/_rels/slide{slide_i}.xml.rels'] = xml(sr)
        entry = child(slides, P, 'sldId', id=255+slide_i); entry.set(f'{{{R}}}id', f'rId{20+slide_i}')
        child(rels, REL, 'Relationship', Id=f'rId{20+slide_i}', Type=R+'/slide', Target=f'slides/slide{slide_i}.xml')
        child(types, CT, 'Override', PartName=f'/ppt/slides/slide{slide_i}.xml', ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
    files['ppt/presentation.xml'] = xml(pres)
    files['ppt/_rels/presentation.xml.rels'] = xml(rels)
    files['[Content_Types].xml'] = xml(types)
    core = ET.fromstring(files['docProps/core.xml'])
    for el in core:
        if el.tag.endswith('}title'): el.text = 'Kitchen collision and avoidance: illustrative motion sequences'
        elif el.tag.endswith('}description'): el.text = description or 'Two camera views. Top row collision, bottom row avoidance. Six independently movable frames per sequence. Constructed IK motions, not policy rollouts.'
    files['docProps/core.xml'] = xml(core)
    app = ET.fromstring(files['docProps/app.xml'])
    for el in list(app):
        if el.tag.endswith('}Slides'): el.text = '2'
        elif el.tag.endswith(('}HeadingPairs', '}TitlesOfParts')): app.remove(el)
    files['docProps/app.xml'] = xml(app)
    with Image.open(OUT/'side_view/comparison.png') as im:
        im.thumbnail((512, 512)); b = BytesIO(); im.save(b, format='JPEG')
        files['docProps/thumbnail.jpeg'] = b.getvalue()
    with ZipFile(OUT/'kitchen_sequences_canva.pptx', 'w', ZIP_DEFLATED) as z:
        for name, data in files.items(): z.writestr(name, data)


def main(snapshot=None):
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, _ = scene(snapshot)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 1440)
    m.vis.global_.fovy = CAMERA['fovy']
    joints = [m.joint(f'robot_0/fr3_joint{i+1}') for i in range(7)]
    adr = [j.qposadr[0] for j in joints]
    source = json.loads((ROOT/'images/whole_body/kitchen_items/manifest.json').read_text())['provenance']
    q0 = np.array(source['adjusted_simulation_pose']['adjusted_arm_qpos'])
    d.qpos[adr] = q0; mujoco.mj_forward(m, d)
    cid = m.camera('robot_0/link6_sensor_4').id
    p0 = d.cam_xpos[cid].copy(); rotation = d.cam_xmat[cid].reshape(3, 3).copy()
    fixed = [i for i in range(m.nbody) if not m.body(i).name.startswith('robot_0/')]
    fixed_pos = d.xpos[fixed].copy(); fixed_rot = d.xmat[fixed].copy()
    robot = [i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith('robot_0/')
             and m.body(int(m.geom_bodyid[i])).name != 'robot_0/base'
             and int(m.geom_contype[i] | m.geom_conaffinity[i])]
    hazard = [i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name == 'pact_intrusion_right'
              and int(m.geom_contype[i] | m.geom_conaffinity[i])]

    def set_pose(q):
        d.qpos[adr] = q; mujoco.mj_forward(m, d)
        assert np.array_equal(d.xpos[fixed], fixed_pos)
        assert np.array_equal(d.xmat[fixed], fixed_rot)

    def panel_distance():
        distances = []
        for a in robot:
            for b in hazard:
                witness = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(m, d, a, b, .5, witness))
                # This installed MuJoCo build can return zero for separated
                # box/mesh pairs while returning distinct nearest witnesses.
                # Recover their unsigned separation, preserving negative
                # penetration distances and the .5 m search cutoff.
                if distance == 0:
                    distance = float(np.linalg.norm(witness[:3]-witness[3:]))
                distances.append(distance)
        return min(distances)

    def solve(delta):
        def residual(q):
            d.qpos[adr] = q; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
            dr = Rotation.from_matrix(d.cam_xmat[cid].reshape(3, 3) @ rotation.T).as_rotvec()
            return np.r_[10*(d.cam_xpos[cid]-p0-delta), dr, .002*(q-q0)]
        fit = least_squares(residual, q0, max_nfev=150,
                            bounds=([j.range[0] for j in joints], [j.range[1] for j in joints]))
        residual(fit.x)
        assert fit.success and np.linalg.norm(d.cam_xpos[cid]-p0-delta) < 1e-5
        set_pose(fit.x)
        return fit.x

    paths = {}; evidence = {}
    for label, deltas in PATHS.items():
        qs = np.array([solve(np.array(delta)) for delta in deltas]); paths[label] = qs
        sample_q = []; checks = []; frame_checks = []
        for seg, (qa, qb) in enumerate(zip(qs[:-1], qs[1:])):
            for fraction in np.linspace(0, 1, 101):
                q = qa*(1-fraction)+qb*fraction
                set_pose(q)
                cs = contacts(m, d)
                distance = panel_distance()
                sample_q.append(q.copy())
                checks.append(dict(segment=seg, fraction=float(fraction), panel_signed_distance_m=distance, contacts=cs))
        for i, q in enumerate(qs):
            set_pose(q)
            frame_checks.append(dict(frame=i+1, contacts=contacts(m, d)))
        if label == 'avoidance':
            assert all(not c['contacts'] for c in checks)
        else:
            assert all(not c['contacts'] for c in frame_checks[:-1])
            assert len(frame_checks[-1]['contacts']) == 1
            for c in checks:
                for hit in c['contacts']:
                    assert hit['bodies'] == ['pact_intrusion_right', 'robot_0/fr3_link7']
                    assert hit['signed_distance_m'] > -.004
        sample_q = np.array(sample_q)
        np.savez_compressed(OUT/f'{label}_path.npz', frame_arm_qpos=qs, sampled_arm_qpos=sample_q,
                            full_qpos_template=d.qpos, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
        evidence[label] = dict(frame_checks=frame_checks, sampled_checks=checks,
                               checked_configurations=len(checks),
                               minimum_panel_distance_m=min(c['panel_signed_distance_m'] for c in checks),
                               maximum_checked_joint_increment_rad=float(abs(np.diff(sample_q, axis=0)).max()))
        print(label, 'checked', len(checks), 'contact samples', sum(bool(c['contacts']) for c in checks), flush=True)
    assert np.array_equal(paths['collision'][:2], paths['avoidance'][:2])
    opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0; opt.sitegroup[:] = 0
    for view, azimuth in VIEWS.items():
        camera = mujoco.MjvCamera(); camera.azimuth = azimuth
        for key in ['lookat', 'distance', 'elevation']: setattr(camera, key, CAMERA[key])
        with mujoco.Renderer(m, 1440, 1920) as renderer:
            for label, qs in paths.items():
                folder = OUT/view/label; folder.mkdir(parents=True, exist_ok=True)
                strip = Image.new('RGB', (6*960+5*16, 720), 'white')
                for i, q in enumerate(qs):
                    set_pose(q); renderer.update_scene(d, camera=camera, scene_option=opt)
                    frame = Image.fromarray(renderer.render().copy())
                    frame.save(folder/f'frame_{i+1:02d}.png')
                    strip.paste(frame.resize((960, 720), Image.Resampling.LANCZOS), (i*976, 0))
                    if i == 5:
                        # Exact pixel crop of the final frame; no image synthesis.
                        crop = (430, 300, 1420, 1000) if view == 'side_view' else (620, 260, 1610, 960)
                        frame.crop(crop).save(folder/'final_detail.png')
                strip.save(OUT/view/f'{label}_strip.png')
        comparison = Image.new('RGB', (5840, 1456), 'white')
        for row, label in enumerate(PATHS):
            with Image.open(OUT/view/f'{label}_strip.png') as im: comparison.paste(im, (0, row*736))
        comparison.save(OUT/view/'comparison.png')
        print('Exported', view, flush=True)
    manifest = dict(
        type='Illustrative kinematic paths in a restored experiment scene; not learned-policy or physics rollouts.',
        source_row=str(ROW), source_scene=str(SCENE), source_frame=0, source_provenance=source,
        motion='Six IK waypoints per path, preserving link6 sensor4 orientation. Shared first two poses. Joint-linear interpolation checked at 101 positions per segment, including both endpoints.',
        collision='First five exported poses are contact-free; final pose contacts the existing hood panel with link7. Only the final approach segment has contact.',
        avoidance='Lateral detour toward the open side, then forward past the panel front. No robot/environment contact at any of the 505 checked configurations. Does not demonstrate reaching a shared task goal or sensor-driven policy behavior.',
        limits='Sampled kinematic collision checks; no continuous swept-volume certificate, time parameterization, dynamics, forces or learned policy were evaluated.',
        distance_method='MuJoCo signed geometry distance, capped at 0.5 m. If it returns zero with distinct nearest witness points, their Euclidean separation is used. Contact decisions use mj_forward contact records independently.',
        camera=CAMERA, azimuth_by_view=VIEWS, individual_frame_pixels=[1920, 1440], frames_per_sequence=6,
        image_order='Left to right in each strip; collision top, avoidance bottom in comparisons and PPTX.',
        changes='Only robot joint positions and camera angle; original objects, panel geometry, material, lighting and object transforms retained. No generated image content or text overlays.',
        sensor4_translation_waypoints_m=PATHS, checks=evidence)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    deck()
    with ZipFile(OUT/'kitchen_sequences_assets.zip', 'w', ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip': z.write(path, path.relative_to(OUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Optional original scene cache from the prior exporter.')
    main(parser.parse_args().snapshot)
