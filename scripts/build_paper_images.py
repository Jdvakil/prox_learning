#!/usr/bin/env python3
"""Build a provenance-tracked paper image collection from local models and data.

Run from any directory with the mlspaces Python environment. Original media and
experiment outputs are read only. Rendering changes are presentation-only.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/prox-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images'
MODEL = ROOT / 'assets/robots/franka_skin/model_hybrid.xml'
CATEGORIES = ['robot', 'skin', 'sensors', 'environments', 'tasks', 'results']
ITEMS = []
POSE = [0., -.35, 0., -2.10, 0., 1.95, .79]


def relative(path):
    return os.path.relpath(path, ROOT)


def record(path, source, method, note='', **extra):
    with Image.open(path) as im:
        size = list(im.size)
    entry = dict(file=str(path.relative_to(OUT)), category=path.parent.name,
                 dimensions=size, source=relative(source), method=method, note=note,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)
    ITEMS.append(entry)
    print(entry['file'], size, flush=True)


def save_render(arr, name, **extra):
    p = OUT / name
    Image.fromarray(arr).save(p, dpi=(300, 300))
    record(p, MODEL, 'native_mujoco_render', **extra)


def build_model():
    s = mujoco.MjSpec.from_file(str(MODEL))
    s.visual.global_.offwidth = 2400
    s.visual.global_.offheight = 2400
    s.visual.quality.offsamples = 8
    s.visual.headlight.ambient = [.35] * 3
    s.visual.headlight.diffuse = [.55] * 3
    s.add_texture(name='paper_white_background', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1,1,1], rgb2=[1,1,1],
                  width=128, height=768)
    s.worldbody.add_light(pos=[1,-2,3], dir=[-.3,.4,-1], diffuse=[.6]*3, specular=[.2]*3)
    s.worldbody.add_light(pos=[-1,1,2], dir=[.3,-.3,-1], diffuse=[.25]*3)
    m = s.compile()
    # The logo insert and collar contain matching letter-shaped geometry.
    # Give both a uniform black finish so neither white inserts nor cutouts form text.
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            if m.mesh(m.geom_dataid[i]).name in ('link0_4', 'link0_6'):
                m.geom_matid[i] = -1
                m.geom_rgba[i] = [0, 0, 0, 1]
    d = mujoco.MjData(m)
    for i, v in enumerate(POSE, 1):
        d.qpos[m.joint(f'fr3_joint{i}').qposadr[0]] = v
    mujoco.mj_forward(m, d)
    return m, d


def add_sphere(scene, pos, radius, color):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                       [radius]*3, pos, np.eye(3).ravel(), color)
    scene.ngeom += 1


def add_line(scene, a, b, radius, color):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
    scene.ngeom += 1


def render(m, d, *, az=135, el=-16, distance=1.35, lookat=(.23,0,.43),
           width=2400, height=2400, overlay=None):
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    cam.lookat = lookat
    cam.distance = distance
    cam.azimuth = az
    cam.elevation = el
    with mujoco.Renderer(m, height, width) as r:
        r.update_scene(d, cam, scene_option=opt)
        if overlay:
            overlay(r.scene)
        return r.render().copy()


def native_images():
    m,d = build_model()
    camera_ids = [i for i in range(m.ncam) if '_sensor_' in m.camera(i).name]
    assert len(camera_ids) == 40, len(camera_ids)
    notes = 'Canonical 40-sensor model; white background; uniform black cosmetic base collar to suppress lettering; no annotations.'
    for name, az in [('three_quarter',135),('opposite',225),('side',90)]:
        save_render(render(m,d,az=az), f'robot/fr3_{name}_2400.png', note=notes,
                    camera={'azimuth':az,'elevation':-16,'distance':1.35,'lookat':[.23,0,.43]})
    def markers(scene):
        for cid in camera_ids:
            add_sphere(scene, d.cam_xpos[cid], .006, [.92,.18,.10,1])
    save_render(render(m,d,overlay=markers), 'sensors/sensor_locations_2400.png',
                note=notes+' Red spheres indicate exact camera origins; markers are a visualization overlay.')
    save_render(render(m,d,az=125,el=-12,distance=.64,lookat=(.28,0,.68),width=2400,height=1600),
                'skin/forearm_skin_detail_2400.png',note=notes)
    save_render(render(m,d,az=130,el=-12,distance=.64,lookat=(.28,0,.68),width=2400,height=1600,overlay=markers),
                'sensors/forearm_sensor_locations_2400.png',note=notes+' Red spheres indicate camera origins.')
    groups = m.geom_group.copy()
    for i in range(m.ngeom):
        if '_skin' in m.body(m.geom_bodyid[i]).name:
            m.geom_group[i] = 5
    save_render(render(m,d), 'robot/fr3_no_skin_2400.png',
                note='Canonical FR3 + Robotiq in the isolated-dermis pose; seven hybrid skin geoms hidden; white background; uniform black cosmetic base collar; no annotations.',
                camera={'azimuth':135,'elevation':-16,'distance':1.35,'lookat':[.23,0,.43]})
    m.geom_group[:] = groups
    for i in range(m.ngeom):
        if '_skin' not in m.body(m.geom_bodyid[i]).name:
            m.geom_group[i] = 5
    save_render(render(m,d), 'skin/skin_shells_isolated_2400.png',
                note='Canonical seven dermis meshes in their assembled pose; underlying arm hidden for anatomy visualization.')
    m.geom_group[:] = groups
    # Show one real camera frustum to avoid an unreadable web of 40 overlapping cones.
    cid = m.camera('link4_sensor_0').id
    pos = d.cam_xpos[cid].copy()
    rot = d.cam_xmat[cid].reshape(3,3)
    length = .22
    half = np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2)*length
    corners = np.array([[-half,-half,-length],[half,-half,-length],
                        [half,half,-length],[-half,half,-length]]) @ rot.T + pos
    def frustum(scene):
        add_sphere(scene,pos,.007,[.96,.30,.10,1])
        for c in corners:
            add_line(scene,pos,c,.001,[.05,.58,.65,1])
        for a,b in zip(corners,np.roll(corners,-1,axis=0)):
            add_line(scene,a,b,.001,[.05,.58,.65,1])
    save_render(render(m,d,az=115,distance=1.7,lookat=(.23,0,.57),overlay=frustum),
                'sensors/single_sensor_field_of_view_2400.png',
                note='Actual link4_sensor_0 position, orientation and field of view; frustum truncated at 0.22 m for display.')


def curate():
    base = ROOT/'experiments_output/default/environment_viz'
    selection = {
        'hallway':'FrankaSkinPactPlaceV5Config',
        'four_object_clutter':'FrankaSkinPactPlaceV1010FourObjectConfig',
        'cabinet_cavity':'FrankaSkinCabinetCavitySmokeConfig',
        'cubby':'FrankaSkinCubbySmokeConfig',
        'panel_slalom':'FrankaSkinPanelSlalomSmokeConfig',
        'shelf_reach':'FrankaSkinShelfReachSmokeConfig',
        'clutter_reach':'FrankaSkinClutterReachSmokeConfig',
        'fumehood_clutter':'FrankaSkinHybridClutterPnPCheckConfig',
        'house_fumehood':'FrankaSkinHouseFumehoodSmokeConfig',
        'house_panel':'FrankaSkinHousePanelSmokeConfig',
        'house_cubby':'FrankaSkinHouseCubbySmokeConfig',
    }
    for name, config in selection.items():
        matches = sorted((base/config).glob('*/sample_00/01_robot_scene.png'))
        if not matches:
            raise FileNotFoundError(config)
        src = matches[0]
        dest = OUT/'environments'/f'{name}_views.png'
        shutil.copy2(src,dest)
        meta = json.loads((src.parent/'metadata.json').read_text())
        record(dest, src, 'unchanged_source_copy',
               note='Existing simulation views; no added captions. Native manufacturer markings may be present. This is environment illustration, not an outcome measurement.',
               config=config, task_description=meta.get('task_description'),
               source_metadata=relative(src.parent/'metadata.json'))
    # Include six real-table scene variants as distinct environment choices.
    for src in sorted((base/'FrankaSkinRealTableConfig').glob('*/sample_00/01_robot_scene.png')):
        name = src.parent.parent.name.replace('_house_', '_h')
        dest = OUT/'environments'/f'{name}_views.png'
        shutil.copy2(src,dest)
        record(dest, src, 'unchanged_source_copy',note='Existing simulation views; no captions; native manufacturer markings may be present.')


def native_environments():
    base = ROOT/'experiments_output/default/environment_viz'
    selections = [('hallway','FrankaSkinPactPlaceV5Config'),
                  ('cabinet_cavity','FrankaSkinCabinetCavitySmokeConfig'),
                  ('shelf_reach','FrankaSkinShelfReachSmokeConfig'),
                  ('panel_slalom','FrankaSkinPanelSlalomSmokeConfig'),
                  ('clutter_reach','FrankaSkinClutterReachSmokeConfig')]
    for name,config in selections:
        meta_path=next((base/config).glob('*/sample_00/metadata.json'))
        meta=json.loads(meta_path.read_text())
        src=Path(meta['scene'])
        spec=mujoco.MjSpec.from_file(str(src))
        spec.visual.global_.offwidth=2400
        spec.visual.global_.offheight=1600
        spec.visual.quality.offsamples=8
        spec.visual.headlight.ambient=[.3]*3
        spec.visual.headlight.diffuse=[.45]*3
        spec.add_texture(name='paper_scene_background',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,rgb1=[1,1,1],rgb2=[1,1,1],
                         width=128,height=768)
        model=spec.compile()
        for i in range(model.ngeom):
            geom_name=model.geom(i).name
            if geom_name.startswith('room_wall') or geom_name=='room_ceiling':
                model.geom_group[i]=5
            if geom_name=='floor':
                model.geom_matid[i]=-1
                model.geom_rgba[i]=[.94,.95,.97,1]
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        cam=meta['camera']
        view=cam['presentation_views'][0]
        arr=render(model,data,az=view['azimuth'],el=-22,distance=cam['distance']*1.15,
                   lookat=cam['lookat'],width=2400,height=1600)
        dest=OUT/'environments'/f'{name}_scene_2400.png'
        Image.fromarray(arr).save(dest,dpi=(300,300))
        record(dest,src,'native_mujoco_scene_render',
               note='Static source environment geometry at XML defaults; no robot or task sampler inserted. Presentation cutaway hides enclosing room walls and ceiling; white background and pale floor. No outcome claim or text overlays.',
               source_metadata=relative(meta_path))


def tasks():
    summary = ROOT/'eval_output/simple_v1011d_smoke_video/eval_summary.json'
    data = json.loads(summary.read_text())
    details = data['collision']['episodes_detail']
    # Preserve full source frames, temporal order, and recorded outcomes.
    for episode in (10,22):
        e = next(e for e in details if e['episode_idx']==episode)
        assert e['success']==1 and e['collision_free']==1
        src = Path(e['video_path'])
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(src)]))['streams'][0]
        nframes = int(probe['nb_frames'])
        for idx,fraction in enumerate((0,.2,.4,.6,.8,.98)):
            frame = min(nframes-1,round((nframes-1)*fraction))
            dest = OUT/'tasks'/f'v1011d_ep{episode:03d}_{idx+1:02d}_frame{frame:04d}.png'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),
                '-vf',f'select=eq(n\\,{frame})','-vsync','0','-frames:v','1',str(dest)],check=True)
            record(dest,src,'lossless_png_of_decoded_video_frame',
                   note='Full original video frame; native resolution, no generative enhancement. Successful collision-free v1011d PACT-raw rollout; separate protocol from historical hallway charts.',
                   episode=episode, frame_index=frame, total_frames=nframes,
                   source_summary=relative(summary), terminal_success=e['success'],collision_free=e['collision_free'])


def results():
    runs = ['place_corridor_vanilla_s0_n50','place_corridor_raw_s0_n50','place_corridor_readout_s0_n50_fast']
    labels = ['ACT','PACT-raw','PACT-readout']
    colors = ['#8794a8','#dc9642','#287db5']
    sources = [ROOT/'eval_output'/run/'eval_summary.json' for run in runs]
    data = [json.loads(p.read_text()) for p in sources]
    assert all(d['total']==50 and d['task_sampler_class']=='PactPlaceCorridorV2Sampler' for d in data)
    metrics = [('place_success',[d['success'] for d in data]),
               ('bar_hits',[d['collision']['bar_hits'] for d in data]),
               ('collision_free',[d['collision']['collision_free'] for d in data])]
    result_info = dict(protocol='Historical August 2026 hallway evaluation, 50 rollouts per arm, one training seed. Readout uses query-sampled history; not corrected-history or v1011d results.',
                       bar_order=labels, colors=colors, y_scale=[0,1],
                       grid_lines=[0,.25,.5,.75,1],error_bars='95% Wilson binomial intervals',metrics={})
    for name,counts in metrics:
        vals=np.array(counts)/50
        z=1.95996398454;n=50
        center=(vals+z*z/(2*n))/(1+z*z/n)
        delta=z*np.sqrt(vals*(1-vals)/n+z*z/(4*n*n))/(1+z*z/n)
        low,high=center-delta,center+delta
        fig,ax=plt.subplots(figsize=(6,4),dpi=400)
        fig.subplots_adjust(left=.03,right=.98,bottom=.04,top=.97)
        ax.set_ylim(0,1);ax.set_xlim(-.65,2.65)
        for y in [0,.25,.5,.75,1]:ax.axhline(y,color='#e4e8ee',lw=.8,zorder=0)
        ax.bar(range(3),vals,width=.58,color=colors,zorder=2)
        ax.errorbar(range(3),vals,yerr=[vals-low,high-vals],fmt='none',
                    ecolor='#263347',capsize=5,elinewidth=1.5,zorder=3)
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        dest=OUT/'results'/f'historical_hallway_{name}.png'
        fig.savefig(dest,dpi=400,facecolor='white')
        fig.savefig(dest.with_suffix('.svg'),facecolor='white')
        fig.savefig(dest.with_suffix('.pdf'),facecolor='white')
        plt.close(fig)
        record(dest,sources[0],'plot_from_saved_eval_json',note=result_info['protocol'],
               sources=[relative(p) for p in sources],bar_order=labels,counts=counts,total=50,
               error_bars='95% Wilson intervals',y_scale=[0,1])
        result_info['metrics'][name]=dict(counts=counts,rates=vals.tolist(),
                                       interval_low=low.tolist(),interval_high=high.tolist())
    (OUT/'results/plot_data.json').write_text(json.dumps(result_info,indent=2)+'\n')


def gallery():
    (OUT/'manifest.json').write_text(json.dumps(ITEMS,indent=2)+'\n')
    cards=[]
    for cat in CATEGORIES:
        cards.append(f'<section id="{cat}"><h2>{cat.title()}</h2><div class="grid">')
        for e in ITEMS:
            if e['category']!=cat:continue
            file=html.escape(e['file'],quote=True)
            title=html.escape(Path(e['file']).stem.replace('_',' '))
            note=html.escape(e['note'])
            source=html.escape('../'+e['source'],quote=True)
            cards.append(f'<article><a href="{file}"><img loading="lazy" src="{file}" alt="{title}"></a>'
                         f'<div class="caption"><a href="{file}" download>{title}</a>'
                         f'<p>{e["dimensions"][0]} × {e["dimensions"][1]} · {e["method"].replace("_"," ")}</p>'
                         f'<details><summary>Source and context</summary><p>{note}</p>'
                         f'<a href="{source}">Original source</a></details></div></article>')
        cards.append('</div></section>')
    markup='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper image collection</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#1c2b3e;font:15px system-ui,sans-serif}
header,main{max-width:1500px;margin:auto;padding:32px}h1{font-size:36px;margin:0 0 12px}p{line-height:1.55}
nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#216a9b;text-decoration:none}h2{font-size:26px;margin-top:40px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:22px}
article{background:white;border:1px solid #e2e6ec;border-radius:10px;overflow:hidden}img{width:100%;height:270px;object-fit:contain;background:white;display:block}
.caption{padding:18px}.caption p{font-size:12px;color:#617084}summary{cursor:pointer;font-size:12px}
@media(max-width:600px){header,main{padding:18px}}
</style><header><h1>Paper image collection</h1><p>Canonical robot renders, source environment views, recorded task frames, and data-backed result panels. Click an image to open its full resolution file. All labels here are outside the image files.</p>
<p>New robot, skin and sensor renders: 2400 pixels, 8× multisampling, white backgrounds, no text or logos. Source media remain at native resolution and can contain original manufacturer markings. Video frames have not been enlarged.</p>
<p>Results: historical August 2026 hallway protocol, 50 rollouts per model. Bars left to right: ACT, PACT-raw, PACT-readout. Common 0–100% scale; grid spacing 25%; whiskers are 95% Wilson intervals. See <a href="results/plot_data.json">plot data</a> and <a href="manifest.json">source manifest</a>. Task videos use a separate v1011d protocol.</p><nav>'''
    markup+=''.join(f'<a href="#{c}">{c.title()}</a>' for c in CATEGORIES)
    markup+='</nav></header><main>'+''.join(cards)+'</main></html>'
    (OUT/'index.html').write_text(markup)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['native','environments','sources','tasks','results','all'],default='all')
    args=parser.parse_args()
    for cat in CATEGORIES:(OUT/cat).mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest.json'
    if manifest.exists():ITEMS.extend(json.loads(manifest.read_text()))
    functions={'native':native_images,'environments':native_environments,'sources':curate,'tasks':tasks,'results':results}
    for key,fn in functions.items():
        if args.only in (key,'all'):fn()
    # Rebuilding a category replaces its manifest records without duplicating gallery cards.
    unique={e['file']:e for e in ITEMS}
    ITEMS[:]=sorted(unique.values(),key=lambda e:(CATEGORIES.index(e['category']),e['file']))
    gallery()
    print(f'{len(ITEMS)} PNG images; gallery: {OUT / "index.html"}',flush=True)


if __name__=='__main__':main()
