#!/usr/bin/env python3
"""Reconstruct link-6 range geometry directly from HDF5. No RGB or scene meshes.

Use mlspaces Python (h5py, numpy, matplotlib). Four depth planes are temporal
substeps, not independent rays. Use the last raw plane, never a temporal mean.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import csv
os.environ.setdefault('MPLCONFIGDIR', '/tmp/prox-link6-mpl')
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'images/link6_reconstruction'
SOURCE = Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v1011d/rows/000_187ba0ce76cb3011/trajectory.h5')
NAMES = [f'link6_sensor_{i}' for i in range(6)]
SNAPSHOT = 86
LAST = 105  # Last pregrasp observation, before phase "grasp" begins at 106.
PALETTE = ['#087f8c','#3964ad','#7762a3','#b0753a','#348775','#407dba']
CMAP = LinearSegmentedColormap.from_list('height', ['#253b62','#087d91','#55afa5'])
BOUNDS = np.array([[.45,-.5,.70], [1.40,.5,1.55]])


def read():
    """Return all 384 points per step with an exact HDF5 address per sample."""
    xyz, depths, sensor, frames, pixels, origins = [], [], [], [], [], []
    v,u = np.indices((8,8))
    with h5py.File(SOURCE, 'r') as f:
        g = f['traj_0']
        for t in range(LAST+1):
            for i,name in enumerate(NAMES):
                raw = g[f'obs/proximity/{name}'][t]
                assert raw.shape == (4,8,8)
                d = raw[-1].astype(np.float64)
                assert np.isfinite(d).all() and (d > 0).all()
                prm = g[f'obs/sensor_param/{name}']
                k = prm['intrinsic_cv'][t].astype(np.float64)
                pose = prm['cam2world_gl'][t].astype(np.float64)
                ext = prm['extrinsic_cv'][t].astype(np.float64)
                assert np.allclose(ext @ pose, np.eye(4)[:3], atol=1e-5)
                # Native MuJoCo 8x8 samples lie at pixel centres. Recorded K has
                # cx=cy=4 at the image boundary coordinate convention: u+0.5.
                pc = np.stack([(u+.5-k[0,2])*d/k[0,0],
                               (v+.5-k[1,2])*d/k[1,1],d,np.ones_like(d)],-1).reshape(64,4)
                world = (pose @ pc.T).T[:,:3]
                recovered = world @ ext[:,:3].T + ext[:,3]
                assert np.allclose(recovered, pc[:,:3], atol=1e-5)
                xyz.extend(world); depths.extend(d.ravel()); sensor.extend([i]*64)
                frames.extend([t]*64); pixels.extend(np.stack([u.ravel(),v.ravel()],1))
                origins.extend([pose[:3,3]]*64)
        phase = g['obs/extra/policy_phase'][:LAST+2]
        assert phase[LAST+1] == 3  # grasp
    return dict(points=np.array(xyz), depth_m=np.array(depths), sensor_id=np.array(sensor),
                frame=np.array(frames), pixel_uv=np.array(pixels), origins=np.array(origins))


def subset(data, mask):
    return {k:v[mask] for k,v in data.items()}


def ply(name, data):
    header = ['ply','format ascii 1.0',f'element vertex {len(data["points"])}',
              'property float x','property float y','property float z',
              'property float depth_m','property int frame','property uchar sensor_id',
              'property uchar pixel_u','property uchar pixel_v','end_header']
    lines = [f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {d:.8f} {t} {i} {uv[0]} {uv[1]}'
             for p,d,t,i,uv in zip(data['points'],data['depth_m'],data['frame'],data['sensor_id'],data['pixel_uv'])]
    (OUT/f'{name}.ply').write_text('\n'.join(header+lines)+'\n')


def project(points, azimuth=-145, elevation=22):
    """Orthographic display transform only; never change reconstructed points."""
    az,el = np.deg2rad([azimuth,elevation])
    eye = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right = np.array([-np.sin(az),np.cos(az),0])
    up = np.cross(eye,right)
    p = points-points.mean(0)
    return p @ right, p @ up, p @ eye


def plate(name, data, snapshot=False, azimuth=-145, elevation=22):
    p = data['points']
    x,y,z = project(p,azimuth,elevation)
    order = np.argsort(z)
    colors = np.array(PALETTE)[data['sensor_id']] if snapshot else CMAP(np.clip((p[:,2]-.70)/.85,0,1))
    fig,ax = plt.subplots(figsize=(7,5.2))
    fig.subplots_adjust(0,0,1,1)
    ax.scatter(x[order],y[order],s=9 if snapshot else .65,c=colors[order],
               alpha=1 if snapshot else .88,linewidths=0,edgecolors='none')
    ax.set_aspect('equal'); ax.axis('off'); ax.margins(.035)
    for suffix in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=400,transparent=True,
                    bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    # Editable 2D coordinates for consumers that cannot manipulate a 3D PLY.
    xy = np.stack([x,y],1)
    lo,hi = xy.min(0),xy.max(0)
    (OUT/f'{name}_layout.json').write_text(json.dumps(dict(
        view_azimuth=azimuth,view_elevation=elevation,
        points_xy=xy.tolist(),order=order.tolist(),bounds=[lo.tolist(),hi.tolist()],
        sensor_id=data['sensor_id'].tolist(),frame=data['frame'].tolist()),separators=(',',':')))


def html_view(data, hood):
    # Self-contained interactive viewer. Rotation/zoom changes only the view.
    payload = dict(points=np.round(data['points'],7).tolist(), sensors=data['sensor_id'].tolist(),
                   frames=data['frame'].tolist(), hood=hood.astype(int).tolist(), snapshot=SNAPSHOT)
    template = '''<!doctype html><meta charset="utf-8"><title>Link 6: recorded range reconstruction</title>
<style>body{margin:0;font:15px system-ui;color:#253b62;background:white}header{padding:18px 24px;border-bottom:1px solid #ddd}button,select,input{margin:0 10px;padding:6px}canvas{display:block;width:100%;height:75vh;touch-action:none}p{margin:6px 24px;line-height:1.6}a{color:#087d91}</style>
<header><b>Link 6 · recorded 8 × 8 range measurements</b>
<select id="mode"><option value="scan">Accumulated approach · frames 0–105</option><option value="single">Single frame · 384 raw samples</option></select>
<select id="sensor"><option value="all">All six sensors</option><option value="0">Sensor 0</option><option value="1">Sensor 1</option><option value="2">Sensor 2</option><option value="3">Sensor 3</option><option value="4">Sensor 4</option><option value="5">Sensor 5</option></select>
<label><input type="checkbox" id="crop" checked>Fume-hood crop</label><button id="reset">Reset view</button><button id="save">Save PNG</button></header>
<canvas id="view"></canvas><p id="status"></p><p>Drag to rotate · scroll to zoom. Each dot is one stored range sample. No RGB, scene meshes, fitted surfaces, interpolation, or invented points. The accumulated scan contains partial observations; gaps are unobserved geometry.</p>
<p><a href="link6_snapshot_384.ply">384-point snapshot</a> · <a href="link6_approach_all.ply">Full accumulated cloud</a> · <a href="link6_hood_scan.png">Paper PNG</a> · <a href="link6_hood_scan.svg">Vector SVG</a></p>
<script>const D=__DATA__,c=document.getElementById('view'),ctx=c.getContext('2d'),mode=document.getElementById('mode'),crop=document.getElementById('crop'),sensor=document.getElementById('sensor');
let az=-145*Math.PI/180,el=22*Math.PI/180,zoom=1,drag=null;const pal=['#087f8c','#3964ad','#7762a3','#b0753a','#348775','#407dba'];
function draw(){let dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
let ids=[];for(let i=0;i<D.points.length;i++)if((mode.value==='scan'||D.frames[i]===D.snapshot)&&(!crop.checked||D.hood[i])&&(sensor.value==='all'||D.sensors[i]===Number(sensor.value)))ids.push(i);
let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(let i of ids)D.points[i].forEach((v,j)=>{lo[j]=Math.min(lo[j],v);hi[j]=Math.max(hi[j],v)});
let center=lo.map((v,j)=>(v+hi[j])/2),extent=Math.hypot(...hi.map((v,j)=>v-lo[j])),scale=Math.min(w,h)*.90/extent*zoom;
let ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),rows=[];
for(let i of ids){let [x,y,z]=D.points[i].map((v,j)=>v-center[j]);rows.push([(-sa*x+ca*y)*scale+w/2,-(-se*ca*x-se*sa*y+ce*z)*scale+h/2,ce*ca*x+ce*sa*y+se*z,i])}rows.sort((a,b)=>a[2]-b[2]);
ctx.globalAlpha=mode.value==='scan'?.82:1;for(let [x,y,z,i]of rows){ctx.fillStyle=pal[D.sensors[i]];let r=mode.value==='scan'?1.0:2.8;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;
document.getElementById('status').textContent=ids.length.toLocaleString()+' measured points displayed · '+(mode.value==='single'?'frame '+D.snapshot+' · six sensors × 64 samples before crop':'106 frames · 40,704 samples before crop');}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.007;el=Math.max(-1.5,Math.min(1.5,el+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw()};mode.onchange=()=>{crop.checked=mode.value==='scan';draw()};crop.onchange=sensor.onchange=draw;document.getElementById('reset').onclick=()=>{az=-145*Math.PI/180;el=22*Math.PI/180;zoom=1;draw()};document.getElementById('save').onclick=()=>{let a=document.createElement('a');a.download='link6-measured-cloud.png';a.href=c.toDataURL('image/png');a.click()};window.onresize=draw;draw();</script>'''
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'))))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = read()
    assert data['points'].shape == (106*6*64,3)
    snap = subset(data,data['frame']==SNAPSHOT)
    assert snap['points'].shape == (384,3)
    with (OUT/'link6_snapshot_384.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sensor','frame','substep','pixel_u','pixel_v','raw_depth_m','world_x_m','world_y_m','world_z_m'])
        for p,d,t,i,uv in zip(snap['points'],snap['depth_m'],snap['frame'],snap['sensor_id'],snap['pixel_uv']):
            writer.writerow([NAMES[i],t,3,*uv,d,*p])
    hood = ((data['points'] >= BOUNDS[0]) & (data['points'] <= BOUNDS[1])).all(1)
    cropped = subset(data,hood)
    np.savez_compressed(OUT/'link6_measurements.npz',**data,hood_mask=hood,sensor_names=NAMES)
    for name,d in [('link6_snapshot_384',snap),('link6_approach_all',data),('link6_hood_scan',cropped)]:
        ply(name,d)
    plate('link6_snapshot_384',snap,snapshot=True)
    plate('link6_hood_scan',cropped)
    # Obstacle detail: a spatial crop of actual returns around the protruding
    # panel. The bounds select measurements; no box or object mesh is drawn.
    q=data['points']
    detail=(q[:,0]>.52)&(q[:,0]<.74)&(q[:,1]>-.48)&(q[:,1]<-.07)&(q[:,2]>.76)&(q[:,2]<1.06)
    obstacle=subset(data,detail)
    plate('link6_obstacle_detail',obstacle,azimuth=-135,elevation=25)
    ply('link6_obstacle_detail',obstacle)
    html_view(data,hood)
    manifest=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        trajectory='traj_0',sensor_names=NAMES,snapshot_frame=SNAPSHOT,
        snapshot_samples=384,accumulation_frames=[0,LAST],accumulated_samples=len(data['points']),
        hood_samples=int(hood.sum()),obstacle_detail_samples=int(detail.sum()),hood_bounds_m=BOUNDS.tolist(),
        hdf5_address='traj_0/obs/proximity/{sensor}[frame,3,pixel_v,pixel_u]',
        backprojection='Latest raw substep only. X=(u+0.5-cx)*depth/fx; Y=(v+0.5-cy)*depth/fy; Z=depth. Transform by recorded cam2world_gl (CV convention).',
        calibration_check='Recorded extrinsic_cv inverts cam2world_gl; world-to-sensor roundtrip within 1e-5 m.',
        caveats=['End-of-policy-step poses paired with latest available substep; no substep poses stored.',
                 'Accumulation is 0–105 inclusive (before grasp starts at 106); no interpolation, meshing, filling, or temporal averaging.',
                 'Static surfaces can accumulate coherently; robot motion can leave traces. Not a complete object reconstruction.',
                 'PNG/SVG colour encodes height in world coordinates; snapshot and interactive viewer use sensor identity.',
                 'Raw readings are from recorded simulation, not physical sensor hardware.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'caption.txt').write_text('Sensor-only reconstruction from six link-6 proximity sensors. Each snapshot contains six raw 8×8 depth arrays (384 measured samples). Accumulating the latest reading at each of 106 recorded approach steps, transformed using the stored sensor poses, reveals observed parts of the fume-hood surfaces and protruding obstacle. No RGB, depth from external/wrist cameras, scene meshes, fitted surfaces or interpolated points contribute to this reconstruction. The cloud is partial: unobserved surfaces remain empty.\n')
    # Replace the previously rejected standalone overlay with the sensor-only scan.
    for ext in ['png','svg','pdf']:
        shutil.copy2(OUT/f'link6_hood_scan.{ext}',ROOT/f'images/first_page/pointcloud_panel.{ext}')
    print(json.dumps({k:manifest[k] for k in ['snapshot_samples','accumulated_samples','hood_samples','obstacle_detail_samples']},indent=2))


if __name__ == '__main__':
    main()
