"""Plot recorded signals only. Never reconstruct or advance a simulator."""
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = Path('/root/prox_learning_pact_remediation')
OUT = Path(__file__).parent
doc = json.loads((OUT/'reconstruction.json').read_text())
selected = [('ACT', '024_', 'No target contact'), ('PACT', '002_', 'No target contact'),
            ('ACT', '014_', 'Two isolated held samples'), ('PACT', '005_', 'Prolonged contact-only held'),
            ('PACT', '030_', 'Near receptacle, never supported'), ('PACT', '031_', 'Transient success lost')]
fig, axes = plt.subplots(len(selected), 3, figsize=(14, 18), layout='constrained')
records = []
for i, (arm, prefix, title) in enumerate(selected):
    row = next(r for r in doc['versions']['V10.10']['rows'] if r['arm'] == arm and Path(r['directory']).name.startswith(prefix))
    directory = ROOT/row['directory']
    def decode(x):
        return json.loads(x.tobytes().split(b'\0', 1)[0])
    with h5py.File(directory/'trajectory.h5', 'r') as h:
        g = h['traj_0']; extra = g['obs/extra']
        ti = [decode(x) for x in extra['task_info'][()]]
        gs = [decode(x)['gripper'] for x in extra['grasp_state_pickup_obj'][()]]
        b = extra['robot_base_pose'][()].astype(float); tcp = extra['tcp_pose'][()].astype(float)
        world = Rotation.from_quat(b[:, [4,5,6,3]]).apply(tcp[:,:3])+b[:,:3]
        target = extra['obj_start'][0,:3]
        scene = json.loads(g['obs_scene'][()]); tray = scene['scene_params']['place_receptacle_start_pose'][:2]
        seen = extra['object_image_points/pickup_obj/wrist_camera/num_points'][()].reshape(-1)>0
        commanded = [decode(x)['gripper'][0] for x in g['actions/commanded_action'][1:]]
    with np.load(directory/'actions.npz', allow_pickle=False) as a:
        grip = a['gripper'].reshape(-1)
        assert np.array_equal(grip, commanded)
        assert np.array_equal(grip, np.where(a['model_output'][:,7]<127.5,0,255))
    close = np.r_[False, grip >= 127.5]
    switch = np.flatnonzero(np.diff(close.astype(int)))+1
    record = {'directory': row['directory'], 'title': title,
              'gripper_command_transitions_observation_steps': [{'step':int(t), 'close':bool(close[t])} for t in switch],
              'first_tray_step':row['first_tray'], 'held_runs':row['held_runs'], 'touch_runs':row['touch_runs'],
              'final_task_info':row['final_task_info']}
    records.append(record)
    ax=axes[i,0]
    ax.plot(world[:,0],world[:,1],lw=1.2,label='Recorded TCP')
    ax.scatter(*world[0,:2],c='black',s=20,label='TCP start')
    ax.scatter(*target[:2],marker='x',c='red',s=60,label='Initial target origin')
    ax.scatter(*tray,marker='s',c='green',s=30,label='Tray XY reference')
    ax.add_patch(plt.Circle(tray,.1,fill=False,color='green',ls=':',lw=1))
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'{arm} {prefix[:-1]}: {title}')
    ax.axis('equal'); ax.grid(alpha=.2)
    if i==0: ax.legend(fontsize=7)
    ax=axes[i,1]
    ax.plot(np.arange(901)*.066,[x['position_error'] for x in ti],label='Target AABB to tray center')
    ax.plot(np.arange(901)*.066,np.linalg.norm(world[:,:2]-tray,axis=1),label='TCP to tray (XY)',alpha=.8)
    ax.set(xlabel='Time (s)',ylabel='Recorded distance (m)');ax.grid(alpha=.2)
    if i==0: ax.legend(fontsize=7)
    ax=axes[i,2]
    signals=[('Wrist target seen',seen),('Gripper touching',[x['touching'] for x in gs]),
             ('Held (contact only)',[x['held'] for x in gs]),('Receptacle support',[x['supported_by_receptacle'] for x in ti]),
             ('Task success',[x['success'] for x in ti]),('Close command',close)]
    for offset,(label,values) in enumerate(signals):
        ax.step(np.arange(901)*.066,np.asarray(values)*.65+offset,where='post',lw=1)
    ax.set(yticks=np.arange(6)+.3,yticklabels=[s[0] for s in signals],xlabel='Time (s)')
    ax.tick_params(axis='y',labelsize=7);ax.grid(axis='x',alpha=.2)
fig.suptitle('V10.10 repaired evaluation: recorded failed trajectories\nHand paths and contact signals do not reconstruct object lift or stable grasp',fontsize=14)
assert not (OUT/'representatives.png').exists()
fig.savefig(OUT/'representatives.png',dpi=135)
flags={k:v for k,v in doc.items() if k.startswith('authorizes_') or k in ('eligible_for_human_review','human_approval_present','phase0_passed')}
with (OUT/'representatives.json').open('x') as f:
    json.dump({**flags,'scope':'Retained signals and original 127.5 gripper threshold; no inference or rollout.',
               'action_timing':'Observation step k>0 has the command from actions.npz[k-1]; step zero is initial.',
               'rows':records},f,indent=2)
print(json.dumps(records,indent=2))
