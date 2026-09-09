"""Mechanism checks for audit math and actual retained inference implementation."""
import ast
import textwrap
import tempfile
import unittest
from types import SimpleNamespace
from audit import *

class AuditTests(unittest.TestCase):
    def test_actual_aggregation_ages_zero_rows_and_expiry(self):
        source=(ROOT/'submodules/act/eval_pact_frontend_screen_row.py').read_text()
        body=source[source.index('        self._pending_chunks.append'):source.index('\n    def get_info',source.index('        self._pending_chunks.append'))]
        ns={'np':np}
        exec('def aggregate(self, chunk):\n'+body,ns)
        policy=SimpleNamespace(_step=0,_pending_chunks=[])
        for t in range(130):
            policy._step=t
            # Use the actual wrist adapter's H=100 bound before the current query.
            policy._pending_chunks=policy._pending_chunks[-99:]
            chunk=np.zeros((100,8),np.float32) if t==37 else np.full((100,8),t,np.float32)
            got=ns['aggregate'](policy,chunk)
            starts=np.arange(max(0,t-99),t+1)
            expected=np.average(np.where(starts==37,0,starts),weights=np.exp(-.01*(t-starts)))
            self.assertTrue(np.allclose(got,expected,atol=1e-5))
        self.assertEqual(len(policy._pending_chunks),100)

    def test_decode_after_affine_averaging_preserves_threshold(self):
        scores=np.array([0.,255.,0.]);weights=np.array([.25,.5,.25])
        for mean,std in [(153.,124.),(0.,.01)]:
            self.assertAlmostEqual(float((weights*((scores-mean)/std)).sum()*std+mean),127.5)
        self.assertEqual(int(np.where(np.array([127.499,127.5,127.501])<127.5,0,255).sum()),510)

    def test_wxyz_base_transform_mount_is_local(self):
        base=R.from_euler('xyz',[90,0,90],degrees=True)
        q_xyzw=base.as_quat();q_wxyz=q_xyzw[[3,0,1,2]]
        restored=R.from_quat(q_wxyz[[1,2,3,0]])
        point=np.array([.4,.1,.2]);origin=np.array([2.,3.,4.]);mount=np.array([0.,0.,.35])
        correct=restored.apply(point+mount)+origin
        self.assertFalse(np.allclose(correct,restored.apply(point)+origin+mount))
        self.assertTrue(np.allclose(restored.inv().apply(correct-origin)-mount,point))

    def test_contacts_overlap_and_pad_continuity(self):
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            path=Path(tmp)/'t.h5'
            names=[dict(names=['robot_0/gripper/left_pad','cavity_obj_0/Cup_10_cup_10_PrimitiveCollider_0'],**{'class':'grasp_target'}),
                   dict(names=['robot_0/gripper/right_pad','cavity_obj_0/Cup_10_cup_10_PrimitiveCollider_0'],**{'class':'grasp_target'}),
                   dict(names=['cavity_obj_0/Cup_10','pact_clutter_01'],**{'class':'clutter'}),
                   dict(names=['robot_0/arm','hazard'],**{'class':'hazard_bar'})]
            with h5py.File(path,'w') as h:
                h.create_dataset('contacts/class_entries',data=[[2,0,0],[0,1,1],[2,1,0]])
                h.create_dataset('contacts/pair_counts',data=[[0,0,1],[0,1,1],[1,2,1],[1,3,1],[2,0,1],[2,1,1],[2,2,1]])
                h.create_dataset('contacts/control_step',data=[0,1,1]);h.create_dataset('contacts/sim_time_s',data=[0,.002,.004])
                h.create_dataset('target/control_step_and_world_xyz',data=np.zeros((2,4)))
                h['contacts'].attrs['class_names']=json.dumps(['grasp_target','clutter','hazard_bar'])
                h['contacts'].attrs['pair_identities']=json.dumps(names)
            _,counts,_,_,_,masks,same,bilateral=raw_contacts(path)
            self.assertEqual(int(((counts[:,1]>0)|(counts[:,2]>0)).sum()),2)
            self.assertEqual(int((counts[:,1]>0).sum()+(counts[:,2]>0).sum()),3)
            self.assertEqual(int(masks['robot_clutter'].sum()),0)
            self.assertEqual(int(masks['cup_clutter'].sum()),2)
            self.assertEqual(maxrun(same),1)
            self.assertEqual(maxrun(bilateral),1)

    def test_actual_pair_rgb_boundary(self):
        tree=ast.parse((ROOT/'scripts/pact_wrist288_analysis.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rgb_difference')
        ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_rgb_difference','exec'),ns)
        a=np.zeros((20,50,3),np.uint8);b=a.copy();b.flat[:3]=2
        self.assertTrue(ns['rgb_difference'](a,b)[0]['passed'])
        b.flat[3]=1;self.assertFalse(ns['rgb_difference'](a,b)[0]['passed'])
        b=a.copy();b.flat[0]=3;self.assertFalse(ns['rgb_difference'](a,b)[0]['passed'])

if __name__=='__main__':unittest.main()
