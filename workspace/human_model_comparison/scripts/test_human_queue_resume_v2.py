"""Exercise takeover and disk guards in a temporary queue; never launch GPU work."""
import fcntl,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import human_native_queue_v2 as q

class StopIterationForTest(Exception): pass
class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.queue=self.root/'queue';self.queue.mkdir()
        for p in ('workers','logs'): (self.queue/p).mkdir()
        (self.root/'runs/rigid111_gpu_locks').mkdir(parents=True)
        self.selection=self.root/'selection.json';self.selection.write_text('{}')
        self.plan=dict(selection=str(self.selection),selection_sha256=q.sha(self.selection),script_hashes={})
        q.save(self.queue/'plan.json',self.plan)
        self.job=dict(key='trellis2_HN0004',method='trellis2',sample_id='HN0004',status='queued')
        self.state=dict(pid=99999999,plan_sha256=q.sha(self.queue/'plan.json'),jobs=[self.job],slots={})
    def run_once(self,disk=10.5,ready=True,old_alive=False):
        q.save(self.queue/'state.json',self.state)
        with patch.object(q,'Q',self.queue),patch.object(q,'P',self.root),patch.object(q,'H',self.root), \
             patch.object(q.resources,'live',return_value=old_alive),patch.object(q,'legacy_owns',return_value=False), \
             patch.object(q.resources,'available',return_value=(ready,dict(uuid='GPU-test'))), \
             patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=disk*1024**3)), \
             patch.object(q.time,'sleep',side_effect=StopIterationForTest), \
             patch.object(q.subprocess,'Popen',return_value=SimpleNamespace(pid=88888888,poll=lambda:None)) as spawn:
            try:q.run()
            except StopIterationForTest:pass
            return q.read(self.queue/'state.json'),spawn.call_count
    def test_below_reserve_never_launches(self):
        state,n=self.run_once(disk=9.9);self.assertEqual(n,0);self.assertEqual(state['jobs'][0]['status'],'queued')
    def test_above_ten_below_twenty_can_launch(self):
        state,n=self.run_once();self.assertEqual(n,1);self.assertEqual(state['disk_launch_floor_gib'],10)
        self.assertEqual(state['jobs'][0]['status'],'running')
    def test_live_old_controller_blocks_takeover(self):
        with self.assertRaisesRegex(AssertionError,'Old controller'):self.run_once(old_alive=True)
    def test_existing_reservation_prevents_launch(self):
        state,n=self.run_once(ready=False);self.assertEqual(n,0)
    def test_locked_gpu_is_skipped(self):
        leases=[]
        try:
            for i in range(8):
                f=(self.root/f'runs/rigid111_gpu_locks/gpu_{i}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(f)
            _,n=self.run_once();self.assertEqual(n,0)
        finally:
            for f in leases:f.close()
    def test_adopt_live_worker_without_duplicate(self):
        self.job.update(status='running',child_pid=os.getpid(),gpu='0')
        _,n=self.run_once();self.assertEqual(n,0)
    def test_completed_adopted_worker_requires_validation_hash(self):
        self.job.update(status='running',child_pid=99999999,gpu='0')
        validation=self.root/'validation.json';validation.write_text('{"status":"passed"}')
        q.save(self.queue/'workers'/f"{self.job['key']}.json",dict(status='native_passed',finished_at=1,validation=str(validation),validation_sha256=q.sha(validation)))
        state,n=self.run_once();self.assertEqual(n,0);self.assertEqual(state['jobs'][0]['status'],'native_passed')
        validation.write_text('{"status":"changed"}')
        with self.assertRaises(AssertionError):self.run_once()
    def test_pinned_input_change_blocks_takeover(self):
        self.selection.write_text('{"changed":true}')
        with self.assertRaises(AssertionError):self.run_once()

if __name__=='__main__':unittest.main()
