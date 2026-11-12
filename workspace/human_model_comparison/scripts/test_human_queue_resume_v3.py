"""Exercise takeover and disk guards in a temporary queue; never launch GPU work."""
import fcntl,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import human_native_queue_v3 as q

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
    def run_once(self,disk=11.0,ready=True,old_alive=False):
        q.save(self.queue/'state.json',self.state)
        with patch.object(q,'Q',self.queue),patch.object(q,'P',self.root),patch.object(q,'H',self.root), \
             patch.object(q.resources,'live',return_value=old_alive),patch.object(q,'legacy_owns',return_value=False), \
             patch.object(q,'shared_slot_available',return_value=(ready,dict(uuid='GPU-test'))), \
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
    def audited_failures(self):
        reviews=[]
        for sid in ['HN0292','HN0834']:
            key='trellis2_'+sid;record=self.queue/'workers'/(key+'.json');log=self.queue/'logs'/(key+'.log')
            q.save(record,dict(status='failed',finished_at=1));log.write_text('native empty sparse structure')
            job=dict(key=key,method='trellis2',sample_id=sid,status='failed',finished_at=1,worker_record=str(record),log=str(log))
            self.state['jobs'].append(job)
            reviews.append(dict(key=key,method='trellis2',classification='native_empty_sparse_structure',continue_pending_jobs=True,retry_failed_job=False,finished_at=1,worker_record=str(record),worker_record_sha256=q.sha(record),log=str(log),log_sha256=q.sha(log)))
        q.save(self.queue/'failure_review.json',dict(status='audited_sample_failures',failures=reviews))
    def test_audited_failures_stay_failed_while_pending_job_runs(self):
        self.audited_failures();state,n=self.run_once()
        self.assertEqual(n,1);self.assertEqual(state['jobs'][0]['status'],'running')
        self.assertTrue(all(j['status']=='failed'for j in state['jobs'][1:]))
    def test_unaudited_failures_still_stop_new_dispatch(self):
        self.audited_failures();(self.queue/'failure_review.json').unlink()
        state,n=self.run_once();self.assertEqual(n,0);self.assertEqual(state['jobs'][0]['status'],'queued')
    def test_changed_failure_evidence_is_rejected(self):
        self.audited_failures();Path(self.state['jobs'][1]['log']).write_text('changed failure')
        with self.assertRaises(AssertionError):self.run_once()

class ConcurrencyTest(unittest.TestCase):
    def test_disk_reserves_inflight_outputs(self):
        self.assertFalse(q.disk_launch_allowed(10.4, []))
        self.assertFalse(q.disk_launch_allowed(12.0, [{'status':'running'}]*4))
        self.assertTrue(q.disk_launch_allowed(16.0, [{'status':'running'}]*7))
    def test_sharing_occupied_but_preserves_reservations(self):
        with patch.object(q.resources,'available',return_value=(False,dict(free_mib=41000,reservations=[],uuid='x'))):
            self.assertTrue(q.shared_slot_available('2')[0])
        with patch.object(q.resources,'available',return_value=(False,dict(free_mib=41000,reservations=[{'pid':1}],uuid='x'))):
            self.assertFalse(q.shared_slot_available('2')[0])
        with patch.object(q.resources,'available',return_value=(False,dict(free_mib=17000,reservations=[],uuid='x'))):
            self.assertFalse(q.shared_slot_available('2')[0])
    def test_eight_method_slots(self):
        self.assertEqual(q.MAX_METHOD_WORKERS,8)

if __name__=='__main__':unittest.main()
