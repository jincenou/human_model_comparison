"""Check resource policy without reading GPU state or starting GPU processes."""
import tempfile,unittest,fcntl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import launch_ours_source_pilot_v2 as q
class SchedulerTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);(self.root/'runs/rigid111_gpu_locks').mkdir(parents=True)
 def call(self,ready,disk=20):
  with patch.object(q,'P',self.root),patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=disk*1024**3)),patch.object(q.resources,'available',return_value=(ready,{'uuid':'mock'})):
   return q.acquire_available_slot()
 def test_reservations_do_not_hold_waiting_leases(self):
  slot,_=self.call(False);self.assertIsNone(slot)
  for i in range(8):
   with (self.root/f'runs/rigid111_gpu_locks/gpu_{i}.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def test_ready_keeps_exclusive_slot_until_released(self):
  slot,_=self.call(True);lease,gpu,_=slot
  try:
   with (self.root/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock').open('a') as other:
    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finally:lease.close()
 def test_disk_reserve(self):self.assertIsNone(self.call(True,9.9)[0])
if __name__=='__main__':unittest.main()
