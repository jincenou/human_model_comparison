"""Resume tests: no duplicated jobs, no silent retry of partial output."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common_human_orbit_batch import pending_groups,output_status,alive,start_ticks


class BatchResumeTest(unittest.TestCase):
    def test_finished_running_and_partial_are_not_dispatched(self):
        rows=[dict(id=sid,method='ours',status='scored')for sid in ['done','live','partial','ready','attempted']]
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'partial/ours').mkdir(parents=True)
            with patch('common_human_orbit_batch.output_status',side_effect=lambda r,s,m:'rendered'if s=='done'else('partial_requires_inspection'if s=='partial'else'missing')):
                groups,inventory=pending_groups(rows,['ours'],root,{'live'},{('attempted','ours')})
            self.assertEqual(dict(groups),{'ready':['ours']})
            self.assertEqual(output_status(root,'partial','ours'),'partial_requires_inspection')

    def test_unavailable_is_kept_in_denominator_not_dispatched(self):
        rows=[dict(id='person',method='trellis2',status='source_registration_unavailable')]
        groups,inventory=pending_groups(rows,['trellis2'],Path('/unused'),set(),set())
        self.assertFalse(groups);self.assertEqual(len(inventory),1)
        self.assertEqual(inventory[0]['status'],'source_unavailable:source_registration_unavailable')

    def test_pid_reuse_is_not_live_handle(self):
        import os
        pid=os.getpid()
        self.assertFalse(alive(dict(pid=pid,start_ticks=start_ticks(pid)+1,id='HN0004')))
        self.assertFalse(alive(dict(pid=pid,start_ticks=start_ticks(pid),id='HN0004')))

    def test_incomplete_manifest_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);folder=root/'HN0004/ours';folder.mkdir(parents=True)
            (folder/'manifest.json').write_text(json.dumps(dict(id='HN0004',method='ours')))
            self.assertEqual(output_status(root,'HN0004','ours'),'invalid_requires_inspection')


if __name__=='__main__':unittest.main()
