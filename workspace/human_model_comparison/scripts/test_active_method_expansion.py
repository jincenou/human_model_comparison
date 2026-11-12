"""Adding partial method coverage must preserve the original paired experiment."""
import copy
import tempfile
import unittest
from pathlib import Path

import human_metric_batch_v1 as batch
from run_human_metric_pipeline_v1 import (
    active_pairing_groups, apply_registration_missingness, apply_native_generation_failures,
    expand_active_methods, scoring_methods,
)


class ActiveMethodExpansion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.active = batch.read(batch.H/'inputs/active_human_benchmark.json')
        cls.current = batch.read(batch.H/'reports/source_camera_metric_cohort_v3/summary.json')
        cls.ids = [r['id'] for r in batch.read(cls.active['selection'])['objects']]
        # Keep the original comparison regardless of a concurrently published
        # sixth-method index. All operations below are in-memory or temporary.
        cls.original = copy.deepcopy(cls.current)
        cls.original['coverage'] = {m:cls.current['coverage'][m] for m in batch.DEFAULT_METHODS}
        cls.original['rows'] = [r for r in cls.original['rows'] if r['method'] in batch.DEFAULT_METHODS]

    def test_all_100_new_rows_without_changing_existing_results(self):
        summary = copy.deepcopy(self.original)
        before = copy.deepcopy(summary['rows'])
        certificates = {(self.ids[0], 'trellis2'): {'registration_record': 'already-verified-by-caller'}}
        self.assertEqual(expand_active_methods(summary, self.active, self.ids, certificates), ['trellis2'])
        self.assertEqual(len(summary['rows']), 600)
        self.assertEqual(summary['rows'][:500], before)
        self.assertEqual(summary['coverage']['trellis2'], 0)
        new = summary['rows'][500:]
        self.assertEqual({r['id'] for r in new}, set(self.ids))
        expected = {r['id']:r for r in batch.read(self.active['reference_qualification'])['objects']}
        for row in new:
            self.assertEqual(row['reference'], expected[row['id']])
            self.assertFalse(row['metric_eligible'])
            self.assertNotIn('result', row)
        self.assertEqual(expand_active_methods(summary, self.active, self.ids, certificates), [])
        self.assertEqual(len(summary['rows']), 600)

    def test_partial_sixth_method_preserves_five_method_common_set(self):
        by_method = {m:{} for m in batch.DEFAULT_METHODS}
        for row in self.original['rows']:
            if row['status'] == 'scored':
                by_method[row['method']][row['id']] = batch.read(row['result'])
        original = {name:batch.m.paired_summary(by_method, name, self.ids)
                    for name in ['PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw']}
        extended = dict(by_method, trellis2={})
        groups = active_pairing_groups(list(extended))
        self.assertIn(batch.DEFAULT_METHODS, groups)
        self.assertIn(list(extended), groups)
        self.assertIn(['ours', 'trellis2'], groups)
        self.assertIn(['trellis_original', 'trellis2'], groups)
        self.assertIn(['omnire', 'splatad'], groups)
        self.assertIn(['ours', 'omnire', 'splatad'], groups)
        for name, prior in original.items():
            five = batch.m.paired_summary({m:extended[m] for m in batch.DEFAULT_METHODS}, name, self.ids)
            self.assertEqual(five, prior)
            six = batch.m.paired_summary(extended, name, self.ids)
            self.assertEqual(six['common_ids'], [])
            for item in six['paired'].values():
                self.assertEqual(item['denominator'], 100)
                self.assertEqual(item['n'], 0)
        # Even one newly available score must not reduce the original common set.
        sid = original['PSNR']['common_ids'][0]
        extended['trellis2'][sid] = copy.deepcopy(by_method['ours'][sid])
        six = batch.m.paired_summary(extended, 'PSNR', self.ids)
        self.assertEqual(six['common_ids'], [sid])
        five = batch.m.paired_summary({m:extended[m] for m in batch.DEFAULT_METHODS}, 'PSNR', self.ids)
        self.assertEqual(five, original['PSNR'])

    def test_current_reference_hash_and_population_required(self):
        certs = {(self.ids[0], 'trellis2'): {}}
        bad_hash = dict(self.active, reference_qualification_sha256='wrong')
        with self.assertRaises(ValueError):
            expand_active_methods(copy.deepcopy(self.original), bad_hash, self.ids, certs)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'incomplete_reference.json'
            q = copy.deepcopy(batch.read(self.active['reference_qualification']))
            q['objects'].pop(); batch.save(p, q)
            invalid = dict(self.active, reference_qualification=str(p), reference_qualification_sha256=batch.m.sha(p))
            with self.assertRaises(ValueError):
                expand_active_methods(copy.deepcopy(self.original), invalid, self.ids, certs)
        with self.assertRaises(ValueError):
            expand_active_methods(copy.deepcopy(self.original), self.active, self.ids, {(self.ids[0], 'not_registered'): {}})

    def test_default_and_explicit_method_scope(self):
        available = batch.DEFAULT_METHODS + ['trellis2']
        self.assertEqual(scoring_methods(None, available), available)
        self.assertEqual(scoring_methods(['trellis2'], available), ['trellis2'])
        with self.assertRaises(ValueError):
            scoring_methods(['trellis2', 'trellis2'], available)
        with self.assertRaises(ValueError):
            scoring_methods(['not_registered'], available)

    def test_frozen_metric_implementations_unchanged(self):
        self.assertEqual(batch.m.sha(batch.m.__file__), '1650c142d165c4e97c1f24ef5ded6a1452e9d661ddae5846000518c6f3a74f91')
        self.assertEqual(batch.m.sha(batch.__file__), '879dde24d94c6e7672cf6379087bb1d239c78dd21dfbea90050077d2551d56a6')

    def test_method_missingness_preserves_reference_and_other_methods(self):
        reference = {'reference_eligible': True, 'condition': 'umbrella_occlusion'}
        row = dict(id='example', method='trellis2', status='registration_pending',
                   metric_eligible=False, reference=reference)
        other = dict(row, method='ours', status='scored', metric_eligible=True)
        summary = {'rows': [row, copy.deepcopy(other)]}
        missing = [dict(id='example', method='trellis2', status='unavailable',
                        reason='fewer_than_six_correspondences')]
        apply_registration_missingness(summary, missing, {})
        self.assertEqual(row['status'], 'source_registration_unavailable')
        self.assertEqual(row['reason'], missing[0]['reason'])
        self.assertTrue(row['reference']['reference_eligible'])
        self.assertEqual(summary['rows'][1], other)

    def test_new_certificate_and_existing_score_override_stale_missingness(self):
        rows = [dict(id='ready', method='trellis2', status='registration_pending'),
                dict(id='done', method='trellis2', status='scored', result='existing')]
        summary = {'rows': copy.deepcopy(rows)}
        missing = [dict(id=r['id'], method=r['method'], status='unavailable') for r in rows]
        apply_registration_missingness(summary, missing, {('ready', 'trellis2'): {}})
        self.assertEqual(summary['rows'], rows)

    def test_native_failures_preserve_references_scores_and_new_certificates(self):
        with tempfile.TemporaryDirectory()as td:
            jobs=[]
            for sid in ['failed','ready','scored','stale']:
                path=Path(td)/(sid+'.json');batch.save(path,dict(key='trellis2_'+sid,status='failed',finished_at=1,reason='generate failed'))
                jobs.append(dict(key='trellis2_'+sid,sample_id=sid,method='trellis2',status='failed',finished_at=2 if sid=='stale'else 1,worker_record=str(path)))
            ref=dict(reference_eligible=True,condition='umbrella')
            rows=[dict(id=sid,method='trellis2',status='scored'if sid=='scored'else'registration_pending',reference=ref.copy())for sid in ['failed','ready','scored','stale']]
            summary=dict(rows=rows);apply_native_generation_failures(summary,dict(jobs=jobs),{('ready','trellis2'):{}})
            self.assertEqual([r['status']for r in rows],['native_generation_failed','registration_pending','scored','registration_pending'])
            self.assertTrue(all(r['reference']==ref for r in rows))
            self.assertFalse(rows[0]['metric_eligible']);self.assertEqual(rows[0]['reason'],'generate failed')


if __name__ == '__main__':
    unittest.main()
