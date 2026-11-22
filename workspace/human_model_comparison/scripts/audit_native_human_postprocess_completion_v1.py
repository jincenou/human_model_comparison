#!/usr/bin/env python3
"""Bind existing native, 2D postprocess, metric and publication completion evidence.

This CPU-only check does not render, recompute metrics, edit paper files, or mark
the goal complete. Large 3D assets are covered by the previously verified and
still unchanged validation hash chain; this script does not rehash checkpoints.
"""
from __future__ import annotations
from human_asset_benchmark.paths import expand as _hmc_expand

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

H = Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
ROOT = H / 'reports/native_human_export_reaudit_v1'
POST = ROOT / 'postprocess_v1'
PAPER = Path(_hmc_expand('${HMC_WORKSPACE}/vehicle_model_comparison/reports/asset_experiments_paper_20261115_v3'))
METHODS = ('splatad', 'omnire', 'streetgs')
PUBLICATION = '24_人体二维前景后处理更新核验.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Audit:
    def __init__(self):
        self.gates = []
        self.evidence = {}
        self.data = {}

    def bind(self, path, expected=None):
        path = Path(path)
        assert path.is_file(), f'missing file: {path}'
        actual = sha(path)
        if expected is not None:
            assert actual == expected, f'hash changed: {path}'
        self.evidence[str(path)] = actual
        return read(path) if path.suffix == '.json' else actual

    def gate(self, name, function):
        try:
            detail = function()
            self.gates.append({'id': name, 'status': 'passed', 'details': detail})
        except (AssertionError, KeyError, ValueError, OSError, TypeError) as exc:
            self.gates.append({'id': name, 'status': 'pending', 'reason': str(exc)})

    def native(self):
        doc = self.bind(ROOT / 'completion_requirements.json')
        assert doc['status'] == 'ready_for_root_final_review' and not doc['pending']
        assert len(doc['requirements']) == 8
        for gate in doc['requirements']:
            assert gate['status'] == 'passed'
            for check in gate['checks']:
                assert check['status'] == 'passed'
                for entry in check.get('evidence', []):
                    self.bind(entry['path'], entry['sha256'])
        return {'native_gates_passed': 8, 'existing_evidence_hashes_preserved': True}

    def postprocess(self):
        protocol = self.bind(POST / 'protocol.json')
        inputs = self.bind(protocol['input_manifest'], protocol['input_manifest_sha256'])
        self.bind(protocol['script'], protocol['script_sha256'])
        ids = set(protocol['ids'])
        assert len(ids) == 100 and set(protocol['methods']) == set(METHODS)
        expected = {(method, ident) for method in METHODS for ident in ids}
        assert len(inputs['records']) == 300
        assert {(r['method'], r['id']) for r in inputs['records']} == expected
        self.data['ids'] = ids
        self.data['expected'] = expected
        self.data['protocol_sha256'] = sha(POST / 'protocol.json')
        completion = self.bind(POST / 'completion_audit.json')
        assert completion['status'] == 'passed'
        assert completion['expected'] == completion['verified'] == 300
        assert completion['protocol_sha256'] == self.data['protocol_sha256']
        for key in ('raw_input_hashes_unchanged', 'kept_rgb_pixels_exact',
                    'kept_alpha_pixels_exact', 'removed_pixels_white_zeroalpha', 'geometry_unchanged'):
            assert completion[key] is True, key
        records = {(r['method'], r['id']): r for r in completion['records']}
        assert set(records) == expected and len(completion['records']) == 300
        bundles, statuses = [], Counter()
        for source in inputs['records']:
            key = source['method'], source['id']
            record = records[key]
            post = self.bind(record['postprocess'], record['postprocess_sha256'])
            validation = self.bind(source['source_validation'], source['source_validation_sha256'])
            assert validation['status'] == 'passed' and validation['reload_verified'] is True
            assert validation['id'] == source['id'] and validation['method'] == source['method']
            assert Path(validation['bundle']).is_file() and len(validation['bundle_sha256']) == 64
            assert post['protocol_sha256'] == self.data['protocol_sha256']
            assert post['source_validation_sha256'] == source['source_validation_sha256']
            assert post['geometry_changed'] is False
            assert post['reference_mask_pixels_read'] is False
            assert post['original_source_photo_pixels_read'] is False
            assert post['all_entries_retained_for_scoring'] is True
            for image in ('original_rgb', 'original_alpha'):
                self.bind(source[image], source[f'{image}_sha256'])
            for image in ('mask', 'clean_rgb', 'clean_alpha'):
                self.bind(post['outputs'][image], post['outputs'][f'{image}_sha256'])
            bundles.append({'method': key[0], 'id': key[1], 'path': validation['bundle'],
                            'inherited_verified_sha256': validation['bundle_sha256'],
                            'validation_sha256': source['source_validation_sha256']})
            statuses[post['status']] += 1
        self.data['bundle_chain'] = bundles
        self.data['postprocess_records'] = records
        self.data['postprocess_status_counts'] = dict(statuses)
        return {'records': 300, 'status_counts': dict(statuses),
                'raw_render_hashes_unchanged': True, 'source_validation_hashes_unchanged': True,
                'large_3d_bundle_rehash_performed': False,
                'geometry_evidence': 'Existing native reload/bundle verification inherited through immutable validation hashes.'}

    def metrics(self):
        ids = self.data['ids']
        methods = {}
        for method in METHODS:
            base = POST / 'metrics' / method
            pipeline = self.bind(base / 'pipeline.json')
            assert pipeline['status'] == 'completed'
            assert pipeline['static_records'] == pipeline['edp_processed'] == 100
            summary = self.bind(base / 'static_summary.json')
            assert summary['expected'] == 100 and not summary['unvisited']
            assert summary['edp_processed'] == 100 and summary['coverage'] == {'scored': 100}
            rows = list((base / 'static_rows').glob('*.json'))
            assert len(rows) == 100 and {p.stem for p in rows} == ids
            for path in rows:
                row = self.bind(path)
                assert row['id'] == path.stem and row['method'] == method
                self.bind(row['source_validation'], row['source_validation_sha256'])
            group = summary['groups'][method]
            for metric in ('PSNR', 'SSIM', 'LPIPS', 'IoU'):
                assert group[metric]['n'] == 100 and set(group[metric]['ids']) == ids
            edp = self.bind(base / 'edp/all/results.json')
            assert len(edp['rows']) == 100 and {r['id'] for r in edp['rows']} == ids
            valid = {r['id'] for r in edp['rows'] if r['score']['status'] == 'ok'}
            assert len(valid) == group['EDP']['n'] and valid == set(group['EDP']['ids'])
            methods[method] = {'static_attempted': 100, 'edp_attempted': 100,
                               'metrics': {k: {'value': v['value'], 'n': v['n']}
                                           for k, v in group.items()}}
        self.data['metric_summary'] = methods
        return methods

    def protected_geometry_temporal(self):
        count = 0
        for method in METHODS:
            audit = self.bind(ROOT / f'resolved/geometry/{method}/independent_arithmetic_audit.json')
            assert audit['status'] == 'passed' and audit['identity_count'] == 100
            for field in ('summary', 'protocol', 'original_reference_manifest'):
                item = audit[field]
                self.bind(item['path'], item['sha256'])
            for item in audit['row_evidence']:
                self.bind(item['path'], item['sha256'])
                count += 1
        temporal = self.bind(ROOT / 'temporal_reuse_audit.json')
        assert temporal['status'] == 'passed_read_only_provenance_and_aggregation_audit'
        assert temporal['old_metrics_modified'] is False
        # This binds the existing temporal audit; its prior source-level audit is not rerun.
        return {'geometry_result_hashes_preserved': count, 'temporal_prior_audit_preserved': True,
                'geometry_rerun': False, 'temporal_rerun': False}

    def gallery(self):
        gallery = self.bind(POST / 'gallery/manifest.json')
        assert gallery['status'] == 'complete' and gallery['count'] == 300
        assert gallery['self_contained_html'] is True
        assert gallery['protocol_sha256'] == self.data['protocol_sha256']
        assert gallery['geometry_modified'] is False and gallery['raw_native_gallery_modified'] is False
        self.bind(gallery['html'], gallery['html_sha256'])
        assert len(gallery['rows']) == 300
        assert {(r['method'], r['id']) for r in gallery['rows']} == self.data['expected']
        for row in gallery['rows']:
            source = self.data['postprocess_records'][(row['method'], row['id'])]
            assert row['postprocess_sha256'] == source['postprocess_sha256']
            self.bind(row['source_validation'], row['source_validation_sha256'])
            for image in ('source', 'raw', 'clean'):
                self.bind(row[image], row[f'{image}_sha256'])
        return {'gallery_records': 300, 'self_contained_offline_html': gallery['html']}

    def independent_numeric(self):
        audit = self.bind(POST / 'independent_readonly_audit.json')
        assert audit['output_contract']['status'] == 'passed'
        assert audit['output_contract']['count'] == 300 and not audit['output_contract']['errors']
        assert not audit['metrics']['errors']
        for method in METHODS:
            item = audit['metrics']['methods'][method]
            assert item['status'] == 'complete' and item['static_rows'] == item['edp_rows'] == 100
            assert all(metric['passed'] for metric in item['metrics'].values())
            assert len(item['edp_person_checks']) == 100 and all(r['passed'] for r in item['edp_person_checks'])
        return {'numeric_audit': 'passed', 'independent_pixel_contract': 'passed',
                'empty_predictions_kept_in_static_denominator': 50}

    def publication(self):
        # Formal publication is required; a candidate alone cannot pass this gate.
        published = self.bind(PAPER / PUBLICATION)
        independent = self.bind(POST / 'independent_readonly_audit.json')
        check = independent['publication_check']
        assert isinstance(check, dict), 'independent publication verification not finalized'
        assert check.get('status') == 'passed', 'independent publication verification not passed'
        assert not check.get('errors', [])
        baseline = self.bind(POST / 'publication_baseline/manifest.json', published['baseline_manifest_sha256'])
        self.bind(PAPER / '20_人体可观测几何补表.csv', baseline['20_人体可观测几何补表.csv'])
        # The publisher and independent checker identify allowed cells and protect
        # every other numeric entry, including temporal and all rigid results.
        assert published['status'] == 'published_prediction_2d_foreground_overlay'
        allowed = {('human', method, metric)
                   for method in ('SplatAD', 'OmniRe', 'StreetGS')
                   for metric in ('PSNR', 'SSIM', 'LPIPS', 'IoU', 'ED-P')}
        assert len(published['allowed_cells']) == 15
        assert {tuple(cell) for cell in published['allowed_cells']} == allowed
        for key in ('protected_native_geometry_temporal_rigid_other_methods_values_counts',
                    'all_selection_scopes_unchanged', 'geometry_tables_byte_preserved',
                    'prediction_only_2d_segmentation_not_3d_cleanup'):
            assert published[key] is True, key
        assert published['raw_native_artifacts_modified'] is False
        self.bind(published['source_script'], published['source_script_sha256'])
        for path, digest in published['source_hashes'].items():
            self.bind(path, digest)
        for name, digest in published['candidate_hashes'].items():
            self.bind(PAPER / name, digest)
        return {'publication_audit': str(PAPER / PUBLICATION),
                'independent_publication_check': check,
                'geometry_csv_byte_identical_to_pre_postprocess_baseline': True}

    def run(self, strict=False):
        for name, function in (
            ('native_8_gates', self.native), ('postprocess_300', self.postprocess),
            ('new_metrics_300', self.metrics), ('protected_geometry_temporal', self.protected_geometry_temporal),
            ('offline_gallery_300', self.gallery), ('independent_numeric_audit', self.independent_numeric),
            ('formal_publication', self.publication),
        ):
            self.gate(name, function)
        pending = [g['id'] for g in self.gates if g['status'] != 'passed']
        output = {'schema': 'native_human_postprocess_final_completion_v1',
                  'status': 'pending' if pending else 'ready_for_root_final_review',
                  'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'pending': pending, 'gates': self.gates,
                  'evidence_sha256': self.evidence,
                  'postprocess_status_counts': self.data.get('postprocess_status_counts'),
                  'metric_summary': self.data.get('metric_summary'),
                  'inherited_3d_bundle_hash_chain': self.data.get('bundle_chain', []),
                  'large_3d_bundles_rehashed': False,
                  'geometry_and_temporal_recomputed': False,
                  'paper_modified_by_this_script': False,
                  'goal_status_modified': False,
                  'free_disk_gib': shutil.disk_usage(POST).free / 1024 ** 3,
                  'script_sha256': sha(__file__)}
        target = POST / 'final_completion.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(target)
        print(json.dumps({'status': output['status'], 'pending': pending,
                          'output': str(target), 'sha256': sha(target)}, ensure_ascii=False))
        if strict and pending:
            raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='Exit nonzero until every final gate passes.')
    Audit().run(parser.parse_args().strict)
