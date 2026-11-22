# Source validation — 2026-11-22

Validation was performed on the existing repository combined with this prepared update. No model generation, training, quantitative re-evaluation, or asset cleanup was launched.

- `tools/check_release.py`: passed for 507 text files; Python/JSON syntax, relative documentation links, file-size/data-file constraints, personal server paths and credential patterns checked.
- Portable core `unittest discover -s tests -v`: 18 passed, 1 optional LPIPS-weight test skipped (weights were not configured for this release check).
- Existing experiment tests: 22 passed across `SourceMetricTest`, six-part ED-P, and source-time optical-flow/perceptual-change contracts.
- All 393 collected source/template files match their published SHA256 values in `metadata/source_snapshot_20261122.json`.
- Additional credential-pattern review found no GitHub/Hugging Face/AWS key, authenticated HTTP URL, or private-key-block candidates.

The existing evaluation environment contained the numerical dependencies but not pytest. The same existing unittest tests were run directly:

```bash
export PYTHONPATH="$PWD/src:$PWD/workspace/human_model_comparison/scripts"
python -m unittest discover -s tests -v
python -m unittest \
  test_human_source_metrics_v1.SourceMetricTest \
  test_human_edp_common \
  test_dynamic_source_metrics -v
python tools/check_release.py
```

This verifies metric contracts and source packaging. It does not certify that every external model dependency, dataset, historical pinned manifest or GPU workflow is reproducible on a fresh machine.
