"""One results pointer for tables, galleries and advisor snapshots."""
import json
from pathlib import Path

H = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'core_summary': 'reports/source_camera_metric_cohort_v3/summary.json',
    'edp_root': 'reports/source_edp_common_v1/cohort_v1',
    'gallery': 'reports/human_comparison_full_v1',
    'advisor': 'reports/advisor_report_20261112_v1',
    'input_budget': 'reports/input_budget_audit_v1',
}


def resolve(key):
    pointer = H/'inputs/active_results.json'
    values = json.loads(pointer.read_text()) if pointer.exists() else DEFAULTS
    path = (H/(values[key] if key in values else DEFAULTS[key])).resolve()
    assert path.is_relative_to(H.resolve()), 'Result path must remain in this experiment'
    return path
