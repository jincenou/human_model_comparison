# Contributing

Start with `python -m pytest -q` and `python tools/check_release.py`. Keep changes to camera geometry, alpha conventions and metric aggregation covered by focused numerical examples. Tests using real assets must document the separate data/weight prerequisites.

When filing a reproducibility issue, include the repository commit, environment versions, failing command, expected/observed behavior and non-sensitive provenance hashes. Do not post credentials or restricted images/weights.

Do not silently modify a frozen cohort, substitute a different exposure, re-pose failed predictions, lower a training budget, or replace a technical failure with a fabricated score. New input modalities and generated auxiliary images require explicit accounting.

Use separate output folders for new protocol versions. Never edit an active controller's pinned worker code or start duplicate GPU owners. Changes to final benchmark eligibility must be based on the reference/input protocol, not on which method scores better.
