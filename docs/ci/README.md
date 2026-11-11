# Optional GitHub Actions setup

The `github-actions.yml.example` file is an inactive workflow template. It runs the portable metric tests, queue takeover/resource tests, and release validation on Python 3.10.

To enable hosted CI, copy it to `.github/workflows/tests.yml` using credentials with GitHub workflow write permission. The publication credential had repository-content access but lacked OAuth `workflow` scope, so this release does not enable GitHub Actions or claim a hosted CI pass.

Local validation before publication passed 19 core metric/contract tests (including LPIPS with local pinned weights), eight queue tests, editable installation, both command-line entry points, and release-tree checks. On a checkout without LPIPS weights, the optional LPIPS numerical comparison is skipped explicitly.
