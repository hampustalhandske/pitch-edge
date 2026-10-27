# r2q3 strategies as submitted to edgelab

The two R2-Q3 strategy specs pitch-edge packaged for edgelab's grading engine (`pitch-edge-r2q3-bet365`, run
once on the fast profile and graded inconclusive; `pitch-edge-r2q3-max`, dry-run only, never counted). Built by
`../make_strategy.py` from the bundles that `../build_dataset.py` writes under `data/research/r2/edgelab/`.

Each `strategy.yaml` differs from the file edgelab ran in two ways: `data.params.path` is written relative to the
repo root (edgelab was given the absolute path), and the `grading:` policy block that edgelab's `strategy init`
appends is left out (it belongs to edgelab, which is private). The sha256 of each file exactly as edgelab hashed it:

- `pitch-edge-r2q3-bet365/strategy.yaml`: `5e8b43fd358aa69b49992c03dc8e6c62a772120d32d72328010d5f488b49edd6`
- `pitch-edge-r2q3-max/strategy.yaml`: `efc990f191563d0016ef922bfee366ec821a1098e97b32b7e6402b3cfcb6b6c6`

The complete folders, including the bet365 run output (summary, stage JSONs, report PDF), are kept locally under
`data/research/r2/edgelab/strategies/` (git-ignored: derived from lake data).
