# Phase 5 analysis

Run the complete workflow from the project root:

```bash
.venv/bin/python bin/phase_5/run_phase_5.py
```

All analysis artifacts are written to `bin/data/phase_5`. Each numbered script also supports `--data-root`, `--manifest`, `--output-dir`, and `--seed`. Output directories must remain beneath the selected data root.

Run focused tests with:

```bash
.venv/bin/python -m unittest bin/phase_5/test_phase_5.py
```

The scalability analysis uses the dated `pricing_2026-06-23.json` snapshot. Confirm its official source before using the estimate for a future production run.
