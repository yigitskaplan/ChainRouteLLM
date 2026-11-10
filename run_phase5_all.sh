#!/bin/bash
# Phase 5: every experiment in order (CPU only; the GPU stays free). Logs in results/phase5/logs/.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
mkdir -p results/phase5/logs
nice -n 10 python3 -m src.evaluation.run_phase5 offline  2>&1 | tee results/phase5/logs/e1.log
nice -n 10 python3 -m src.evaluation.run_phase5 online   2>&1 | tee results/phase5/logs/e2.log
nice -n 10 python3 -m src.evaluation.run_phase5 attacks --w 0.1 0.3 2>&1 | tee results/phase5/logs/e345.log
nice -n 10 python3 -m src.evaluation.figures_phase5 2>&1 | tee results/phase5/logs/fig.log
