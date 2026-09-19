"""Real fault-injection experiments: tensor.

    pip install 'experionyx[torch,faults]'
    python examples/fault_tensor.py [workspace]

Trains a tiny model in the workspace, runs a baseline evaluation, then real seeded fault
experiments against that baseline, and prints what was measured.
"""

import sys
from pathlib import Path

from experionyx.artifacts import LocalArtifactStore
from experionyx.faults.demos import run_fault_demo
from experionyx.faults.report import load_analysis, summary_rows
from experionyx.sqlite import SqliteRegistry

if __name__ == "__main__":
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".experionyx")
    results = run_fault_demo("tensor", workspace)
    registry = SqliteRegistry(workspace / "registry.sqlite")
    store = LocalArtifactStore(workspace / "experiments")
    for result in results:
        a = load_analysis(registry, store, result.fault_experiment.id)
        print(
            f"\n{result.fault_experiment.name}  (baseline {a.primary_metric} = {a.baseline_value})"
        )
        for row in summary_rows(a):
            change = f"deterioration={row['deterioration_mean']}"
            print(f"  {row['parameter']}={row['value']}  faulted={row['faulted_mean']}  {change}")
            print(f"      -> {row['classification']}")
    registry.close()
