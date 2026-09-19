"""Real baseline evaluation (adapter-backed): torch-classification.

    pip install 'experionyx[torch]'
    python examples/adapter_torch_classification.py [workspace]

Trains a tiny model inside the workspace, registers the model and dataset, executes the
experiment through the execution engine, and prints what was actually measured and recorded.
"""

import sys
from pathlib import Path

from experionyx.demos import run_demo
from experionyx.execution import ExecutionResult


def report(result: ExecutionResult) -> None:
    print(f"run {result.run.id}: {result.status}")
    shown = ("metric.", "calibration.", "latency.median", "imbalance.")
    for obs in result.observations:
        if obs.name.startswith(shown) and not obs.name.endswith(".interval"):
            print(f"  {obs.name} = {obs.value}" + (f" {obs.unit}" if obs.unit else ""))
    print(f"  {len(result.observations)} observations, {len(result.artifacts)} artifacts recorded")
    for art in result.artifacts:
        print(f"  artifact {art.path} {art.digest}")
    inputs = result.provenance.inputs
    if inputs and inputs.model and inputs.dataset:
        m, d = inputs.model, inputs.dataset
        print(f"  model   {m.adapter} {m.adapter_version} {m.fingerprint}")
        print(f"  dataset {d.adapter} {d.adapter_version} {d.fingerprint}")
        print(f"  device  {inputs.device}")
    print(f"  provenance fingerprint {result.provenance.fingerprint}")


if __name__ == "__main__":
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".experionyx")
    report(run_demo("torch-classification", workspace))
