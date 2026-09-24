"""Runs the canonical end-to-end example exactly as a researcher would (`python examples/
end_to_end_workflow.py <workspace>`) so it cannot silently bitrot into a decorative tutorial."""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parent.parent / "examples" / "end_to_end_workflow.py"


def test_end_to_end_workflow_runs_and_produces_real_artifacts(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), str(ws)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    for marker in (
        "reliability_profile=",
        "graph_snapshot=",
        "reproduction_attempt=",
        "report=",
        "dossier=",
        "dossier_snapshot=",
    ):
        assert marker in result.stdout, f"missing {marker!r} in output:\n{result.stdout}"

    report_md = ws / "report.md"
    dossier_md = ws / "dossier.md"
    assert report_md.is_file()
    assert report_md.stat().st_size > 0
    assert dossier_md.is_file()
    assert dossier_md.stat().st_size > 0
    assert (ws / "registry.sqlite").is_file()
