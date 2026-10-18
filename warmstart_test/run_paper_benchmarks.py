"""Run the four paper comparisons serially with fixed CPU threading."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "svm_l1": "warmstart_test/svm_L1.jl",
    "svm_l2": "warmstart_test/svm_L2_robust.jl",
    "rebalancing": "warmstart_test/portfolioOpt/portfolioOpt.jl",
    "frontier": "warmstart_test/portfolioOpt/efficientFrontier.jl",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    directory = ROOT / "warmstart_test/results/run_logs"
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", JULIA_NUM_THREADS="1",
               DATADEPS_ALWAYS_ACCEPT="true", GKSwstype="100")
    julia = shutil.which("julia")
    if not julia:
        raise SystemExit("Julia was not found on PATH")
    status_file = directory / "run_status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    for case in args.cases:
        log = directory / (case + ".log")
        status[case] = {"state": "running", "log": str(log),
                        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        print("Starting", case, "log:", log, flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run([
                julia, "--project=warmstart_test", "--startup-file=no", "-e",
                "using LinearAlgebra; BLAS.set_num_threads(1); include(ARGS[1])",
                CASES[case],
            ], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status[case].update(state="complete" if result.returncode == 0 else "failed",
                            exit_code=result.returncode, elapsed_seconds=time.monotonic()-start)
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        print(case, status[case]["state"], "after",
              round(status[case]["elapsed_seconds"], 1), "seconds", flush=True)
        if result.returncode:
            raise SystemExit(f"{case} failed; inspect {log}")


if __name__ == "__main__":
    main()
