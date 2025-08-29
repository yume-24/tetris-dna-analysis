# analysis/main.py
import argparse
import sys, pathlib as P

# allow running as a script (no package context)
if __package__ is None:
    sys.path.insert(0, str(P.Path(__file__).resolve().parents[1]))
    from pipelines.run_all import main as _run_all_main
else:
    from .pipelines.run_all import main as _run_all_main

def _default_config_path() -> P.Path:
    repo = P.Path(__file__).resolve().parents[1]  # repo/analysis -> repo
    # preferred default: .venv/config/default.yaml
    p1 = repo / ".venv" / "config" / "default.yaml"
    if p1.exists():
        return p1
    # fallback: repo/default.yaml
    p2 = repo / "default.yaml"
    return p2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Path to YAML config (default tries .venv/config/default.yaml, then default.yaml in repo root)",
    )
    args = ap.parse_args()

    # Delegate to the pipeline entrypoint, forwarding the --config arg
    sys.argv = ["run_all.py", "--config", args.config]
    _run_all_main()

if __name__ == "__main__":
    main()
