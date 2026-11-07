"""
One-command local runner for the final thesis upgrade.

Examples
--------
python 02_code/pipeline/run_all.py --root . --n-personas 500 --sessions 18
python 02_code/pipeline/run_all.py --root . --skip-sensitivity --no-thesis
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, check=check)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--n-personas", type=int, default=500)
    parser.add_argument("--sessions", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--no-thesis", action="store_true", help="Skip LaTeX compilation.")
    parser.add_argument("--no-r", action="store_true", help="Skip optional R scripts.")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    pipeline = root / "02_code" / "pipeline" / "loklok_pipeline.py"
    figures = root / "02_code" / "pipeline" / "generate_figures.py"
    cmd = [
        sys.executable,
        pipeline,
        "--root",
        root,
        "--n-personas",
        args.n_personas,
        "--sessions",
        args.sessions,
        "--seed",
        args.seed,
    ]
    if args.skip_sensitivity:
        cmd.append("--skip-sensitivity")
    run(cmd)
    run([sys.executable, figures, "--root", root])

    if not args.no_r:
        rscript = shutil.which("Rscript")
        if rscript:
            for script in ["01_measured_wave.R", "02_cfa_invariance.R", "03_bayes_crosscheck.R"]:
                run([rscript, root / "02_code" / "R" / script], cwd=root, check=False)
            run([sys.executable, root / "02_code" / "pipeline" / "make_r_macros.py"], check=False)
        else:
            print("Rscript not found; skipping R reproduction scripts.")

    if not args.no_thesis:
        src = root / "01_thesis" / "src"
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            for _ in range(3):
                run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "thesis_final.tex"], cwd=src, check=True)
            build = root / "01_thesis" / "build"
            build.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "thesis_final.pdf", build / "thesis_final.pdf")
            print("Thesis PDF:", build / "thesis_final.pdf")
        else:
            print("pdflatex not found; skipping thesis compilation.")

    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
