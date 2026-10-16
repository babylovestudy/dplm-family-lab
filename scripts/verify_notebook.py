"""Execute the teaching notebook without changing its saved cells or outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dplm_family_lab.common import environment_info, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="notebooks/01_fasta_and_tensors.ipynb")
    parser.add_argument("--output", default="artifacts/notebook_validation.json")
    args = parser.parse_args()
    notebook = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__"}
    started = time.perf_counter()
    executed = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{args.notebook}:cell_{index + 1}", "exec"), namespace)
            executed.append(index + 1)
    report = {
        "status": "passed", "notebook": args.notebook, "code_cells": executed,
        "elapsed_seconds": time.perf_counter() - started, "environment": environment_info(),
        "scope": "teaching notebook execution; no foundation model training",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "executed_code_cells": len(executed)}))


if __name__ == "__main__":
    main()
