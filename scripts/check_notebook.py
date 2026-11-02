"""Report error outputs and selected text outputs from executed notebooks.

Used to confirm a notebook ran top to bottom, and to read its printed results back
without opening it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def outputs_of(cell: dict) -> list[str]:
    chunks = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            chunks.append("".join(output.get("text", [])))
        elif kind in {"execute_result", "display_data"}:
            text = output.get("data", {}).get("text/plain")
            if text:
                chunks.append("".join(text))
        elif kind == "error":
            chunks.append(f"ERROR {output.get('ename')}: {output.get('evalue')}")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--show", action="store_true", help="print every text output")
    parser.add_argument("--cells", type=str, help="comma-separated cell indices to print")
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    errors = [
        (index, output.get("ename"), output.get("evalue"))
        for index, cell in enumerate(cells)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]

    wanted = (
        {int(c) for c in args.cells.split(",")}
        if args.cells
        else set(range(len(cells)))
        if args.show
        else set()
    )

    for index in sorted(wanted):
        chunks = outputs_of(cells[index])
        if chunks:
            print(f"\n----- cell {index} -----")
            print("\n".join(chunks).rstrip())

    print(f"\n{args.notebook.name}: {len(cells)} cells, {len(errors)} error(s)")
    for index, name, value in errors:
        print(f"  cell {index}: {name}: {value}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
