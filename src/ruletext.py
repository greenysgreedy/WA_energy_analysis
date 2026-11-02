"""Extract searchable text from the reference documents so clauses can be quoted exactly.

PDF text extraction is slow, so each document is cached as a ``.txt`` file alongside the
PDF with a page marker before each page. Page markers let ``docs/assumptions.md`` cite a
page as well as a clause number.

Usage::

    .venv\\Scripts\\python.exe -m src.ruletext --extract
    .venv\\Scripts\\python.exe -m src.ruletext --find "Reference Trading Price" --doc wem_rules_2023-10-01
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.download_docs import DOCS_DIR

PAGE_MARKER = "\n\n<<<PAGE {page}>>>\n\n"
PAGE_RE = re.compile(r"<<<PAGE (\d+)>>>")


def text_path(pdf: Path) -> Path:
    return pdf.with_suffix(".txt")


def extract(pdf: Path, *, force: bool = False) -> Path:
    """Write a cached text rendering of ``pdf`` and return its path."""
    out = text_path(pdf)
    if out.exists() and not force:
        return out

    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(pdf) as doc:
        total = len(doc.pages)
        for number, page in enumerate(doc.pages, start=1):
            chunks.append(PAGE_MARKER.format(page=number))
            chunks.append(page.extract_text() or "")
            if number % 100 == 0 or number == total:
                print(f"    {pdf.name}: page {number}/{total}")

    out.write_text("".join(chunks), encoding="utf-8")
    return out


def extract_all(*, force: bool = False) -> list[Path]:
    paths = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        print(f"  {pdf.name}")
        paths.append(extract(pdf, force=force))
    return paths


def page_of(text: str, position: int) -> int | None:
    """Page number containing character offset ``position``."""
    last = None
    for match in PAGE_RE.finditer(text, 0, position):
        last = match
    return int(last.group(1)) if last else None


def find(doc_stem: str, pattern: str, *, context: int = 600, limit: int = 10) -> list[tuple[int, str]]:
    """Return ``(page, excerpt)`` for each match of ``pattern`` in the named document."""
    path = DOCS_DIR / f"{doc_stem}.txt"
    if not path.exists():
        pdf = DOCS_DIR / f"{doc_stem}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"no document {doc_stem} in {DOCS_DIR}")
        path = extract(pdf)

    text = path.read_text(encoding="utf-8")
    results = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - context // 3)
        end = min(len(text), match.end() + context)
        results.append((page_of(text, match.start()), text[start:end]))
        if len(results) >= limit:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="cache text for every PDF")
    parser.add_argument("--force", action="store_true", help="re-extract even if cached")
    parser.add_argument("--doc", help="document stem, e.g. wem_rules_2023-10-01")
    parser.add_argument("--find", help="regex to search for")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--context", type=int, default=900)
    args = parser.parse_args()

    if args.extract:
        extract_all(force=args.force)

    if args.find:
        if not args.doc:
            parser.error("--find needs --doc")
        for page, excerpt in find(args.doc, args.find, context=args.context, limit=args.limit):
            print(f"\n===== page {page} =====")
            print(excerpt.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
