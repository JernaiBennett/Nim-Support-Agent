"""
Splits markdown documentation files into retrievable chunks.

Chunking strategy: split on markdown headings (## and ###) so each chunk is a
self-contained topic (e.g. "Profile Selection", "Reasoning Mode") rather than
an arbitrary fixed-length window. This keeps retrieved context coherent, which
matters more than raw chunk-size optimization for a small, curated corpus like
this one.
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def split_into_chunks(filepath: Path) -> list[Chunk]:
    content = filepath.read_text(encoding="utf-8")
    source_name = filepath.stem

    # Split on any heading line (#, ##, ###). Keep the heading with its body.
    pattern = re.compile(r"^(#{1,3}\s+.*)$", re.MULTILINE)
    parts = pattern.split(content)

    chunks = []
    # parts alternates: [preamble, heading1, body1, heading2, body2, ...]
    current_heading = "Introduction"
    idx = 0

    if parts[0].strip():
        chunks.append(Chunk(
            id=f"{source_name}_{idx}",
            source=source_name,
            heading=current_heading,
            text=parts[0].strip(),
        ))
        idx += 1

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip().lstrip("#").strip()
        body = parts[i + 1].strip()
        if body:
            chunks.append(Chunk(
                id=f"{source_name}_{idx}",
                source=source_name,
                heading=heading,
                text=f"{heading}\n\n{body}",
            ))
            idx += 1
        i += 2

    return chunks


def build_corpus(docs_dir: Path) -> list[Chunk]:
    all_chunks = []
    for filepath in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(split_into_chunks(filepath))
    return all_chunks


if __name__ == "__main__":
    docs_dir = Path(__file__).parent.parent / "docs"
    chunks = build_corpus(docs_dir)
    print(f"Built {len(chunks)} chunks from {len(list(docs_dir.glob('*.md')))} doc files\n")
    for c in chunks:
        print(f"[{c.id}] {c.heading}  ({len(c.text)} chars)")

    out_path = Path(__file__).parent.parent / "corpus.json"
    out_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2))
    print(f"\nSaved corpus to {out_path}")
