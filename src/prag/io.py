"""JSONL helpers for intermediate artifacts (pages.jsonl, chunks.jsonl)."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson
from pydantic import BaseModel


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record.model_dump(mode="json")) + b"\n")
            n += 1
    return n


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> Iterator[T]:
    with path.open("rb") as f:
        for line in f:
            yield model.model_validate(orjson.loads(line))
