"""Alternate MIDI arrangements on a single backing's recording clock."""
import gzip
import json
from functools import lru_cache

from recording_parts import part_path

# Native slots and saved IDs stay stable when a recording has fewer arrangements.
VERSIONS = {"all": "All parts", "focus": "Focused part", "extra": "Additional instruments",
            "original": "Original score", "earlier": "Earlier extraction"}


@lru_cache(maxsize=32)
def bundled_versions(path, modified):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def arrangements(backing, score):
    part = part_path(backing)
    if part:
        package = part.parent / "versions" / (part.stem + ".json.gz")
        if package.is_file():
            return bundled_versions(package, package.stat().st_mtime_ns)
    if "earlier" in score:
        return {"earlier": score["earlier"]}
    return {}
