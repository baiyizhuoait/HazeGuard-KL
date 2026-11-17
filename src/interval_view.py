"""Interval artefacts are nested inconsistently; pick the layer that carries the numbers.

The two files member 5 ships do not share a shape:

* ``metrics_stage5_interval_adaptive.json`` → ``test.overall.PICP``
* ``metrics_stage5_interval_calibrated.json`` →
  ``test.calibrated.PICP`` and ``test.raw_tree_spread.PICP``, and its
  ``by_node`` nests one level deeper still.

Reading ``record["test"]["by_node"][node]["PICP"]`` therefore works on the
adaptive file and raises ``KeyError: 'PICP'`` on the calibrated one — which is
exactly what the fixed-width fallback did whenever the adaptive file was absent.
The dashboard should not care which file it got, so the unwrapping lives here and
is covered by tests.
"""

from __future__ import annotations

from typing import Any, Iterable

# Order matters only when a file carries more than one block; the first hit wins.
LAYER_NAMES = ("overall", "calibrated", "raw_tree_spread")


def pick_block(container: Any, *names: str) -> dict | None:
    """Return the first nested dict that actually contains ``PICP``."""
    if not isinstance(container, dict):
        return None
    for name in names or LAYER_NAMES:
        block = container.get(name)
        if isinstance(block, dict) and "PICP" in block:
            return block
    return None


def node_blocks(test_section: Any) -> dict[str, dict]:
    """Per-node summary, flattened across both file shapes."""
    if not isinstance(test_section, dict):
        return {}
    raw = test_section.get("by_node") or {}
    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    for name, block in raw.items():
        # Adaptive file stores the numbers directly under the node; the calibrated
        # file nests them one level down. Accept both.
        if isinstance(block, dict) and "PICP" in block:
            out[name] = block
            continue
        picked = pick_block(block)
        if picked:
            out[name] = picked
    return out


def summary(record: Any) -> dict | None:
    """Nominal coverage plus the validation/test/ per-node blocks.

    Returns ``None`` when the record does not carry a usable test summary, so the
    caller can fall back to a plain caption instead of raising.
    """
    if not isinstance(record, dict):
        return None
    test = pick_block(record.get("test"))
    val = pick_block(record.get("val"))
    if not test or not val:
        return None
    return {
        "nominal": record.get("nominal_coverage", 0.9),
        "test": test,
        "val": val,
        "nodes": node_blocks(record.get("test")),
    }
