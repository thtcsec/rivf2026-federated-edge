"""Export a RFC 8259 JSON copy of results/verified/summary.json.

The frozen original uses Python's default json encoder, which writes bare
NaN for undefined total-variation diagnostics on continuous features. That
file is never overwritten. This script writes summary.strict.json with null
in those positions so strict parsers can load the same finite metrics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'results' / 'verified' / 'summary.json'
STRICT = ROOT / 'results' / 'verified' / 'summary.strict.json'
ORIGINAL_SHA256 = 'd7f95fa62bc5823575296d39d9ffcb36222189ddd0f28249aab847c2ab5cabec'


def sanitize(obj):
    """Replace non-finite floats with None; leave finite values unchanged."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def finite_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            finite_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(
            finite_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and not math.isfinite(a):
        return b is None
    if isinstance(b, float) and not math.isfinite(b):
        return False
    return a == b


def load_original(path: Path = ORIGINAL):
    # Python json accepts NaN; that is the frozen convention, not RFC 8259.
    return json.loads(path.read_text(encoding='utf8'))


def export(src: Path = ORIGINAL, dst: Path = STRICT) -> Path:
    if dst.resolve() == ORIGINAL.resolve():
        raise SystemExit('refusing to overwrite the frozen original summary.json')
    original = load_original(src)
    strict = sanitize(original)
    if not finite_equal(original, strict):
        raise SystemExit('strict export altered a finite metric')
    dst.write_text(json.dumps(strict, indent=2, allow_nan=False) + '\n',
                   encoding='utf8')
    return dst


def main() -> int:
    export()
    print(f'wrote {STRICT.relative_to(ROOT)} from {ORIGINAL.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
