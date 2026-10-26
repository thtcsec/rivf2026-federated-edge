"""Fail-closed output paths for Phase B.

Phase B may write only under results/phase_b/. Any path that resolves inside
results/verified/ is refused before a file is opened. Legacy
run_verified_federated_baseline.main() still writes the accepted reference
files; Phase B must never call it.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = ROOT / 'results' / 'verified'
PHASE_B_DIR = ROOT / 'results' / 'phase_b'
MECHANISM_DIR = PHASE_B_DIR / 'mechanism'
ROBUSTNESS_DIR = PHASE_B_DIR / 'robustness'
MISSINGNESS_DIR = PHASE_B_DIR / 'missingness'


class UnsafeOutputError(RuntimeError):
    """Raised when a write would escape the Phase B directory or hit verified/."""


def _resolve(path: Path) -> Path:
    return Path(path).resolve()


def is_inside(path: Path, root: Path) -> bool:
    path, root = _resolve(path), _resolve(root)
    return path == root or root in path.parents


def assert_phase_b_output(path: Path | str, *, allowed_root: Path | None = None,
                          forbidden_root: Path | None = None) -> Path:
    """Return a resolved path, or raise if it is not an isolated Phase B write."""
    allowed = _resolve(allowed_root if allowed_root is not None else PHASE_B_DIR)
    forbidden = _resolve(forbidden_root if forbidden_root is not None else VERIFIED_DIR)
    resolved = _resolve(Path(path))
    if is_inside(resolved, forbidden):
        raise UnsafeOutputError(
            f'refused to write inside reference directory {forbidden}: {resolved}')
    if resolved != allowed and not is_inside(resolved, allowed):
        raise UnsafeOutputError(
            f'refused to write outside Phase B directory {allowed}: {resolved}')
    return resolved


def assert_mechanism_output(path: Path | str) -> Path:
    """Phase B.2 writes only under results/phase_b/mechanism/."""
    return assert_phase_b_output(
        path, allowed_root=MECHANISM_DIR, forbidden_root=VERIFIED_DIR)


def assert_robustness_output(path: Path | str) -> Path:
    """Phase B.3 writes only under results/phase_b/robustness/."""
    return assert_phase_b_output(
        path, allowed_root=ROBUSTNESS_DIR, forbidden_root=VERIFIED_DIR)


def assert_missingness_output(path: Path | str) -> Path:
    """Phase B.4 writes only under results/phase_b/missingness/."""
    return assert_phase_b_output(
        path, allowed_root=MISSINGNESS_DIR, forbidden_root=VERIFIED_DIR)


def prepare_phase_b_dir(root: Path | None = None) -> Path:
    allowed = _resolve(root if root is not None else PHASE_B_DIR)
    assert_phase_b_output(allowed, allowed_root=allowed)
    allowed.mkdir(parents=True, exist_ok=True)
    return allowed


def write_bytes(path: Path | str, data: bytes, *, allowed_root: Path | None = None) -> Path:
    dest = assert_phase_b_output(path, allowed_root=allowed_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def write_text(path: Path | str, text: str, *, encoding: str = 'utf8',
               allowed_root: Path | None = None) -> Path:
    return write_bytes(path, text.encode(encoding), allowed_root=allowed_root)
