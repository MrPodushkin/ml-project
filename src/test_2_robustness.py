"""Compatibility command; delegates to src.robustness, without a second procedure."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.robustness import run

if __name__ == "__main__":
    run()
