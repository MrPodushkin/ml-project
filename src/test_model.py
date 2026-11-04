"""Compatibility command; delegates to src.demo_predict, without a second procedure."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo_predict import main

if __name__ == "__main__":
    main()
