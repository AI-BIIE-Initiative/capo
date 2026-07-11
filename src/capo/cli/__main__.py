"""Allow python -m capo.cli as an alias for the capo entry point."""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    main()
