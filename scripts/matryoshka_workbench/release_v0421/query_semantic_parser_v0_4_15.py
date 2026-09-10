#!/usr/bin/env python3
"""Coordinated CLI for the Matryoshka Query Parser v0.4.15."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from matryoshka_v0420.parser import main


if __name__ == "__main__":
    raise SystemExit(main())
