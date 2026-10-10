#!/usr/bin/env python3
"""Public ToGo entry point, runnable from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ToGo" / "src"))
from ToGo.pipeline.handoff import main

if __name__ == "__main__":
    raise SystemExit(main())
