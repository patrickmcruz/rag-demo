"""Test configuration for ensuring src is importable."""

import sys
from pathlib import Path

# Add project root to sys.path for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
