import sys
from pathlib import Path

# Let the tests import `src` however pytest is started.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
