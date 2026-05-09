import os
import sys

_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)
