"""Compatibility imports for pre-split InkDepth checkpoints and scripts.

The implementation now lives in InkMARC.Depth. Installed users should install
inkmarc-depth; source checkouts can use the adjacent suite component directly.
"""

import sys
from pathlib import Path

_depth_root = Path(__file__).resolve().parents[3] / "InkMARC.Depth"
if _depth_root.is_dir() and str(_depth_root) not in sys.path:
    sys.path.insert(0, str(_depth_root))

from inkmarc_depth.layers import ConvBNAct
from inkmarc_depth.model import DepthProxyMobileNet, Fixed504x322Decoder, InkMARCDepth

__all__ = ["InkMARCDepth", "DepthProxyMobileNet", "Fixed504x322Decoder", "ConvBNAct"]
