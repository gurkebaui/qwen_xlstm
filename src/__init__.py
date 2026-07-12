# src package — re-exports the public building blocks.
# We add THIS folder to sys.path so the sibling modules (xlstm_layer.py,
# patcher.py) are importable by name regardless of how the test is launched.
import os, sys
_sys_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

from xlstm_layer import XLSTMLayer, XLSTMLayerConfig
from patcher import XlstmQwenModel, XlstmQwenLayer
