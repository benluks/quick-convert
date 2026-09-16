import sys

from .. import matcha as _matcha


sys.modules.setdefault("cosyvoice", sys.modules[__name__])
sys.modules.setdefault("matcha", _matcha)
