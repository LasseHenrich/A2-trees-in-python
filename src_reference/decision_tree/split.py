# Reference implementation. Do not edit.
from importlib.machinery import SourcelessFileLoader as _L
from importlib.util import module_from_spec as _ms, spec_from_file_location as _sf
from pathlib import Path as _P
from sys import modules as _m, version_info as _v
_f = str(
    _P(__file__).parent
    / f"{_P(__file__).stem}.cpython-{_v.major}{_v.minor}.pyc"
)
_s = _sf(__name__, _f, loader=_L(__name__, _f))
_x = _ms(_s)
_s.loader.exec_module(_x)
_m[__name__] = _x
del _L, _ms, _sf, _P, _m, _v, _f, _s, _x
