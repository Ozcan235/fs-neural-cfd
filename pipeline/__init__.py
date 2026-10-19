"""OpenCFD veri üretim pipeline'ı: mesh -> case kurulumu -> çözüm -> katsayılar -> sweep."""

import sys as _sys

# Windows konsolunda Türkçe karakterler için
for _s in (_sys.stdout, _sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
