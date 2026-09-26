"""Vercel entrypoint for the FX bot dashboard.

Vercel's Python runtime finds a function by looking for a **top-level class
named ``handler``** in each ``api/*.py`` file (an AST-level check, so an
assignment such as ``handler = X`` is *not* detected and the build fails with
"doesn't match any Serverless Functions inside the api directory").

So this file exists purely to satisfy that contract and hand off to the real
dashboard handler in :mod:`bot.web.server`.  All the routing, the JSON API and
the static-asset serving live there, which means the local
``python -m bot web`` server and the deployed serverless function run the
exact same code.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Vercel sets the working directory to the project base, but be explicit so the
# import works no matter how the function is invoked.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.web.server import DashboardHandler  # noqa: E402  (needs the path above)


class handler(DashboardHandler):  # noqa: N801  (Vercel requires this exact name)
    """The dashboard, as a Vercel serverless function.

    Subclassing rather than aliasing is deliberate: Vercel's analyser only
    recognises a class *definition* named ``handler``.
    """
