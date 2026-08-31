"""
api/index.py
------------
Vercel serverless entrypoint.

Vercel's Python runtime loads this file and serves the exported ``app`` (a
Flask instance). The application itself lives in ``app.py`` at the repository
root and is shared verbatim with the Render deployment, so both platforms
always run the exact same code — this file only bridges Vercel's ``api/``
layout to the root-level module imports (``deep_dive``, ``student_branch``,
``context_builder``, ``llm_utils``).
"""

import os
import sys

# The function bundle places this file at <root>/api/index.py; put <root>
# first on sys.path so `import app` resolves to the repository root rather
# than anything inside this directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402
