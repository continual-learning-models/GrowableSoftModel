"""Launch the SoftModel dual API server.

Run:  python scripts/run_api.py
Env:  SOFTMODEL_BACKEND=mock|hf   SOFTMODEL_BASE_MODEL=...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from generator.api import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
