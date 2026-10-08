"""RAG evaluation framework for a support assistant over the Ledgerly help center."""

import os
import sys

__version__ = "0.1.0"

if sys.platform == "darwin":
    # On macOS, FAISS and PyTorch (used by the BGE embedding model) each ship their own
    # OpenMP runtime, and loading both aborts the process. This is the standard workaround.
    # Limiting OpenMP to one thread avoids the conflicts the duplicate runtime can cause.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
