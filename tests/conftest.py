"""Test-only startup configuration."""

import os


# Production requires the database-backed registry. Unit tests run against checked-in caches and
# should not require a production DATABASE_URL merely to import rag_server.
os.environ.setdefault("RAG_USE_LOCAL_MANUAL_REGISTRY", "1")
