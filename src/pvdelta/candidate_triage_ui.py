"""Cached read-only support evidence shared by review and overview."""
from pathlib import Path

import streamlit as st

from pvdelta.candidate_triage import candidate_support_audit
from pvdelta.store import connect_store


@st.cache_data(show_spinner=False, max_entries=8)
def _read(database, run_id, signature):
    connection = connect_store(database, read_only=True)
    try:
        return candidate_support_audit(connection, database, run_id)
    finally:
        connection.close()


def support_audit(database, run_id):
    path = Path(database).resolve()
    signature = []
    for item in (path, Path(str(path)+'.wal'), path.with_name('summary.json'),
                 path.with_name('surface-references.json')):
        stat = item.stat() if item.exists() else None
        signature.append((stat.st_mtime_ns, stat.st_size) if stat else None)
    return _read(str(path), run_id, tuple(signature))
