"""Prepare old dashboard stores without rewriting current stores on every rerun."""
from pathlib import Path

from pvdelta.store import SCHEMA_VERSION, connect_store


def ensure_dashboard_schema(database):
    """Migrate missing/older schemas once; inspect current schemas read-only."""
    path = Path(database)
    if path.is_file():
        connection = connect_store(path, read_only=True)
        try:
            exists = connection.execute("SELECT count(*) FROM information_schema.tables "
                                        "WHERE table_schema='main' AND table_name='store_metadata'").fetchone()[0]
            version = connection.execute("SELECT value FROM store_metadata "
                                         "WHERE key='schema_version'").fetchone() if exists else None
        finally:
            connection.close()
        if version and version[0] == str(SCHEMA_VERSION):
            return False
        if version:
            try:
                newer = int(version[0]) > int(SCHEMA_VERSION)
            except (ValueError, TypeError) as exc:
                raise ValueError('Unsupported local database schema version') from exc
            if newer:
                raise ValueError('Database schema is newer than this dashboard; migration withheld')
    connection = connect_store(path)
    connection.close()
    return True
