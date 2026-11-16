import hashlib

import pytest

pytest.importorskip('duckdb')

from pvdelta.dashboard_store import ensure_dashboard_schema
from pvdelta.store import SCHEMA_VERSION, connect_store


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_schema_is_read_only_across_reruns(tmp_path):
    path = tmp_path/'inspection.duckdb'
    assert ensure_dashboard_schema(path) is True
    before = digest(path)
    for _ in range(3):
        assert ensure_dashboard_schema(path) is False
        assert digest(path) == before


def test_old_schema_migrates_once_and_preserves_user_rows(tmp_path):
    path = tmp_path/'old.duckdb'
    connection = connect_store(path)
    connection.execute("UPDATE store_metadata SET value='1' WHERE key='schema_version'")
    connection.execute('CREATE TABLE user_notes (note VARCHAR)')
    connection.execute("INSERT INTO user_notes VALUES ('preserve')")
    connection.close()
    assert ensure_dashboard_schema(path) is True
    connection = connect_store(path, read_only=True)
    assert connection.execute('SELECT note FROM user_notes').fetchall() == [('preserve',)]
    assert connection.execute("SELECT value FROM store_metadata WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    connection.close()
    assert ensure_dashboard_schema(path) is False


def test_future_schema_is_not_downgraded(tmp_path):
    path = tmp_path/'future.duckdb'
    connection = connect_store(path)
    connection.execute("UPDATE store_metadata SET value='999' WHERE key='schema_version'")
    connection.close()
    before = digest(path)
    with pytest.raises(ValueError, match='newer'):
        ensure_dashboard_schema(path)
    assert digest(path) == before
