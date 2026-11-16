import json

import pytest

from pvdelta.inspection_library import load_inspection_library


def test_explicit_library_resolves_relative_paths_without_opening_or_creating_databases(tmp_path):
    path = tmp_path/'inspection-library.json'
    entry = {'id':'native', 'label':'Radiometric inspection', 'database_path':'native/inspection.duckdb',
             'analysis_run_id':'native-run'}
    path.write_text(json.dumps({'schema':'pvdelta.inspection-library/v1','inspections':[entry]}))
    found = load_inspection_library(path)
    assert found == [entry | {'database_path':str(tmp_path/'native/inspection.duckdb')}]
    assert not (tmp_path/'native').exists()


def test_missing_optional_library_does_not_create_files(tmp_path):
    assert load_inspection_library(tmp_path/'inspection-library.json') == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure', ['not_object', 'wrong_schema', 'not_list', 'empty_run', 'duplicate', 'reserved'])
def test_invalid_library_cannot_silently_change_an_inspection(tmp_path, failure):
    entry = {'id':'native','label':'Radiometric inspection','database_path':'inspection.duckdb',
             'analysis_run_id':'native-run'}
    value = {'schema':'pvdelta.inspection-library/v1','inspections':[entry]}
    if failure == 'not_object':
        value = []
    elif failure == 'wrong_schema':
        value['schema'] = 'other'
    elif failure == 'not_list':
        value['inspections'] = {}
    elif failure == 'empty_run':
        entry['analysis_run_id'] = ''
    elif failure == 'duplicate':
        value['inspections'].append(dict(entry))
    else:
        entry['id'] = 'current'
    path = tmp_path/'inspection-library.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_inspection_library(path)
    assert not (tmp_path/'inspection.duckdb').exists()
