"""Identity-based candidate browsing and explicit panel/table association."""
from __future__ import annotations

import csv
import io
import json

import duckdb
import streamlit as st

from pvdelta.identity_review import save_decisions, validate_decisions, workbench_package
from pvdelta.identity_store import (
    assign_identity,
    assign_table_batch,
    clear_identity,
    identity_groups,
    identity_snapshot,
)


def identity_browser(database, groups, *, key):
    snapshot=identity_snapshot(database,groups)
    with st.expander('Anomalies by panel / table ID',expanded=bool(snapshot['assignments'])):
        mode=st.radio('Group by',('Panel when known','Table'),horizontal=True,key=key+'-identity-mode')
        grouped=identity_groups(groups,snapshot,prefer_table=mode=='Table')
        counts=st.columns(3)
        counts[0].metric('Panel ID groups',sum(g['level']=='panel' for g in grouped))
        counts[1].metric('Table ID groups',sum(g['level']=='table' for g in grouped))
        counts[2].metric('Observations awaiting ID',sum(g['level']=='unassigned' for g in grouped))
        st.caption('Provisional geometry links require review. Region and photo counts can include repeated views of one defect.')
        st.caption(f"Registry: {sum(e['kind']=='table' for e in snapshot['entities'])} tables · "
                   f"{sum(e['kind']=='panel' for e in snapshot['entities'])} panel positions.")
        if groups:
            identity_workbench(database, groups[0]['analysis_run_id'], key=key)
            from pvdelta.panel_pair_review_ui import panel_pair_review

            panel_pair_review(database, groups, key=key)
        linked=[g for g in grouped if g['level']!='unassigned']
        if snapshot['stale']:
            st.warning(f"{len(snapshot['stale'])} identity links need review after source geometry changed.")
        with st.expander('Assign several source cards to one table'):
            candidates={g['thermal_path']+' · '+g['module_id']:g for g in groups}
            with st.form(key+'-bulk-table-form'):
                selected_cards=st.multiselect('Source cards confirmed on the same table',list(candidates),
                                              key=key+'-bulk-cards')
                table_label=st.text_input('Shared table ID',key=key+'-bulk-label')
                submitted=st.form_submit_button('Assign selected cards to table')
            if submitted:
                try:
                    assign_table_batch(database,[candidates[s] for s in selected_cards],table_label)
                    st.rerun()
                except (OSError,ValueError,duckdb.Error) as exc:
                    st.error(f'Table assignment could not be saved: {exc}')
        if linked:
            choices={g['label']+' · '+g['level']:g for g in linked}
            selected=st.selectbox('Panel or table',list(choices),key=key+'-identity-group')
            group=choices[selected]
            rows=[]
            for observation in group['observations']:
                for region in observation['regions']:
                    rows.append({'ID':group['label'],'Level':group['level'],'Frame':observation['thermal_path'],
                        'Finding':region['finding_id'],'Delta T (K)':region['delta_t_k'],
                        'Review':region['review_status'],'Basis':snapshot['assignments'][observation['observation_id']]['basis']})
            st.dataframe(rows,hide_index=True)
            source_choices={o['thermal_path']+' · '+o['module_id']:o['observation_id'] for o in group['observations']}
            source_label=st.selectbox('Source observation',list(source_choices),
                                     key=key+'-identity-source-'+group['uid'])
            observation=source_choices[source_label]
            if st.button('Open source card',key=key+'-identity-open'):
                st.session_state[key+'-open-related']=observation
                st.rerun()
        export=[]
        for group in grouped:
            for observation in group['observations']:
                for region in observation['regions']:
                    export.append({'identity':group['label'],'identity_uid':group['uid'] if group['level']!='unassigned' else '',
                        'level':group['level'],'observation':observation['observation_id'],'finding_id':region['finding_id'],
                        'frame':observation['thermal_path']})
        if export:
            text=io.StringIO()
            writer=csv.DictWriter(text,fieldnames=list(export[0]))
            writer.writeheader()
            writer.writerows({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v)
                              for k,v in row.items()} for row in export)
            st.download_button('Download ID-based candidate list',text.getvalue(),file_name='candidate-identities.csv',
                               mime='text/csv',key=key+'-identity-export')


def identity_workbench(database, run, *, key):
    try:
        package = workbench_package(database, run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f'Identity verification package unavailable: {exc}')
        return
    if package is None:
        return
    pack, archive = package
    with st.expander('Verify table identities offline'):
        st.caption(f"{len(pack['records'])} source cards. Extract the entire ZIP, open "
                   'review.html inside the workbench folder, then export your decision JSON.')
        st.caption('Confirm table means a human-reviewed association, not a confirmed thermal fault. '
                   'Uploaded decisions are archived separately; identity links remain unchanged.')
        if archive.is_file():
            size = archive.stat().st_size / 1024**2
            if st.checkbox(f'Load offline package ({size:.0f} MB)', key=key+'-workbench-load'):
                with archive.open('rb') as source:
                    st.download_button('Download identity verification ZIP', source,
                        file_name=archive.name, mime='application/zip', key=key+'-workbench-download')
        else:
            st.warning('Prepared ZIP is missing.')
        upload = st.file_uploader('Upload exported identity decisions', type=['json'],
                                  key=key+'-workbench-upload')
        if upload is not None:
            try:
                if upload.size > 10 * 1024**2:
                    raise ValueError('Decision JSON must be smaller than 10 MB')
                document = json.loads(upload.getvalue())
                report = validate_decisions(database, pack, document)
                st.json(report)
                if st.button('Save validated operator review', key=key+'-workbench-save'):
                    path, _ = save_decisions(database, pack, document)
                    st.success(f'Operator review saved: {path.name}')
                    st.caption(str(path))
            except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as exc:
                st.error(f'Review rejected: {exc}')


def identity_assignment(database, group, *, key):
    snapshot=identity_snapshot(database,[group])
    entities={e['uid']:e for e in snapshot['entities']}
    link=snapshot['assignments'].get(group['observation_id'])
    table=entities[link['table_uid']]['label'] if link else ''
    panel=entities[link['panel_uid']]['label'] if link and link['panel_uid'] else ''
    with st.expander('Panel / table identity',expanded=bool(link)):
        if link:
            st.caption(f"Table {table} · Panel {panel or 'awaiting identification'} · {link['basis']}")
        st.caption('Use the same table label across its photos. Leave panel ID empty to associate only the table.')
        tables=[e['label'] for e in snapshot['entities'] if e['kind']=='table']
        if tables:
            st.caption('Existing tables: '+', '.join(tables))
        with st.form(key+'-identity-form-'+group['observation_id']):
            table_label=st.text_input('Table ID',value=table)
            panel_label=st.text_input('Panel position ID (optional)',value=panel)
            submitted=st.form_submit_button('Save identity link')
        if submitted:
            try:
                assign_identity(database,group,table_label,panel_label)
                st.rerun()
            except (OSError,ValueError,duckdb.Error) as exc:
                st.error(f'Identity could not be saved: {exc}')
        if link and st.button('Remove identity link',key=key+'-identity-clear-'+group['observation_id']):
            clear_identity(database,group)
            st.rerun()
