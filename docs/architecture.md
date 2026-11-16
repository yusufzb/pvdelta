# Architecture

| Responsibility | Implemented entry points | Contract |
| --- | --- | --- |
| User interface and execution | [Streamlit](../dashboard/flight_planner.py), [Basic dashboard](../src/pvdelta/basic_dashboard.py), [CLI guide](stage1-cli.md) | Operator configuration and inspection or batch run |
| Indexing and decoding | [Indexing](../src/pvdelta/ingest.py), [flight scan](../src/pvdelta/flight_scan.py), [R-JPEG decoder](../src/pvdelta/decode.py) | Source identifiers, capture metadata and native temperature array |
| Geometry and measurement | [Native scan](../src/pvdelta/native_flight.py), [support partition](../src/pvdelta/support_partition.py), [detector](../src/pvdelta/detect.py) | Candidate components, thermal coordinates, temperature, residuals and diagnostics |
| Evidence review | [Panel review](../src/pvdelta/panel_review_ui.py), [RGB pair check](../src/pvdelta/rgb_pair_check.py), [unassigned evidence](../src/pvdelta/native_gap_review.py) | Original-frame context, registration decision, unresolved evidence and reviewer decisions |
| State and delivery | [DuckDB store](../src/pvdelta/store.py), [evidence export](../src/pvdelta/inspection_export.py) | Inspection records and verifiable local evidence bundles |
| Location and identity | [Geolocation](../src/pvdelta/geolocation.py), [identity storage](../src/pvdelta/identity_store.py) | Estimated geographic context, provisional grouping and optional plant identity reconciliation |

The underlying measurement follows the native thermal grid. Geometry, review and presentation can change how evidence is assigned or displayed. Saved settings, source ownership and array coordinates make it possible to trace reported measurements. Local map overlays and orthoviews are separate presentation paths. [Data contracts](data-contracts.md) · [Inspection store](local-inspection-store.md) · [Diagram set](diagrams/README.md)
