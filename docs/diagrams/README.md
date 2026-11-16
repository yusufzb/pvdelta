# PV Delta engineering diagrams

Teal nodes indicate inputs, blue nodes indicate computations, amber diamonds indicate decisions, and lavender nodes indicate outputs or operator views. Dashed arrows indicate optional paths or reassessment.

## End-to-end analysis

The [main workflow](../../README.md#end-to-end-workflow) starts at capture planning and ends with reviewed output. [Editable Mermaid source](end-to-end.mmd) · [SVG](end-to-end.svg)

Implementation follows [flight planning](../../src/pvdelta/flight.py), [indexing](../../src/pvdelta/ingest.py), [native scanning](../../src/pvdelta/native_flight.py), [unassigned evidence](../../src/pvdelta/native_gap_review.py), [Basic review](../../src/pvdelta/basic_dashboard.py), [export](../../src/pvdelta/inspection_export.py) and [identity storage](../../src/pvdelta/identity_store.py).

## System architecture

```mermaid
flowchart TD
    ui["Streamlit · Basic and Advanced"] --> plan["Flight geometry and field checklist"]
    ui --> scan["Index and scan orchestration"]
    cli["CLI · Single flight or batch"] --> scan
    scan --> sdk["DJI SDK wrapper and saved measurement profiles"]
    scan --> geometry["Native module geometry and surface constraints"]
    sdk --> models["Support partition and local thermal detector"]
    geometry --> models
    models --> db[("DuckDB inspection records")]
    db --> review["Source previews, panel review and Folium map"]
    review -->|Review decisions| db
    db --> report["HTML, JSON and CSV evidence bundle"]
    ui --> ortho["Experimental RGB and thermal orthoview"]
    class ui,cli input
    class plan,scan,sdk,geometry,models process
    class db,review,report,ortho output
    classDef input fill:#d9f2ee,stroke:#17786e,color:#112e2b
    classDef process fill:#e8eef8,stroke:#496684,color:#162b45
    classDef decision fill:#fff1d2,stroke:#976529,color:#44301b
    classDef output fill:#e7ebff,stroke:#6265a9,color:#28264e
```

Streamlit and the CLI share the scan and measurement functions. DuckDB links source evidence, review and export. [Editable Mermaid source](system-architecture.mmd) · [SVG](system-architecture.svg)

[`dashboard/flight_planner.py`](../../dashboard/flight_planner.py), [`basic_dashboard.py`](../../src/pvdelta/basic_dashboard.py), [`native_flight.py`](../../src/pvdelta/native_flight.py), [`store.py`](../../src/pvdelta/store.py) and [`inspection_export.py`](../../src/pvdelta/inspection_export.py).

## Scientific computation

```mermaid
flowchart TD
    sdk["SDK temperature grid · °C"] --> support["Partition finite target and background support"]
    regions["Module geometry and source-bound surface labels"] --> support
    support --> viable{"Sufficient usable support and fit?"}
    viable -->|Yes| model["Fit robust local plane in native pixel coordinates"]
    viable -->|No| unresolved["Unresolved measurement"]
    model --> diff["Evaluate pixel residuals · K"]
    diff --> components["Threshold and form four-connected components"]
    components --> result["Region median · °C; median residual · K; exact pixels"]
    result -.-> conditions{"Valid scoped irradiance and documented exponent?"}
    conditions -->|Yes| norm["Optional irradiance-normalised ΔT · K"]
    conditions -->|Unavailable| raw["Report measured ΔT"]
    class sdk,regions input
    class support,model,diff,components process
    class viable,conditions decision
    class unresolved,result,norm,raw output
    classDef input fill:#d9f2ee,stroke:#17786e,color:#112e2b
    classDef process fill:#e8eef8,stroke:#496684,color:#162b45
    classDef decision fill:#fff1d2,stroke:#976529,color:#44301b
    classDef output fill:#e7ebff,stroke:#6265a9,color:#28264e
```

Candidate temperature is measured within eligible native support. The optional irradiance calculation is downstream of the measured residual. [Editable Mermaid source](scientific-pipeline.mmd) · [SVG](scientific-pipeline.svg)

[`decode.py`](../../src/pvdelta/decode.py) (`RJpeg.temperature`), [`support_partition.py`](../../src/pvdelta/support_partition.py) (`partition_measurement_support`), [`detect.py`](../../src/pvdelta/detect.py) (`assess_intra_module_anomalies`), [`conditions.py`](../../src/pvdelta/conditions.py) and [`normalize.py`](../../src/pvdelta/normalize.py).

## Data flow and units

```mermaid
flowchart TD
    rjpeg["DJI R-JPEG bytes"] --> bindings["Digest, capture pairing and source-binding checks"]
    rgb["RGB JPEG and EXIF/XMP · optional"] --> bindings
    bindings --> native["Native temperature array · °C"]
    bindings --> views["RGB preview and optional GPS or map context"]
    profiles["SDK measurement profile · JSON"] --> native
    labels["Module polygons and optional surface labels · native px"] --> detector["Mask and region assessment"]
    native --> detector
    detector --> database[("DuckDB · regions, coverage and provenance")]
    views --> dashboard["Review images, findings and map"]
    database --> dashboard
    dashboard -->|Operator decision| database
    database --> bundle["HTML, JSON and CSV evidence; export manifest"]
    class rjpeg,rgb,profiles,labels input
    class bindings,native,views,detector process
    class database,dashboard,bundle output
    classDef input fill:#d9f2ee,stroke:#17786e,color:#112e2b
    classDef process fill:#e8eef8,stroke:#496684,color:#162b45
    classDef decision fill:#fff1d2,stroke:#976529,color:#44301b
    classDef output fill:#e7ebff,stroke:#6265a9,color:#28264e
```

Native thermal coordinates remain the measurement reference. RGB, GPS and display layers provide interpretation and context. [Editable Mermaid source](data-flow.mmd) · [SVG](data-flow.svg)

[`ingest.py`](../../src/pvdelta/ingest.py) (`index_flight_folder`), [`measurement_profile.py`](../../src/pvdelta/measurement_profile.py), [`store.py`](../../src/pvdelta/store.py) and [`inspection_export.py`](../../src/pvdelta/inspection_export.py).

## Dashboard navigation

```mermaid
flowchart TD
    entry{"Dashboard mode"}
    entry --> basic["Basic · Start a new inspection"]
    basic --> results["Open scan overview and candidate list"]
    results --> paired["Review paired evidence · Figures 1 and 3"]
    paired --> choice["OK / Not OK / Wait"]
    entry --> plan["Advanced · Flight plan and Field checklist"]
    plan --> scan["Radiometric scan and Panel boundaries"]
    scan --> unassigned["Unassigned evidence · Figure 2"]
    scan --> data["Local data & map and RGB quality"]
    data --> panel["User review and Panel review (v2)"]
    panel --> export["Export"]
    data -.-> orthoview["Orthoview"]
    class basic,plan input
    class results,scan,data process
    class entry,choice decision
    class paired,unassigned,panel,export,orthoview output
    classDef input fill:#d9f2ee,stroke:#17786e,color:#112e2b
    classDef process fill:#e8eef8,stroke:#496684,color:#162b45
    classDef decision fill:#fff1d2,stroke:#976529,color:#44301b
    classDef output fill:#e7ebff,stroke:#6265a9,color:#28264e
```

Basic guides the inspection and review workflow. Advanced provides access to diagnosis, boundaries, mapping and export. [Editable Mermaid source](dashboard-navigation.mmd) · [SVG](dashboard-navigation.svg)

[`basic_dashboard.py`](../../src/pvdelta/basic_dashboard.py), [`flight_planner.py`](../../dashboard/flight_planner.py), [`panel_review_ui.py`](../../src/pvdelta/panel_review_ui.py) and the [illustrated guide](../user-guide.md).
