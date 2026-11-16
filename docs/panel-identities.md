# Panel and table identities

Open Basic → Review candidates → Panel cards (v2). **Anomalies by panel / table
ID** groups saved regions across their source observations. Choose **Table** for
the entire rack, or **Panel when known** for panel links with a table fallback.
Open source card returns to the original images and measurements.

Each card has a **Panel / table identity** form. Use the same table label across
its photos. An empty panel position links only the table. Panel UUIDs persist
across repeated saves. A second observation in the same frame cannot reuse an
already assigned panel UUID. A changed source polygon invalidates its assignment.

**Assign several source cards to one table** links multiple selected observations
in one transaction. Confirm their source photos before saving. An existing panel
link must be removed explicitly before a bulk table-only replacement. A failed
batch rolls back every selected link.

Assignments are stored beside the inspection as `inspection.identities.duckdb`.
Back up this file together with the inspection database. The file contains the
entity registry, source-bound assignments and their provenance. Registry import
allows explicitly supplied UUIDs to be retained across inspection databases.

The current P01 registry contains36 provisional positions from the RGB pilot.
Three candidate observations in1107/1127/1129 have provisional table links.
Other candidates await source-supported identification. Automatic geometry
associations and operator assignments are labelled separately in the interface
and the ID-based CSV export.

Where a local verification package has been prepared, **Verify table identities
offline** provides a ZIP download and a JSON upload. Package loading is explicit
because full photo context can be large. Extract the whole ZIP before opening
the workbench's `review.html`; keep the relative image folders together.

After drawing table boundaries and recording decisions, export JSON and upload
it here. Validation checks the inspection/run, observation geometry, original RGB
and thermal file hashes, reviewer, and confirmed polygons (including self-crossing
rejection). **Save validated operator review** rechecks the sources and archives
the decisions beside the database in `inspection.operator-reviews/`, named by
content hash. Repeated saves retain the same file. This review archive does not
automatically change identity assignments or anomaly verdicts; use the explicit
identity assignment forms for those changes. A confirmed table association is
not confirmation of a thermal fault.

**Review same-panel photo proposals** opens an optional paired-photo workflow
when source-bound proposals exist beside the inspection. It shows both marked
thermal previews and original RGB context with Previous/Next. The geometry
overlays on these JPEG previews are not temperature measurements. The selected
pair's RGB follow-up outcome remains separate from the human decision.

Choose **Same panel**, **Different panels**, or **Wait** with a reviewer name.
Every save verifies the current observation geometry and all four original
photo hashes. Immutable review revisions are stored in
`inspection.panel-pair-reviews/`; the latest timestamped revision is displayed.
Changing your decision retains earlier revisions. These decisions do not assign
panel UUIDs, merge anomaly regions, establish fault status or resolve glare.
Stale proposals and changed evidence are withheld.
