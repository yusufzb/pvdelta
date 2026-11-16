# Protected observed-fragment search (research only)

`protected_fragment_completion_proposals` retains the protected dense baseline
and performs an additional native search with the minimum observed LSD segment
length reduced from 24 to 16 pixels. This changes which actual fragments enter
line clustering; it does not invent missing edges or extrapolate a table grid.

```python
from pvdelta.native_fragment_search import protected_fragment_completion_proposals

geometry = protected_fragment_completion_proposals(temperature_c)
```

Four-side coverage/gaps, shape/aspect, internal-divider, source-border, neighbour
and completion gates remain unchanged. The same bounded 2M pair cap applies; more
fragments can also produce explicit family/pair-budget skips. Never interpret a
skipped search as absence of panels. Existing geometry, IDs and partitioned
measurement pixels remain protected; extra alternatives need two disjoint
baseline scale references with one adjacent. Neither labels nor image palettes
enter the search. Native temperatures are not interpolated or modified.

EXP-092 development drafts improved at IoU .75: M4T 9/12→11/12, XT2
55/74→58/74, another XT2 44/65→53/65. The last source also produced **one extra
unmatched proposal**, so the strict no-extra-unmatched gate failed. New fixed
controls were unchanged: M4T 12/12 and a visually reviewed developer-draft
vegetation frame 0 proposals. There is no new positive-control improvement or
independent human acceptance. Unmatched geometry is not a confirmed false
thermal anomaly; all reference semantics remain provisional.

This function is not connected to automatic scans or Streamlit. Its metadata
explicitly keeps efficacy/measurement acceptance false. It cannot establish
physical inventory, calibrated surface temperatures, glare resolution or IEC
compliance. Defaults still use the original 24px minimum segment length.

Native SDK/FFF regression checks preserved all 159 previous research module
reports and 434,444 measurement pixels across seven sources. FFF outputs remain
apparent blackbody temperatures, not environmentally corrected surface readings.
Three added, matched full-panel drafts passed synthetic 16px/+10K hotspot
recovery with a separate hotter ground patch excluded. That checks support and
detector wiring; no reviewed real hotspot sensitivity was demonstrated in the
added areas. No-candidate observations must not be labelled healthy.

See [dense search](native-dense-search.md),
[decision tracing](native-search-tracing.md) and
[protected completion](native-completion.md).

[Native ridge localization](native-localization.md) is a separate research
follow-up. It does not replace missing panel detection or authorize production.
