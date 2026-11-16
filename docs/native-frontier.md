# Native frontier research

`frontier_native_candidates` uses observed baseline axes and scale to place
bounded, provisional adjacent search hypotheses. Projected corners are not
observations and cannot become measurement masks. There is no recursive array
completion: newly fitted candidates cannot seed another search.

All four sides must independently pass the existing native ridge localisation:
three of four spatial strips agree, narrow peak, search within four native
pixels, fresh full-edge coverage at least .8 and largest gap at most .18.
Step-only edges, missing sides and ambiguous ridges remain unresolved. Input
temperatures are unchanged; interpolation is used only for geometric evidence.

The caller verifies source identity and the required native SHA256, then applies
separate completion gates. `complete_native_proposals` still requires two old
disjoint scale references, one adjacent, and protects old measurement support.
Conflicting candidates are withheld; overflow withholds the full fitted pool.

Every returned geometry remains semantically unverified. Regular non-PV frames
can satisfy the same evidence; baseline preservation does not establish baseline
correctness. This opt-in API does not modify production extraction, infer physical
inventory, confirm a fault or clear glare/IEC field compliance. Empty findings or
unresolved boundaries must not become healthy-panel counts. Verified RGB geometry
is a separate route; a thermal preview must never be treated as an RGB capture.
