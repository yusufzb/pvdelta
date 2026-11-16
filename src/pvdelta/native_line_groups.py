"""Research affine groups using actual four-side segment evidence before selection."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_affine_groups import _basis, _cycles, _observed_quads
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_lines import NativeLineConfig, _line_pass
from pvdelta.native_search_trace import trace_native_search


def _valid_affine(quad, shape):
    if (quad.shape != (4,2) or not np.isfinite(quad).all() or (quad < 3).any()
            or (quad[:,0] >= shape[1]-3).any() or (quad[:,1] >= shape[0]-3).any()):
        return False
    try:
        convex_iou(quad,quad)
    except ValueError:
        return False
    vectors=np.roll(quad,-1,axis=0)-quad
    lengths=np.linalg.norm(vectors,axis=1)
    return bool(lengths.min() >= 18 and np.linalg.cond(_basis(quad)) <= 6
        and np.linalg.norm(vectors[0]+vectors[2]) <= .2*lengths[[0,2]].mean()
        and np.linalg.norm(vectors[1]+vectors[3]) <= .2*lengths[[1,3]].mean())


class _LineEvidence:
    def __init__(self, shape):
        self.shape=shape;self.observations=[];self.passes=[];self.ends=[]
        self.seen=0;self.affine_rejected=0;self.overflow=False

    def __call__(self,event):
        if event['event']=='pass_begin':
            self.passes.append(event)
        elif event['event']=='pass_end':
            self.ends.append(event)
        elif event['event']=='candidate' and event['stage']=='four_sides_passed':
            self.seen+=1
            if self.seen > 8192:
                self.overflow=True
                return
            support=np.asarray(event['support'],float)
            if (support.shape != (4,3) or not np.isfinite(support).all()
                    or not (support[:,0]==1).all() or (support[:,1] < .8).any()
                    or (support[:,1] > 1+1e-9).any() or (support[:,2] < 0).any()
                    or (support[:,2] > .18).any()):
                raise ValueError('Four observed segment sides required')
            quad=np.asarray(event['polygon_native_xy'],float)
            if not _valid_affine(quad,self.shape):
                self.affine_rejected+=1
                return
            info=next(p for p in self.passes if p['pass_id']==event['pass_id'])
            self.observations.append({'quad':quad,'evidence':{
                'source_method':'four_observed_lines',
                'side_coverage':np.minimum(support[:,1],1).tolist(),
                'side_max_gap_fraction':support[:,2].tolist(),
                'original_side_support':event['support'],
                'native_window_xyxy':info['window_xyxy'],
                'support_mode':'segments', 'local_contrast':info['local_contrast']}})


def line_affine_native_groups(temperature_c, *, max_pair_product=250_000):
    """Combine closed quads and four-supported-line quads, then require 2×2 cycles.

    Segment coverage >=.8 and largest gap <=.18 apply to all four sides. Only
    actual four-side pass events are consumed; failed sides are never completed.
    Events are collected before legacy dedup/neighbour selection. The broader
    extraction aspect/angle limits are not approval: local affine shape and
    two-axis cycle checks must still pass. Regular non-PV structures can pass.
    This neither authorises masks nor changes production/default extraction.
    """
    values=np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape,default=0) < 32 or values.dtype.kind not in 'fiu'
            or not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native 2D numeric grid, minimum 32 per axis, required')
    line_config=NativeLineConfig(min_segment_px=16,max_pair_product=max_pair_product)
    digest=hashlib.sha256(values.tobytes()).hexdigest()
    closed,raw_closed=_observed_quads(values)
    sink=_LineEvidence(values.shape)
    height,width=values.shape
    windows=[(0,0,width,height)]
    if min(values.shape) >= 48:
        tile_h,tile_w=(2*height+2)//3,(2*width+2)//3
        windows.extend((x,y,x+tile_w,y+tile_h) for y in (0,height-tile_h) for x in (0,width-tile_w))
    skipped=[]
    # Numerical extraction envelope; the affine-group test supplies the shape prior.
    config=NativeBoundaryConfig(min_aspect=.5,max_aspect=6.1,max_corner_cosine=.95)
    with trace_native_search(values,sink):
        for x,y,right,bottom in windows:
            grid=values[y:bottom,x:right]
            low,high=np.percentile(grid,[2,98])
            if high-low <= 1e-6:
                skipped.append({'native_window_xyxy':[x,y,right,bottom],
                                'reason':'insufficient_structural_dynamic_range'})
                continue
            _line_pass(grid,config=config,line_config=line_config,local_contrast=True,support_mode='segments')
    assert len(sink.passes)==len(sink.ends)
    pool=[{'quad':g['quad'],'evidence':{'source_method':'closed_contour','threshold_votes':len(g['votes'])}}
          for g in closed]+sink.observations
    retained=[]
    for candidate in sorted(pool,key=lambda g:(g['evidence']['source_method']!='closed_contour',
            -min(g['evidence'].get('side_coverage',[1.])),*g['quad'].mean(axis=0)[::-1],*g['quad'].ravel())):
        if not any(convex_iou(candidate['quad'],other['quad']) > .65 for other in retained):
            retained.append(candidate)
    retained.sort(key=lambda g:(*g['quad'].mean(axis=0)[::-1],*g['quad'].ravel()))
    cycles,status=([],'line_evidence_budget_exceeded') if sink.overflow else _cycles(retained)
    supported=sorted({i for c in cycles for i in c});conflicts=set()
    for i,j in combinations(supported,2):
        if convex_iou(retained[i]['quad'],retained[j]['quad']) > .05:
            conflicts.update((i,j))
    selected_cycles=[c for c in cycles if not conflicts.intersection(c)]
    selected=sorted({i for c in selected_cycles for i in c})
    ids={i:f'L{n+1:04d}' for n,i in enumerate(selected)}
    groups=[{'group_id':f'LG{n+1:04d}','proposal_ids':[ids[i] for i in c]} for n,c in enumerate(selected_cycles)]
    proposals=[]
    for i in selected:
        quad=retained[i]['quad'];members=[g for g in groups if ids[i] in g['proposal_ids']]
        shared={p for g in members for p in g['proposal_ids']}
        inverse=np.linalg.inv(_basis(quad));neighbours=[]
        for j in selected:
            if i==j or ids[j] not in shared:continue
            delta=inverse@(retained[j]['quad'].mean(axis=0)-quad.mean(axis=0))
            if ((.9 <= abs(delta[0]) <= 1.3 and abs(delta[1]) <= .15)
                    or (.9 <= abs(delta[1]) <= 1.3 and abs(delta[0]) <= .15)):
                neighbours.append(ids[j])
        proposals.append(dict(retained[i]['evidence'],proposal_id=ids[i],polygon_native_xy=quad.tolist(),
            proposal_origin='observed_line_affine_group',affine_group_ids=[g['group_id'] for g in members],
            neighbour_ids=neighbours,semantic_panel_membership='unverified',measurement_authorised=False))
    if hashlib.sha256(values.tobytes()).hexdigest()!=digest:
        raise RuntimeError('Native grid changed during line affine search')
    return {'schema':'pvdelta.native-line-affine-groups/v1',
        'status':'unreviewed_line_affine_geometry' if status=='complete' else status,
        'proposals':proposals,'groups':groups,'native_shape':list(values.shape),
        'native_temperature_sha256':digest,'raw_closed_observations':raw_closed,
        'closed_two_vote_quads':len(closed),'raw_four_side_events':sink.seen,
        'line_quads_rejected_affine':sink.affine_rejected,'deduplicated_observed_quads':len(retained),
        'conflicted_quads_withheld':len(conflicts),'search_passes':sink.passes,'search_ends':sink.ends,
        'pair_budget_skipped_windows':sum(e['pair_search'].get('reason')=='too_many_line_pairs_for_bounded_search'
                                         for e in sink.ends),
        'family_budget_skipped_windows':sum(e['pair_search'].get('reason')=='too_many_line_families_for_bounded_search'
                                           for e in sink.ends),
        'skipped_windows':skipped,'line_evidence_budget':8192,
        'extraction_config':{'min_aspect':.5,'max_aspect':6.1,'max_corner_cosine':.95,
            'min_segment_px':16,'max_pair_product_per_window':line_config.max_pair_product},
        'radiometric_resampling':False,'semantic_panel_membership':'unverified',
        'measurement_authorised':False,'efficacy_accepted':False}
