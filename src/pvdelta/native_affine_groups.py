"""Observed local affine quad groups; research geometry, never panel approval."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

from pvdelta.boundaries import order_quad
from pvdelta.convex_geometry import convex_iou
from pvdelta.junction_registration import radiometric_structure


def _basis(quad):
    return np.column_stack(((quad[1]-quad[0]+quad[2]-quad[3])/2,
                            (quad[3]-quad[0]+quad[2]-quad[1])/2))


def _observed_quads(values):
    import cv2

    gray, _ = radiometric_structure(values)
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    observations = []
    for threshold, (low, high) in enumerate(((10,30),(20,60),(35,100))):
        edges = cv2.Canny(smooth,low,high)
        for close in (False,True):
            binary = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)) if close else edges
            contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not 500 <= area <= values.size*.08:
                    continue
                hull = cv2.convexHull(contour)
                quad = cv2.approxPolyDP(hull,.018*cv2.arcLength(hull,True),True)
                if len(quad) != 4:
                    continue
                quad = order_quad(quad.reshape(4,2)).astype(float)
                vectors = np.roll(quad,-1,axis=0)-quad
                lengths = np.linalg.norm(vectors,axis=1)
                basis = _basis(quad)
                if (lengths.min() < 18 or area/cv2.contourArea(quad.astype(np.float32)) < .86
                        or (quad < 3).any() or (quad[:,0] >= values.shape[1]-3).any()
                        or (quad[:,1] >= values.shape[0]-3).any()
                        or np.linalg.cond(basis) > 6
                        or np.linalg.norm(vectors[0]+vectors[2]) > .2*lengths[[0,2]].mean()
                        or np.linalg.norm(vectors[1]+vectors[3]) > .2*lengths[[1,3]].mean()):
                    continue
                observations.append((quad,float(cv2.contourArea(quad.astype(np.float32))),threshold))
    groups = []
    for quad,area,threshold in sorted(observations,key=lambda r:(-r[1],*r[0].ravel())):
        match = next((g for g in groups if convex_iou(quad,g['quad']) > .65),None)
        if match is None:
            groups.append({'quad':quad,'area':area,'votes':{threshold}})
        else:
            match['votes'].add(threshold)
    retained = [g for g in groups if len(g['votes']) >= 2]
    retained.sort(key=lambda g:(*g['quad'].mean(axis=0)[::-1],*g['quad'].ravel()))
    return retained,len(observations)


def _cycles(observed):
    """Require four actual non-overlapping cells, never a predicted fourth cell."""
    if len(observed) > 512:
        return [],'too_many_observed_quads'
    if not observed:
        return [],'complete'
    quads = np.asarray([g['quad'] for g in observed])
    centres = quads.mean(axis=1)
    cycles = set()
    for i,seed in enumerate(quads):
        inverse = np.linalg.inv(_basis(seed))
        positions = (centres-centres[i])@inverse.T
        local_seed = (seed-centres[i])@inverse.T
        # Cyclic corner alignment removes the arbitrary top-left corner label.
        shape_errors = np.min(np.stack([np.linalg.norm(
            (np.roll(quads,shift,axis=1)-centres[:,None,:])@inverse.T-local_seed,axis=2).max(axis=1)
            for shift in range(4)]),axis=0)
        similar = shape_errors <= .15
        for sx in (-1,1):
            along = np.flatnonzero(similar & (sx*positions[:,0] >= .9)
                & (sx*positions[:,0] <= 1.3) & (np.abs(positions[:,1]) <= .15))
            for sy in (-1,1):
                across = np.flatnonzero(similar & (sy*positions[:,1] >= .9)
                    & (sy*positions[:,1] <= 1.3) & (np.abs(positions[:,0]) <= .15))
                if len(along)*len(across) > 64:
                    return [],'ambiguous_neighbour_search_budget'
                for j in along:
                    for k in across:
                        predicted = positions[j]+positions[k]
                        diagonal = np.flatnonzero(similar & (np.abs(positions-predicted).max(axis=1) <= .15))
                        for m in diagonal:
                            indices = tuple(sorted({i,int(j),int(k),int(m)}))
                            if len(indices) != 4 or indices in cycles:
                                continue
                            if any(convex_iou(quads[a],quads[b]) > .05 for a,b in combinations(indices,2)):
                                continue
                            cycles.add(indices)
                            if len(cycles) > 4096:
                                return [],'too_many_group_cycles'
    return sorted(cycles),'complete'


def affine_native_groups(temperature_c):
    """Find locally affine 2×2 groups of independently observed closed quads.

    No hard projected aspect or right-angle prior is used. Instead, four cells
    must have compatible projected shapes and displacements in both local axes.
    This is a local affine approximation, not calibrated projective rectification.
    Roof tiles, windows or other regular non-PV structures can also pass: semantic
    membership, full-module scale and measurement authorisation stay unresolved.
    Temperatures are used only to render geometry; the native array is unchanged.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape,default=0) < 32
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('Finite native numeric 2D grid, minimum 32 per axis, required')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    observed,raw_count = _observed_quads(values)
    cycles,status = _cycles(observed)
    supported = sorted({i for cycle in cycles for i in cycle})
    # Ambiguous overlapping alternatives are all withheld, never temperature-ranked.
    conflict = set()
    for i,j in combinations(supported,2):
        if convex_iou(observed[i]['quad'],observed[j]['quad']) > .05:
            conflict.update((i,j))
    kept_cycles = [c for c in cycles if not conflict.intersection(c)]
    retained = sorted({i for c in kept_cycles for i in c})
    identifiers = {i:f'A{index+1:04d}' for index,i in enumerate(retained)}
    groups = [{'group_id':f'G{n+1:04d}','proposal_ids':[identifiers[i] for i in c]}
              for n,c in enumerate(kept_cycles)]
    proposals = []
    for i in retained:
        memberships = [g for g in groups if identifiers[i] in g['proposal_ids']]
        inverse = np.linalg.inv(_basis(observed[i]['quad']))
        centre = observed[i]['quad'].mean(axis=0)
        neighbours = []
        for j in retained:
            if i == j:
                continue
            delta = inverse@(observed[j]['quad'].mean(axis=0)-centre)
            if ((.9 <= abs(delta[0]) <= 1.3 and abs(delta[1]) <= .15)
                    or (.9 <= abs(delta[1]) <= 1.3 and abs(delta[0]) <= .15)):
                neighbours.append(identifiers[j])
        proposals.append({'proposal_id':identifiers[i],
            'polygon_native_xy':observed[i]['quad'].tolist(), 'source_method':'closed_contour',
            'threshold_votes':len(observed[i]['votes']),
            'proposal_origin':'observed_affine_quad_group',
            'affine_group_ids':[g['group_id'] for g in memberships],
            'neighbour_ids':neighbours,
            'semantic_panel_membership':'unverified', 'measurement_authorised':False})
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during affine group search')
    return {'schema':'pvdelta.native-affine-groups/v1','status':'unreviewed_affine_geometry' if status=='complete'
            else status,'proposals':proposals,'groups':groups,
        'native_shape':list(values.shape),'native_temperature_sha256':digest,
        'raw_contour_observations':raw_count,'distinct_two_vote_quads':len(observed),
        'cycles_before_conflict_check':len(cycles),'conflicted_quads_withheld':len(conflict),
        'radiometric_resampling':False,'measurement_authorised':False,'efficacy_accepted':False,
        'semantic_panel_membership':'unverified',
        'limits':{'min_area_px':500,'max_area_fraction':.08,'min_side_px':18,
            'max_basis_condition':6,'max_opposite_edge_error':.2,'min_threshold_votes':2,
            'max_local_shape_error':.15,'max_local_position_error':.15,
            'cell_step_range':[.9,1.3],'max_observed_quads':512,'max_cycles':4096}}
