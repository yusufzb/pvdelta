"""Published hotspot-box evaluation, distinct from full-panel boundary labels."""
from __future__ import annotations

import math

import numpy as np

from pvdelta.convex_geometry import convex_iou


def parse_yolo_hotspots(text, *, native_shape):
    """Parse class-0 YOLO boxes on the verified native grid; empty text is explicit.

    A missing label file must be handled as an error by the caller, never passed
    here as empty text. These are supplied hotspot labels, not panel polygons.
    """
    if (len(native_shape)!=2 or any(type(n) is not int or n<1 for n in native_shape)
            or not isinstance(text,str)):
        raise ValueError('Text labels and a positive native (height, width) grid required')
    height,width=native_shape
    result=[]
    for line_number,line in enumerate(text.splitlines(),1):
        if not line.strip():
            continue
        tokens=line.split()
        if len(tokens)!=5 or tokens[0]!='0':
            raise ValueError(f'Line {line_number}: exactly class 0 and four YOLO coordinates required')
        values=[float(v) for v in tokens[1:]]
        cx,cy,bw,bh=values
        if not all(math.isfinite(v) for v in values) or bw<=0 or bh<=0:
            raise ValueError(f'Line {line_number}: finite positive box size required')
        left,right=cx-bw/2,cx+bw/2
        top,bottom=cy-bh/2,cy+bh/2
        if min(left,top)<-1e-7 or max(right,bottom)>1+1e-7:
            raise ValueError(f'Line {line_number}: label extends outside the native frame')
        left,right=max(0,left)*width,min(1,right)*width
        top,bottom=max(0,top)*height,min(1,bottom)*height
        result.append({'region_id':f'H{len(result)+1:04d}',
            'polygon_native_xy':[[left,top],[right,top],[right,bottom],[left,bottom]]})
    return result


def evaluate_hotspot_regions(references, predictions, *, native_shape, iou_threshold=.5):
    """One-to-one maximum-cardinality matching against supplied hotspot boxes.

    No panel recall, fault diagnosis, semantic mask approval or healthy-module
    count follows from this region-label comparison. Unsupported frames must
    remain separately reported by the runner.
    """
    if (isinstance(iou_threshold,bool) or not math.isfinite(iou_threshold)
            or not 0<iou_threshold<=1):
        raise ValueError('IoU threshold must lie in (0,1]')
    if len(native_shape)!=2 or any(type(n) is not int or n<1 for n in native_shape):
        raise ValueError('Positive native grid required')
    def validate(items):
        ids=[];polygons=[]
        for item in items:
            identifier=item.get('region_id')
            if not isinstance(identifier,str) or not identifier or identifier in ids:
                raise ValueError('Unique nonempty region IDs required')
            p=np.asarray(item.get('polygon_native_xy'),float)
            if (p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all()
                    or (p<0).any() or (p[:,0]>native_shape[1]).any() or (p[:,1]>native_shape[0]).any()):
                raise ValueError('Finite polygons inside the native frame required')
            convex_iou(p,p)
            ids.append(identifier);polygons.append(p)
        return ids,polygons
    ref_ids,truth=validate(references);pred_ids,predicted=validate(predictions)
    overlaps=np.array([[convex_iou(p,q) for q in truth] for p in predicted]).reshape(len(predicted),len(truth))
    edges=[sorted(np.flatnonzero(row>=iou_threshold),key=lambda j:(-row[j],ref_ids[j])) for row in overlaps]
    by_ref={};by_pred={}
    for start in sorted(range(len(pred_ids)),key=lambda i:pred_ids[i]):
        queue=[start];seen={start};previous={};free=None
        for proposal in queue:
            for ref in edges[proposal]:
                if ref in previous:continue
                previous[ref]=proposal
                if ref not in by_ref:
                    free=ref;break
                occupied=by_ref[ref]
                if occupied not in seen:
                    seen.add(occupied);queue.append(occupied)
            if free is not None:break
        while free is not None:
            proposal=previous[free];old=by_pred.get(proposal)
            by_ref[free]=proposal;by_pred[proposal]=free;free=old
    tp=len(by_pred);fp=len(pred_ids)-tp;fn=len(ref_ids)-tp
    return {'iou_threshold':iou_threshold,'reference_hotspot_boxes':len(ref_ids),
        'predicted_hotspot_regions':len(pred_ids),'true_positives':tp,'false_positives':fp,'false_negatives':fn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'matches':[{'prediction_id':pred_ids[i],'reference_id':ref_ids[j],'iou':float(overlaps[i,j])}
                   for i,j in sorted(by_pred.items())],
        'unmatched_prediction_ids':[p for i,p in enumerate(pred_ids) if i not in by_pred],
        'missed_reference_ids':[r for i,r in enumerate(ref_ids) if i not in by_ref],
        'scope':'Frame-local candidate regions against supplied hotspot labels; not panel inventory or fault diagnosis',
        'measurement_authorised':False}
