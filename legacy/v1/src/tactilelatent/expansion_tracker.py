"""Baihu marker identities tracked temporally, without capping total motion."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .marker_targets import MarkerTracker,detect_markers


class BaihuMarkerTracker(MarkerTracker):
    """Keep initial reference coordinates; gate each temporal association.

    A dot can accumulate displacement exceeding the dot pitch. Missing tracks
    are recovered only near a locally predicted displacement from valid nearby
    tracks; ambiguity and area checks still mask uncertain observations.
    """
    def update(self,image):
        if image.shape!=self.shape:raise ValueError('Native sensor dimensions changed')
        observed,area,_=detect_markers(image,self.config);count=len(self.reference)
        xy=np.full((count,2),np.nan,np.float32);valid=np.zeros(count,bool);quality=np.zeros(count,np.float32)
        predicted=self.previous.copy()
        # Use the local deformation field to bridge missing observations.
        for i in np.flatnonzero(~self.previous_valid):
            candidates=np.flatnonzero(self.previous_valid)
            if len(candidates)<3:continue
            distance=np.linalg.norm(self.reference[candidates]-self.reference[i],axis=-1)
            nearest=candidates[np.argsort(distance)[:6]]
            nearest=nearest[np.linalg.norm(self.reference[nearest]-self.reference[i],axis=-1)<3*self.spacing]
            if len(nearest)>=3:
                predicted[i]=self.reference[i]+np.median(self.previous[nearest]-self.reference[nearest],axis=0)
        if len(observed):
            distance=cdist(predicted,observed);ratio=area[None,:]/self.area[:,None]
            gate=self.spacing*self.config.gate_spacing_fraction
            allowed=(distance<=gate)&(ratio>=self.config.min_area_ratio)&(ratio<=self.config.max_area_ratio)
            cost=np.where(allowed,distance,1e6)
            assignment=np.concatenate((cost,np.full((count,count),2*gate)),axis=1)
            ids,obs=linear_sum_assignment(assignment)
            for i,j in zip(ids,obs):
                if j>=len(observed) or not allowed[i,j]:continue
                if np.delete(cost[i],j).min(initial=1e6)-cost[i,j]<self.config.ambiguity_margin_fraction*self.spacing:continue
                xy[i]=observed[j];valid[i]=True;quality[i]=np.clip(1-distance[i,j]/(2*gate),.5,1.)
        self.previous[valid]=xy[valid];self.previous_valid=valid.copy()
        return dict(position_px=xy,displacement_px=xy-self.reference,valid=valid,tracking_quality=quality)

    def metadata(self):
        return dict(super().metadata(),association='temporal-centroid-v1',
            total_displacement_cap=None,recovery='median displacement of supported local neighbors')
