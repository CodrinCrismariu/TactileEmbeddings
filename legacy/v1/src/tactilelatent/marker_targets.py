"""Native-resolution optical marker displacements; no RGB reconstruction.

Reference coordinates are episode-local calibration metadata, not universal
marker IDs. Frame zero is an episode reference, never assumed to be unloaded.
Missing/ambiguous observations are invalid, not zero-displacement labels.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

MARKER_SOURCES = ('freetacman', 'ftp_unit', 'ftp_unit_bimanual', 'ftp_vla_touch')
VERSION = 'native-marker-displacement-v2'


@dataclass(frozen=True)
class MarkerConfig:
    background_sigma: float = 5.0
    contrast_threshold: float = 10.0
    min_area: int = 7
    max_area: int = 230
    max_aspect: float = 3.0
    border: int = 6
    min_markers: int = 12
    gate_spacing_fraction: float = .40
    ambiguity_margin_fraction: float = .08
    min_area_ratio: float = .35
    max_area_ratio: float = 2.5
    merge_spacing_fraction: float = .35

    def __post_init__(self):
        if (self.background_sigma <= 0 or self.contrast_threshold <= 0
                or not 1 <= self.min_area <= self.max_area or self.border < 0
                or self.min_markers < 2 or self.max_aspect < 1
                or not 0 < self.gate_spacing_fraction < .5
                or not 0 <= self.ambiguity_margin_fraction < .5
                or not 0 <= self.merge_spacing_fraction < .5
                or not 0 < self.min_area_ratio <= 1 <= self.max_area_ratio):
            raise ValueError('Invalid marker detector/tracker configuration')


def detect_markers(image, config=MarkerConfig()):
    """Local dark-blob centroids in native (x,y) pixels, with geometry filters."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected native uint8 RGB image')
    gray = image.astype(np.float32).mean(-1)
    contrast = ndimage.gaussian_filter(gray, config.background_sigma) - gray
    labels, _ = ndimage.label(contrast > config.contrast_threshold)
    positions, areas, strengths = [], [], []
    height, width = gray.shape
    for label, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        yy, xx = bounds
        if (yy.start < config.border or xx.start < config.border
                or yy.stop > height-config.border or xx.stop > width-config.border):
            continue
        mask = labels[bounds] == label
        area = int(mask.sum())
        h, w = yy.stop-yy.start, xx.stop-xx.start
        if not config.min_area <= area <= config.max_area or max(h/w, w/h) > config.max_aspect:
            continue
        y, x = np.nonzero(mask)
        weights = contrast[bounds][mask]
        positions.append((xx.start + np.average(x, weights=weights),
                          yy.start + np.average(y, weights=weights)))
        areas.append(area)
        strengths.append(float(weights.mean()))
    positions=np.asarray(positions,dtype=np.float32).reshape(-1,2)
    areas=np.asarray(areas,dtype=np.float32);strengths=np.asarray(strengths,dtype=np.float32)
    if len(positions)>2 and config.merge_spacing_fraction:
        # A highlight can split one physical black dot into two dark components.
        # Merge only far below the robust dot pitch, before assigning track IDs.
        distances=cdist(positions,positions);np.fill_diagonal(distances,np.inf)
        pitch=float(np.median(distances.min(1)))
        parent=list(range(len(positions)))
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,j in zip(*np.where(np.triu(distances<config.merge_spacing_fraction*pitch,1))):
            parent[find(int(j))]=find(int(i))
        groups={}
        for i in range(len(positions)):groups.setdefault(find(i),[]).append(i)
        centers=[];sizes=[];contrasts=[]
        for group in groups.values():
            weight=areas[group]*strengths[group]
            centers.append(np.average(positions[group],axis=0,weights=weight))
            sizes.append(areas[group].sum());contrasts.append(np.average(strengths[group],weights=areas[group]))
        positions=np.asarray(centers,np.float32);areas=np.asarray(sizes,np.float32);strengths=np.asarray(contrasts,np.float32)
    return positions,areas,strengths


class MarkerTracker:
    """Reference-anchored assignment with subpixel centroids and explicit gaps.

    A conservative reference-distance gate prevents jumping to a neighboring
    dot. Large displacements outside that gate are rejected, not clipped.
    Uniform motion is retained: subtracting it would erase real tangential load.
    """
    def __init__(self, reference, config=MarkerConfig()):
        self.config, self.shape = config, reference.shape
        xy, area, _ = detect_markers(reference, config)
        if len(xy) < config.min_markers:
            raise ValueError(f'Only {len(xy)} reference markers; reject this sensor episode')
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        self.reference, self.area = xy[order], area[order]
        distance = cdist(self.reference, self.reference)
        np.fill_diagonal(distance, np.inf)
        self.spacing = float(np.median(distance.min(1)))
        if self.spacing <= 2:
            raise ValueError('Reference detections have implausible spacing')
        self.previous = self.reference.copy()
        self.previous_valid = np.ones(len(xy), dtype=bool)

    def update(self, image):
        if image.shape != self.shape:
            raise ValueError('Native sensor shape changed within episode')
        observed, area, _ = detect_markers(image, self.config)
        count = len(self.reference)
        xy = np.full((count, 2), np.nan, dtype=np.float32)
        valid, confidence = np.zeros(count, bool), np.zeros(count, np.float32)
        if len(observed):
            gate = self.spacing * self.config.gate_spacing_fraction
            anchor_distance = cdist(self.reference, observed)
            previous_distance = cdist(self.previous, observed)
            ratio = area[None, :] / self.area[:, None]
            allowed = ((anchor_distance <= gate)
                       & (ratio >= self.config.min_area_ratio)
                       & (ratio <= self.config.max_area_ratio))
            cost = np.where(self.previous_valid[:, None], previous_distance, anchor_distance)
            cost = np.where(allowed, cost, 1e6)
            # Per-track dummy columns allow missing observations without forcing
            # a bad real assignment that steals another track's observation.
            assignment = np.concatenate((cost, np.full((count, count), 2*gate)), axis=1)
            track_ids, observation_ids = linear_sum_assignment(assignment)
            for i, j in zip(track_ids, observation_ids):
                if j >= len(observed) or not allowed[i, j]:
                    continue
                alternatives = np.delete(cost[i], j)
                margin = alternatives.min(initial=1e6) - cost[i, j]
                if margin < self.config.ambiguity_margin_fraction * self.spacing:
                    continue
                xy[i], valid[i] = observed[j], True
                # Tracking-quality heuristic, NOT calibrated predictive uncertainty.
                confidence[i] = np.clip(1-anchor_distance[i, j]/(2*gate), .5, 1.)
        self.previous[valid] = xy[valid]
        self.previous_valid = valid.copy()
        return {'position_px': xy, 'displacement_px': xy-self.reference,
                'valid': valid, 'tracking_quality': confidence}

    def metadata(self):
        return {'version': VERSION, 'config': asdict(self.config), 'shape': list(self.shape),
                'markers': len(self.reference), 'spacing_px': self.spacing,
                'reference_kind': 'episode_frame_zero_not_verified_unloaded',
                'units': 'native_pixels', 'axis_order': ['x_right', 'y_down'],
                'global_motion_subtracted': False}
