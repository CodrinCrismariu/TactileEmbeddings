"""Explicit missing-data policies for acquired recordings; no timestamp repair."""
import csv
import io

import numpy as np


class UnavailableTactile(ValueError):
    """The pinned source contains no tactile measurements for this recording."""
    pass


def timestamp_ambiguity(camera_times, sensor_times, sensor_values):
    """Mask tied camera clocks and conflicting measurements at one sensor time.

    Preserve acquisition order and all timestamps. Repeated identical sensor
    packets are harmless; different packets with the same clock are ambiguous.
    """
    camera = np.asarray(camera_times, np.float64).reshape(-1)
    sensor = np.asarray(sensor_times, np.float64).reshape(-1)
    values = np.asarray(sensor_values)
    if (len(values) != len(sensor) or not np.isfinite(camera).all() or
            not np.isfinite(sensor).all() or np.any(np.diff(camera) < 0) or
            np.any(np.diff(sensor) < 0)):
        raise ValueError('Backward/nonfinite acquisition clocks; do not reorder frames')
    tied = np.flatnonzero(np.diff(camera) == 0)
    camera_valid = np.ones(len(camera), bool)
    camera_valid[tied] = False
    camera_valid[tied+1] = False
    sensor_tied = np.flatnonzero(np.diff(sensor) == 0)
    flattened = values.reshape(len(sensor), -1)
    different = ~np.all(flattened[sensor_tied] == flattened[sensor_tied+1], axis=1)
    conflicting_times = np.unique(sensor[sensor_tied[different]])
    return camera_valid, conflicting_times, dict(
        duplicate_camera_pairs=len(tied), ambiguous_camera_frames=int((~camera_valid).sum()),
        duplicate_sensor_pairs=len(sensor_tied), conflicting_sensor_times=len(conflicting_times))


def supervised_history_ends(times, observed, mapped_frames=None, *, frames=8, stride=2):
    """Return usable complete histories instead of rejecting a partial episode."""
    times = np.asarray(times).reshape(-1)
    observed = np.asarray(observed, bool).reshape(-1)
    if len(times) != len(observed):
        raise ValueError('History observation mask has wrong length')
    ends = np.arange((frames-1)*stride, len(times))
    indices = ends[:, None] - np.arange(frames-1, -1, -1)[None, :]*stride
    okay = observed[indices].all(1) & (np.diff(times[indices], axis=1) > 0).all(1)
    if mapped_frames is not None:
        mapped = np.asarray(mapped_frames)[indices]
        okay &= (mapped >= 0).all(1) & (np.diff(mapped, axis=1) > 0).all(1)
    return ends[okay]


def parse_contact_csv(data, name, *, absent_streams=()):
    """Read the released TacQuad column order, preserving unavailable intervals.

    An empty [0,-1] interval is accepted only with archive evidence that the
    corresponding sensor stream is absent. Other invalid intervals are isolated
    and reported per sensor; they never get sorted or turned into contact.
    """
    sensors = ('gelsight_mini', 'digit', 'duragel')
    kind = 'fine' if name == 'contact_fine.csv' else 'coarse'
    records, issues = [], []
    for number, row in enumerate(csv.reader(io.StringIO(data.decode('utf-8-sig'))), 1):
        if not row:
            continue
        if len(row) < 7:
            issues.append(dict(csv_member=name, row=number, reason='fewer than seven columns'))
            continue
        try:
            numbers = [float(v) for v in row[1:7]]
            if not all(np.isfinite(v) and v.is_integer() for v in numbers):
                raise ValueError('Non-integer frame index')
            raw = [int(v) for v in numbers]
        except (ValueError, OverflowError):
            issues.append(dict(csv_member=name, row=number, reason='non-integer frame index'))
            continue
        intervals, valid, reasons = {}, {}, {}
        for i, sensor in enumerate(sensors):
            pair = raw[2*i:2*i+2]
            supported = pair[0] >= 0 and pair[1] >= pair[0]
            reason = None
            absent = (row[0], sensor) in absent_streams
            if absent and (supported or pair == [0, -1]):
                supported = False
                reason = 'sensor stream absent from archive'
            elif not supported:
                reason = 'invalid released contact interval'
                issues.append(dict(csv_member=name, row=number, folder=row[0],
                                   sensor=sensor, raw_interval=pair, reason=reason))
            intervals[sensor] = pair if supported else None
            valid[sensor] = supported
            reasons[sensor] = reason
        records.append(dict(csv_member=name, row=number, folder=row[0],
            intervals=intervals, interval_valid=valid, interval_missing_reasons=reasons,
            raw_intervals={sensor:raw[2*i:2*i+2] for i,sensor in enumerate(sensors)},
            text=','.join(row[7:]), alignment_level=kind,
            interval_end_inclusive=True, framewise_correspondence_verified=False))
    return records, issues
