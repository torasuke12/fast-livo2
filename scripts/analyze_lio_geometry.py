#!/usr/bin/env python3
"""Validate saved matching geometry against poses and point/gain diagnostics.

Streams the large point files one scan at a time. NumPy is required.
Run analyze_lio_diagnostics.py separately for filter/covariance checks.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from analyze_lio_diagnostics import Table


def groups(path, selected=None):
    with path.open() as source:
        names = next(source).strip().split(',')
        key, lines = None, []
        for line in source:
            fields = line.split(',', 4)
            current = (int(fields[0]), int(fields[2]))
            if current != key:
                if lines:
                    yield key, names, np.loadtxt(io.StringIO(''.join(lines)), delimiter=',', ndmin=2)
                key, lines = current, []
            if selected is None or key in selected:
                lines.append(line)
        if lines:
            yield key, names, np.loadtxt(io.StringIO(''.join(lines)), delimiter=',', ndmin=2)


def analyze(run, output):
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata.get('schema_version') == 4:
        from analyze_voxel_geometry import analyze as analyze_voxel
        return analyze_voxel(run, output, groups)
    if not metadata.get('geometry_enabled', False):
        raise ValueError('Geometry was not enabled for this run')
    schema3 = metadata.get('schema_version', 2) >= 3
    status_size = 7 if schema3 else 6
    parameters = metadata.get('ros_parameters', {})
    gate_enabled = str(parameters.get('lio.normal_consistency.enabled', 'false')).lower() == 'true'
    angle_limit = float(parameters.get('lio.normal_consistency.max_angle_deg', 30))
    frames = Table(run / 'geometry_states.csv')
    scans = Table(run / 'scans.csv', ('status',))
    scan_lookup = {int(s): j for j, s in enumerate(scans['scan_id'])}
    frame_lookup = {(int(s), int(i)): j for j, (s, i) in enumerate(zip(frames['scan_id'], frames['iteration']))}
    if len(frame_lookup) != len(frames) or not np.isfinite(frames.values).all():
        raise ValueError('Invalid/duplicate geometry frames')
    expected = {(int(s), int(i)) for s, i, t in zip(scans['scan_id'], scans['last_iteration'], scans['time'])
                if i >= -1 and metadata['detail_start_sec'] <= t < metadata['detail_end_sec']}
    if set(frame_lookup) != expected:
        raise ValueError('Missing or unexpected terminal geometry frames')
    committed = {key for key in frame_lookup if scans['status'][scan_lookup[key[0]]] == 'updated'}
    for key, fi in frame_lookup.items():
        si = scan_lookup[key[0]]
        if frames['time'][fi] != scans['time'][si]:
            raise ValueError('Geometry/scan time mismatch')
        if key in committed and key[1] != scans['committed_iteration'][si]:
            raise ValueError('Geometry is not the committed iteration')
    details = iter(groups(run / 'correspondences.csv', committed))
    errors = dict(world_transform=0., plane_distance=0., plane_fit_rms=0., plane_fit_max=0.,
                  plane_center=0., unit_normal=0., jacobian=0., residual=0.)
    counts = np.zeros(status_size, dtype=np.int64)
    cached_points, total_points = 0, 0
    seen = set()
    by_frame = []

    def compare(name, actual, target):
        if np.size(actual):
            if not np.isfinite(actual).all() or not np.isfinite(target).all():
                raise ValueError('Non-finite reconstructed ' + name)
            errors[name] = max(errors[name], float(np.max(abs(actual - target) / (1 + abs(target)))))

    def frame_matrix(name, row, n, m):
        return np.array([frames[f'{name}_{i}_{j}'][row] for i in range(n) for j in range(m)]).reshape(n, m)

    for key, names, data in groups(run / 'geometry.csv'):
        if key not in frame_lookup or key in seen:
            raise ValueError('Unknown or repeated geometry point group')
        seen.add(key)
        fi = frame_lookup[key]
        col = lambda name: data[:, names.index(name)]
        vec = lambda name: np.column_stack([col(f'{name}_{i}_0') for i in range(3)])
        if len(data) != frames['points'][fi] or not np.array_equal(col('source_index'), np.arange(len(data))):
            raise ValueError('Missing/reordered scan query points')
        if not np.all(col('time') == frames['time'][fi]):
            raise ValueError('Point/frame time mismatch')
        status = col('status').astype(int)
        if np.any((status < 0) | (status >= status_size)):
            raise ValueError('Invalid rejection reason')
        counts += np.bincount(status, minlength=status_size)
        count = col('neighbor_count').astype(int)
        if np.any((count < 0) | (count > 5)):
            raise ValueError('Invalid neighbor count')
        total_points += len(data)
        cached_points += int(np.sum((col('search_iteration') >= -1) & (col('search_iteration') != key[1])))
        accepted = status == 5
        if not np.array_equal(col('match_index')[accepted], np.arange(accepted.sum())):
            raise ValueError('Accepted correspondence mapping is not contiguous')
        if not np.all(col('match_index')[~accepted] == -1):
            raise ValueError('Rejected point has a correspondence index')
        if np.any(count[status >= 3] != 5):
            raise ValueError('Plane evaluated without five neighbors')
        R = frame_matrix('linearization_R', fi, 3, 3)
        position = frame_matrix('linearization_position', fi, 3, 1).ravel()
        extrinsic_R = frame_matrix('extrinsic_R', fi, 3, 3)
        extrinsic_t = frame_matrix('extrinsic_t', fi, 3, 1).ravel()
        imu = vec('lidar') @ extrinsic_R.T + extrinsic_t
        finite = status != 0
        transformed = (imu[finite] @ R.T + position).astype(np.float32).astype(float)
        compare('world_transform', vec('world')[finite], transformed)
        plane = status >= 4
        normals = vec('normal')[plane]
        distance = np.sum(normals * vec('world')[plane], axis=1) + col('plane_d')[plane]
        compare('plane_distance', col('distance')[plane], distance)
        compare('unit_normal', np.linalg.norm(normals, axis=1), np.ones(plane.sum()))
        neighbors = np.stack([np.column_stack([col(f'neighbors_{j}_{axis}') for axis in range(3)]) for j in range(5)], axis=1)
        fit = np.einsum('nj,nkj->nk', normals, neighbors[plane]) + col('plane_d')[plane, None]
        compare('plane_fit_rms', col('plane_fit_rms')[plane], np.sqrt(np.mean(fit * fit, axis=1)))
        compare('plane_fit_max', col('plane_fit_max')[plane], np.max(abs(fit), axis=1))
        compare('plane_center', vec('center')[plane], neighbors[plane].mean(axis=1))
        if schema3:
            usable = col('source_usable') == 1
            quality = col('source_quality').astype(int)
            angle_valid = col('normal_angle_valid') == 1
            eligible = (status == 5) | (status == 6)
            if np.any((quality < 0) | (quality > 8)) or not np.array_equal(usable, quality == 1):
                raise ValueError('Source normal quality/usable mismatch')
            if not np.array_equal(angle_valid, usable & eligible):
                raise ValueError('Source normal comparison eligibility mismatch')
            if not np.isnan(col('normal_angle_deg')[~angle_valid]).all():
                raise ValueError('Unavailable angle must be nan')
            eigenvalues = vec('source_eigenvalues')[usable]
            if len(eigenvalues):
                if (np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues[:, 0] < 0)
                        or np.any(np.diff(eigenvalues, axis=1) < 0) or np.any(eigenvalues[:, 1:] <= 1e-12)
                        or np.any(eigenvalues[:, 0] / eigenvalues[:, 1] >= float(parameters.get('lio.normal_consistency.max_planarity_ratio', .1)))
                        or np.any(eigenvalues[:, 1] / eigenvalues[:, 2] <= float(parameters.get('lio.normal_consistency.min_spread_ratio', .1)))
                        or np.any(col('source_neighbor_count')[usable] != int(parameters.get('lio.normal_consistency.neighbors', 20)))
                        or np.any(col('source_max_neighbor_distance')[usable] >= float(parameters.get('lio.normal_consistency.max_neighbor_distance', 1)))):
                    raise ValueError('Usable source normal violates quality criteria')
            compare('unit_normal', np.linalg.norm(vec('source_normal')[usable], axis=1), np.ones(usable.sum()))
            world_source = vec('source_normal')[angle_valid] @ extrinsic_R.T @ R.T
            cosine = abs(np.sum(world_source * vec('normal')[angle_valid], axis=1))
            angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
            # acos is ill-conditioned at parallel normals; allow 2 microdegrees there.
            if len(angle) and np.max(abs(angle - col('normal_angle_deg')[angle_valid])) > 2e-6:
                raise ValueError('Reconstructed normal angle differs')
            expected_rejection = angle_valid & (col('normal_angle_deg') > angle_limit + 1e-12) & gate_enabled
            if not np.array_equal(status == 6, expected_rejection):
                raise ValueError('Normal gate decision mismatch')
            si = scan_lookup[key[0]]
            for name, value in [('source_normal_valid', usable.sum()), ('source_normal_invalid', (~usable).sum()),
                                ('normal_comparisons', angle_valid.sum()), ('normal_rejected', (status == 6).sum())]:
                if scans[name][si] != value:
                    raise ValueError('Normal scan count mismatch: ' + name)
        if key in committed:
            detail_key, detail_names, detail = next(details)
            if detail_key != key or len(detail) != accepted.sum():
                raise ValueError('Geometry/correspondence group mismatch')
            H = np.column_stack([np.cross(imu[accepted], vec('normal')[accepted] @ R), vec('normal')[accepted]])
            saved_H = np.column_stack([detail[:, detail_names.index(f'H_0_{j}')] for j in range(6)])
            compare('jacobian', H, saved_H)
            # The estimator stores point-to-plane distance as float before forming residual.
            compare('residual', -col('distance')[accepted].astype(np.float32).astype(float),
                    detail[:, detail_names.index('residual')])
        by_frame.append({'scan_id': key[0], 'time': float(frames['time'][fi]), 'iteration': key[1],
                         'points': len(data), 'accepted': int(accepted.sum()),
                         'status_counts': np.bincount(status, minlength=status_size).tolist()})
    nonempty = {key for key, fi in frame_lookup.items() if frames['points'][fi] > 0}
    if seen != nonempty:
        raise ValueError('Missing geometry point groups')
    if next(details, None) is not None:
        raise ValueError('Unconsumed committed correspondence group')
    summary = {'run': str(run), 'frames': len(frames), 'points': total_points,
               'first_time': float(frames['time'][0]) if len(frames) else None,
               'last_time': float(frames['time'][-1]) if len(frames) else None,
               'status_order': ['invalid_point', 'insufficient_neighbors', 'neighbor_distance', 'plane_fit', 'residual', 'accepted'] + (['normal_mismatch'] if schema3 else []),
               'status_counts': counts.tolist(), 'cached_neighbor_points': cached_points,
               'max_scaled_errors': errors, 'passed': max(errors.values()) < 1e-6,
               'geometry_logging_ms_mean': float(scans['geometry_logging_ms'].mean()),
               'geometry_logging_ms_max': float(scans['geometry_logging_ms'].max()),
               'csv_bytes': {p.name: p.stat().st_size for p in run.glob('*.csv')},
               'frames_detail': by_frame,
               'notes': ['Saved neighbors describe the map actually queried, not the complete tree or persistent point IDs.',
                         'Numerical consistency does not establish correct correspondence or accurate attitude.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'geometry_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f"{len(frames)} frames, {total_points} query points; geometry checks {'PASS' if summary['passed'] else 'FAIL'}")
    return summary['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if analyze(args.run, args.output) else 1)
