"""Schema-v4 voxel-plane checks. Called by analyze_lio_geometry.py."""
import json
import numpy as np
from analyze_lio_diagnostics import Table


def analyze(run, output, groups):
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata.get('schema_version') != 4 or metadata.get('correspondence_mode') != 'voxel_plane':
        raise ValueError('Expected voxel-plane schema-v4')
    if not metadata.get('geometry_enabled'):
        raise ValueError('Geometry was not enabled')
    parameters = metadata.get('ros_parameters', {})
    sigma_num = float(parameters.get('lio.sigma_num', 3))
    frames = Table(run / 'geometry_states.csv')
    scans = Table(run / 'scans.csv', ('status',))
    lookup = {int(s): j for j, s in enumerate(scans['scan_id'])}
    frame_lookup = {(int(s), int(i)): j for j, (s, i) in enumerate(zip(frames['scan_id'], frames['iteration']))}
    expected = {(int(s), int(i)) for s, i, t in zip(scans['scan_id'], scans['last_iteration'], scans['time'])
                if i >= -1 and metadata['detail_start_sec'] <= t < metadata['detail_end_sec']}
    if len(frame_lookup) != len(frames) or not np.isfinite(frames.values).all() or set(frame_lookup) != expected:
        raise ValueError('Invalid/duplicate/missing geometry frames')
    committed = {key for key in frame_lookup if scans['status'][lookup[key[0]]] == 'updated'}
    details = iter(groups(run / 'correspondences.csv', committed))
    errors = {k: 0. for k in ['world_transform', 'unit_normal', 'plane_distance', 'tangential_distance',
                              'gate_distance', 'gate_plane_variance', 'gate_point_variance', 'gate_variance',
                              'probability', 'update_plane_variance', 'update_point_variance', 'update_variance',
                              'inverse_variance', 'jacobian', 'residual']}
    counts = np.zeros(11, dtype=np.int64)
    seen, by_frame, total = set(), [], 0

    def compare(name, a, b):
        if np.size(a):
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Non-finite ' + name)
            errors[name] = max(errors[name], float(np.max(abs(a-b)/(1+abs(b)))))

    def covariance(a, label):
        if len(a) and (not np.isfinite(a).all() or np.max(abs(a-a.transpose(0, 2, 1))) > 1e-8
                       or np.linalg.eigvalsh(a).min() < -1e-9):
            raise ValueError('Invalid covariance: ' + label)

    for key, names, data in groups(run / 'geometry.csv'):
        if key not in frame_lookup or key in seen:
            raise ValueError('Unknown/duplicate geometry group')
        seen.add(key)
        fi, si = frame_lookup[key], lookup[key[0]]
        col = lambda name: data[:, names.index(name)]
        vec = lambda name: np.column_stack([col(f'{name}_{i}_0') for i in range(3)])
        mat = lambda name, n: np.stack([col(f'{name}_{i}_{j}') for i in range(n) for j in range(n)], axis=1).reshape(-1, n, n)
        fm = lambda name, n, m: np.array([frames[f'{name}_{i}_{j}'][fi] for i in range(n) for j in range(m)]).reshape(n, m)
        if (len(data) != frames['points'][fi] or not np.array_equal(col('source_index'), np.arange(len(data)))
                or not np.all(col('time') == frames['time'][fi]) or frames['time'][fi] != scans['time'][si]):
            raise ValueError('Missing/reordered geometry queries or mismatched times')
        if key in committed and key[1] != scans['committed_iteration'][si]:
            raise ValueError('Geometry is not the committed iteration')
        status = col('status').astype(int)
        if not np.isin(status, [0, 4, 5, 7, 8, 9, 10]).all():
            raise ValueError('Invalid voxel-plane match status')
        accepted, plane = status == 5, col('plane_valid') == 1
        if not np.array_equal(plane, np.isin(status, [4, 5, 9, 10])):
            raise ValueError('Plane validity/status mismatch')
        if (not np.array_equal(col('match_index')[accepted], np.arange(accepted.sum()))
                or not np.all(col('match_index')[~accepted] == -1)):
            raise ValueError('Invalid correspondence mapping')
        if np.any(col('plane_id')[plane] < 0) or np.any(col('support_points')[plane] < 3) or np.any(col('layer')[plane] < 0):
            raise ValueError('Invalid plane identity/support')
        R, E = fm('linearization_R', 3, 3), fm('extrinsic_R', 3, 3)
        imu = vec('lidar') @ E.T + fm('extrinsic_t', 3, 1).ravel()
        finite = status != 0
        world = (imu @ R.T + fm('linearization_position', 3, 1).ravel()).astype(np.float32).astype(float)
        compare('world_transform', vec('world')[finite], world[finite])
        normal, center = vec('normal'), vec('center')
        distance = np.sum(normal * vec('world'), axis=1) + col('plane_d')
        valid_plane = plane & (status != 10)
        compare('unit_normal', np.linalg.norm(normal[valid_plane], axis=1), np.ones(valid_plane.sum()))
        compare('plane_distance', col('distance')[valid_plane], distance[valid_plane])
        gate_distance = abs(distance).astype(np.float32)
        compare('gate_distance', col('gate_distance')[valid_plane], gate_distance[valid_plane])
        center_distance = np.sum((vec('world')-center)**2, axis=1).astype(np.float32)
        tangent = np.sqrt(np.maximum(0., center_distance - gate_distance**2))
        compare('tangential_distance', col('tangential_distance')[valid_plane], tangent[valid_plane])
        if np.any(col('tangential_distance')[status == 9] <= 3 * col('radius')[status == 9]):
            raise ValueError('Outside-extent rejection is inside the plane')
        gated = np.isin(status, [4, 5])
        # C++ multiplies the stored float radius by integer 3 in float precision.
        extent = (3 * col('radius').astype(np.float32)).astype(float)
        if np.any(col('tangential_distance')[gated] > extent[gated]):
            raise ValueError('Residual evaluated outside plane extent')
        S, Q, B = mat('plane_covariance', 6), mat('point_world_covariance', 3), mat('point_body_covariance', 3)
        covariance(S[valid_plane], 'plane'); covariance(Q[gated], 'matching point'); covariance(B[accepted], 'LiDAR point')
        J = np.column_stack([vec('world')-center, -normal])
        a = np.einsum('ni,nij,nj->n', J[gated], S[gated], J[gated])
        b = np.einsum('ni,nij,nj->n', normal[gated], Q[gated], normal[gated])
        compare('gate_plane_variance', col('gate_plane_variance')[gated], a)
        compare('gate_point_variance', col('gate_point_variance')[gated], b)
        compare('gate_variance', col('gate_variance')[gated], a+b)
        variance = col('gate_variance')
        if np.any(variance[gated] <= 0) or np.any(~np.isfinite(variance[gated])):
            raise ValueError('Invalid accepted/gated variance')
        passed = col('gate_distance')[gated] < sigma_num * np.sqrt(variance[gated])
        if not np.array_equal(passed, accepted[gated]):
            raise ValueError('Variance gate decision mismatch')
        invalid = status == 10
        if np.any(np.isfinite(variance[invalid]) & (variance[invalid] > 0) & np.isfinite(col('gate_distance')[invalid])):
            raise ValueError('Invalid-variance rejection contains a valid variance')
        probability = np.exp(-.5 * col('gate_distance')[accepted]**2 / variance[accepted]) / np.sqrt(variance[accepted])
        compare('probability', col('probability')[accepted], probability)
        Rp = fm('prior_R', 3, 3)
        prior_point = imu[accepted] @ Rp.T + fm('prior_position', 3, 1).ravel()
        Jp = np.column_stack([prior_point-center[accepted], -normal[accepted]])
        up = np.einsum('ni,nij,nj->n', Jp, S[accepted], Jp)
        local_normal = normal[accepted] @ (Rp @ E)
        uq = np.einsum('ni,nij,nj->n', local_normal, B[accepted], local_normal)
        compare('update_plane_variance', col('update_plane_variance')[accepted], up)
        compare('update_point_variance', col('update_point_variance')[accepted], uq)
        compare('update_variance', col('update_variance')[accepted], .001+up+uq)
        if key in committed:
            detail_key, detail_names, detail = next(details)
            if detail_key != key or len(detail) != accepted.sum():
                raise ValueError('Geometry/detail group mismatch')
            dc = lambda name: detail[:, detail_names.index(name)]
            H = np.column_stack([np.cross(imu[accepted], normal[accepted] @ R), normal[accepted]])
            compare('jacobian', np.column_stack([dc(f'H_0_{j}') for j in range(6)]), H)
            compare('residual', dc('residual'), -distance[accepted].astype(np.float32).astype(float))
            compare('inverse_variance', dc('inverse_variance'), 1 / col('update_variance')[accepted])
        counts += np.bincount(status, minlength=11); total += len(data)
        by_frame.append(dict(scan_id=key[0], time=float(frames['time'][fi]), iteration=key[1], points=len(data),
                             accepted=int(accepted.sum()), status_counts=np.bincount(status, minlength=11).tolist(),
                             distinct_accepted_planes=len(np.unique(col('plane_id')[accepted]))))
    if seen != {key for key, j in frame_lookup.items() if frames['points'][j] > 0} or next(details, None) is not None:
        raise ValueError('Missing geometry or unconsumed correspondence groups')
    summary = dict(run=str(run), schema_version=4, correspondence_mode='voxel_plane', frames=len(frames), points=total,
                   status_order=['invalid_point','insufficient_neighbors','neighbor_distance','plane_fit','residual','accepted',
                                 'normal_mismatch','no_voxel','no_plane','outside_extent','invalid_variance'],
                   status_counts=counts.tolist(), max_scaled_errors=errors, passed=max(errors.values()) < 1e-6,
                   frames_detail=by_frame,
                   notes=['Rejected rows represent one candidate, not every candidate examined.',
                          'Covariance and numerical consistency do not establish correct geometry or state accuracy.'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'geometry_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(f"{len(frames)} voxel-plane frames, {total} queries; geometry checks {'PASS' if summary['passed'] else 'FAIL'}")
    return summary['passed']
