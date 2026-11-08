#!/usr/bin/env python3
"""Analyze schema-v1/v2/v3/v4 LIO diagnostics. NumPy required; matplotlib only for --plot.

The CSVs remain the source of truth. Only final rows of scans marked updated
are treated as committed updates. Matrices retain their recorded tangent frames.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np


class Table:
    def __init__(self, path, text_columns=()):
        with path.open() as source:
            names = source.readline().strip().split(',')
            indices = [i for i, n in enumerate(names) if n not in text_columns]
            position = source.tell()
            empty = not source.readline()
            source.seek(position)
            self.values = (np.empty((0, len(indices))) if empty else
                           np.loadtxt(source, delimiter=',', usecols=indices, ndmin=2))
        self.columns = {names[i]: self.values[:, j] for j, i in enumerate(indices)}
        if text_columns:
            with path.open() as source:
                next(source)
                rows = [line.rstrip().split(',') for line in source]
            for name in text_columns:
                self.columns[name] = np.array([r[names.index(name)] for r in rows])

    def __getitem__(self, name):
        return self.columns[name]

    def __len__(self):
        return len(self.values)

    def matrix(self, name, rows, cols):
        return np.stack([self[f'{name}_{i}_{j}'] for i in range(rows) for j in range(cols)], axis=1).reshape(-1, rows, cols)


def maximum(values):
    return float(np.max(np.abs(values))) if np.size(values) else 0.0


def detail_checks(path, iterations, metadata):
    """Stream one correspondence group at a time, including large detail logs."""
    lookup = {(int(s), int(i)): j for j, (s, i) in enumerate(zip(iterations['scan_id'], iterations['iteration']))}
    errors = dict(hessian=0.0, score=0.0, innovation=0.0, distance_sign=0.0,
                  gain_contribution=0.0, gain_noise=0.0, gain_norm=0.0, KH=0.0, residual_stats=0.0)
    groups, points = set(), 0

    def compare(name, actual, expected):
        error = maximum((actual - expected) / (1.0 + np.abs(expected)))
        errors[name] = max(errors[name], error)

    def consume(key, lines, columns):
        nonlocal points
        if key not in lookup or key in groups:
            raise ValueError(f'Unknown/duplicate correspondence group {key}')
        index = lookup[key]
        a = np.loadtxt(io.StringIO(''.join(lines)), delimiter=',', ndmin=2)
        if not np.isfinite(a).all():
            raise ValueError(f'Non-finite detail values at {key}')
        column = lambda name: a[:, columns.index(name)]
        if len(a) != int(iterations['matches'][index]) or not np.array_equal(column('point_index'), np.arange(len(a))):
            raise ValueError(f'Incomplete correspondence group {key}')
        times = column('time')
        if not (np.all(times == iterations['time'][index]) and
                np.all(times >= metadata['detail_start_sec']) and np.all(times < metadata['detail_end_sec'])):
            raise ValueError(f'Detail time mismatch at {key}')
        H = np.column_stack([column(f'H_0_{i}') for i in range(6)])
        K = np.stack([column(f'K_{i}') for i in range(7, 16)])
        W, residual, innovation = (column(n) for n in ['inverse_variance', 'residual', 'innovation'])
        def saved(name, rows, cols):
            return np.array([iterations[f'{name}_{i}_{j}'][index] for i in rows for j in cols]).reshape(len(rows), len(cols))
        delta = saved('prior_delta', range(6), range(1))[:, 0]
        compare('innovation', innovation, residual - H @ delta)
        compare('distance_sign', column('distance'), -residual)
        compare('hessian', H.T @ (W[:, None] * H), saved('HWH', range(6), range(6)))
        compare('score', H.T @ (W * innovation), saved('HTWinnovation', range(6), range(1))[:, 0])
        compare('gain_contribution', K @ innovation, saved('Kinnovation', range(7, 16), range(1))[:, 0])
        compare('gain_noise', K**2 @ (1 / W), saved('gain_noise_diag', range(7, 16), range(1))[:, 0])
        compare('gain_norm', np.linalg.norm(K, axis=1), saved('gain_norm', range(7, 16), range(1))[:, 0])
        compare('KH', K @ H, saved('KH', range(7, 16), range(6)))
        compare('residual_stats', np.array([residual.mean(), np.sqrt(np.mean(residual**2)), np.max(abs(residual))]),
                np.array([iterations[n][index] for n in ['residual_mean', 'residual_rms', 'residual_max_abs']]))
        groups.add(key)
        points += len(a)

    with path.open() as source:
        columns = source.readline().strip().split(',')
        key, lines = None, []
        for line in source:
            prefix = line.split(',', 4)
            new_key = (int(prefix[0]), int(prefix[2]))
            if key is not None and new_key != key:
                consume(key, lines, columns)
                lines = []
            key = new_key
            lines.append(line)
        if lines:
            consume(key, lines, columns)
    expected = {(int(s), int(i)) for s, i, t, final in
                zip(iterations['scan_id'], iterations['iteration'], iterations['time'], iterations['final'])
                if metadata['detail_start_sec'] <= t < metadata['detail_end_sec']
                and (not metadata.get('detail_final_only', False) or final == 1)}
    if groups != expected:
        raise ValueError('Missing detailed iterations inside the configured window')
    return {'groups': len(groups), 'points': points, 'max_scaled_errors': errors,
            'passed': max(errors.values()) < 1e-8}


def analyze(run, output, plot=False):
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata['schema_version'] not in (1, 2, 3, 4):
        raise ValueError('Unsupported diagnostic schema')
    iterations = Table(run / 'iterations.csv')
    scans = Table(run / 'scans.csv', ('status',))
    if not len(scans):
        raise ValueError('No completed scans; the diagnostic log is empty or interrupted')
    if not np.isfinite(iterations.values).all() or not np.isfinite(scans.values).all():
        raise ValueError('Non-finite diagnostic values; inspect the original CSVs')
    scan_ids = {int(s): j for j, s in enumerate(scans['scan_id'])}
    if len(scan_ids) != len(scans):
        raise ValueError('Duplicate scan IDs')
    parent = np.array([scan_ids[int(s)] for s in iterations['scan_id']], dtype=int)
    if not np.array_equal(iterations['time'], scans['time'][parent]):
        raise ValueError('Iteration/scan time mismatch')
    committed = ((scans['status'][parent] == 'updated') & (iterations['final'] == 1) &
                 (iterations['iteration'] == scans['committed_iteration'][parent]))
    keys = list(zip(iterations['scan_id'], iterations['iteration']))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate iteration keys')
    for scan_id, row in scan_ids.items():
        selected = iterations['scan_id'] == scan_id
        if np.sum(selected) != scans['iterations'][row]:
            raise ValueError(f'Incomplete iteration log for scan {scan_id}')
        accepted = scans['status'][row] == 'updated'
        if np.sum(selected & committed) != int(accepted) or np.sum(iterations['final'][selected]) != int(accepted):
            raise ValueError(f'Commit status mismatch for scan {scan_id}')

    A = iterations.matrix('HWH', 6, 6)
    P = iterations.matrix('P_iter', 19, 19)
    score = iterations.matrix('HTWinnovation', 6, 1)[..., 0]
    contribution = iterations.matrix('Kinnovation', 19, 1)[..., 0]
    delta = iterations.matrix('prior_delta', 19, 1)[..., 0]
    solution = iterations.matrix('solution', 19, 1)[..., 0]
    C = iterations.matrix('C', 6, 19)
    target = iterations.matrix('target', 6, 1)[..., 0]
    prior, after = scans.matrix('P_prior', 19, 19), scans.matrix('P_after', 19, 19)
    # Equivalent to (P^-1 + Hfull^T W Hfull)^-1 * [score; 0], without
    # explicitly inverting a 19-state covariance with tiny gravity variance.
    rhs = np.linalg.solve(np.eye(6)[None] + A @ P[:, :6, :6], score[..., None])
    unconstrained = (P[:, :, :6] @ rhs)[..., 0]
    nis = iterations['innovation_wrss'] - np.sum(score * unconstrained[:, :6], axis=1)
    diagonal = np.diagonal(A, axis1=1, axis2=2)
    scale = np.divide(1.0, np.sqrt(np.maximum(diagonal, 0)), out=np.zeros_like(diagonal), where=diagonal > 0)
    normalized = A * scale[:, :, None] * scale[:, None, :]
    normalized_eigenvalues = np.linalg.eigvalsh(normalized)
    covariance_eigenvalues = np.linalg.eigvalsh(after)
    checks = {
        # Gravity is projected after solving; its applied correction need not
        # equal the unprojected Kalman contribution. Test the other 16 states.
        'correction_identity_max_abs': maximum((solution - delta - contribution)[:, :16]),
        'constraint_correction_max_abs': maximum((C @ (solution - delta)[..., None])[..., 0] - target),
        'covariance_asymmetry_max_abs': maximum(after - after.transpose(0, 2, 1)),
        'covariance_min_eigenvalue': float(covariance_eigenvalues.min()) if len(scans) else None,
    }
    statuses, counts = np.unique(scans['status'], return_counts=True)
    summary = {'schema_version': 1, 'run': str(run), 'scans': len(scans), 'iterations': len(iterations),
               'committed_updates': int(committed.sum()), 'scan_status_counts': dict(zip(statuses, counts.tolist())),
               'checks': checks,
               'iteration_logging_ms': {'mean_per_scan': float(scans['iteration_logging_ms'].mean()) if len(scans) else 0.0,
                                        'max_per_scan': maximum(scans['iteration_logging_ms'])},
               'csv_bytes': {p.name: p.stat().st_size for p in run.glob('*.csv')},
               'notes': ['NIS is descriptive; map/correspondence independence is not guaranteed.',
                         'P_prior, P_iter and P_after refer to their respective tangent frames.',
                         'K0 is a same-state counterfactual, not a separately replayed trajectory.']}
    windows = []
    for start, end in [(0, 170), (170, 180), (180, 200), (200, 220), (220, 280), (280, 310), (310, 320), (320, 337), (337, 360)]:
        selected = committed & (iterations['time'] >= start) & (iterations['time'] < end)
        if not selected.any():
            continue
        windows.append({'start': start, 'end': end, 'committed_updates': int(selected.sum()),
                        'residual_rms_median': float(np.median(iterations['residual_rms'][selected])),
                        'normalized_hessian_eigenvalues_median': np.median(normalized_eigenvalues[selected], axis=0).tolist(),
                        'nis_per_match_median': float(np.median(nis[selected] / iterations['matches'][selected])),
                        'Kinnovation_sum': contribution[selected].sum(axis=0).tolist(),
                        'K0innovation_sum': unconstrained[selected].sum(axis=0).tolist()})
    summary['windows'] = windows
    if metadata['detail_enabled']:
        summary['detail'] = detail_checks(run / 'correspondences.csv', iterations, metadata)
    summary['passed'] = (checks['correction_identity_max_abs'] < 1e-8 and
                         checks['constraint_correction_max_abs'] < 1e-8 and
                         checks['covariance_asymmetry_max_abs'] < 1e-8 and
                         (not len(scans) or checks['covariance_min_eigenvalue'] >= -1e-10) and
                         summary.get('detail', {}).get('passed', True))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if plot and len(iterations):
        plots(output, iterations, scans, committed, prior, after, normalized_eigenvalues, contribution, unconstrained)
    print(f"{run}: {len(scans)} scans, {len(iterations)} iterations, {int(committed.sum())} committed; checks {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Results: {output / 'summary.json'}")
    return summary['passed']


def plots(output, iterations, scans, committed, prior, after, eigenvalues, contribution, unconstrained):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, axes, name):
        for axis in np.atleast_1d(axes):
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel('Node-relative time [s]')
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=150)
        plt.close(fig)

    t = iterations['time'][committed]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name in ['residual_rms', 'innovation_rms']:
        axes[0].plot(t, iterations[name][committed], label=name)
    axes[0].set_ylabel('RMS [m]')
    for j in range(6):
        axes[1].plot(t, eigenvalues[committed, j], label=f'eigenvalue {j}')
    axes[1].set_ylabel('Column-normalized HWH')
    save(fig, axes, 'residuals_hessian')

    for name, values, reference, label in [
        ('gains', iterations.matrix('gain_norm', 19, 1)[..., 0], None, 'K row norm'),
        ('corrections', contribution, unconstrained, 'K * innovation'),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for axis, start, quantity in zip(axes, [7, 10, 13], ['velocity', 'gyro bias', 'accel bias']):
            for j, letter in enumerate('xyz'):
                line, = axis.plot(t, values[committed, start + j], label=f'{quantity} {letter}')
                if reference is not None:
                    axis.plot(t, reference[committed, start + j], ':', color=line.get_color(), label=f'K0 {letter}')
            axis.set_ylabel(label)
        save(fig, axes, name)

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    for axis, start, label in zip(axes[:4], [0, 7, 10, 13], ['right rotation [rad]', 'velocity [m/s]', 'gyro bias [rad/s]', 'accel bias [m/s²]']):
        for j, letter in enumerate('xyz'):
            line, = axis.plot(scans['time'], np.sqrt(np.maximum(after[:, start+j, start+j], 0)), label=f'after {letter}')
            axis.plot(scans['time'], np.sqrt(np.maximum(prior[:, start+j, start+j], 0)), ':', color=line.get_color(), label=f'prior {letter}')
        # Preserve visibility after the much larger initialization uncertainty.
        if (np.any(np.diagonal(prior, axis1=1, axis2=2)[:, start:start+3] > 0) or
                np.any(np.diagonal(after, axis1=1, axis2=2)[:, start:start+3] > 0)):
            axis.set_yscale('log')
        axis.set_ylabel('std: ' + label)
    for i, j, name in [(1, 13, 'rotY/baX'), (0, 14, 'rotX/baY'), (7, 13, 'velX/baX')]:
        denominator = np.sqrt(np.maximum(after[:, i, i] * after[:, j, j], 0))
        correlation = np.divide(after[:, i, j], denominator, out=np.full(len(after), np.nan), where=denominator > 0)
        axes[4].plot(scans['time'], correlation, label=name)
    axes[4].set_ylabel('posterior correlation')
    save(fig, axes, 'covariance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Directory containing metadata.json and the CSVs')
    parser.add_argument('--output', type=Path, help='Default: RUN/analysis')
    parser.add_argument('--plot', action='store_true', help='Write four PNG figures using matplotlib')
    args = parser.parse_args()
    return 0 if analyze(args.run, args.output or args.run / 'analysis', args.plot) else 1


if __name__ == '__main__':
    raise SystemExit(main())
