#!/usr/bin/env python3
"""Summarize fixed-lag LIO diagnostics; optionally compare the prior replay NPZ."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze(source):
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Empty or incomplete temporal CSV; use a completed capture/snapshot')
    def column(name):
        return np.array([float(row[name]) for row in rows])
    def columns(names):
        return np.column_stack([column(k) for k in names])
    time = column('time')
    if (np.diff(time) <= 0).any():
        raise ValueError('Non-monotonic temporal scan timestamps')
    q = columns(['qw','qx','qy','qz'])
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7):
        raise ValueError('Invalid rotation quaternion')
    w,x,y,z = q.T
    axis = np.column_stack([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    velocity = columns(['vx','vy','vz'])
    position = columns(['px','py','pz'])
    bg,ba = columns(['bgx','bgy','bgz']),columns(['bax','bay','baz'])
    correction = columns([f'correction_{i}' for i in range(15)])
    cov = columns([f'cov_{i}_{j}' for i in range(15) for j in range(15)]).reshape(-1,15,15)
    info = columns([f'marginal_information_{i}_{j}' for i in range(15) for j in range(15)]).reshape(-1,15,15)
    numeric = [column(k) for k in rows[0] if k != 'status']
    if not all(np.isfinite(a).all() for a in numeric):
        raise ValueError('Nonfinite diagnostic value')
    minimum_eigenvalue = float(np.linalg.eigvalsh(cov).min())
    if minimum_eigenvalue <= 0:
        raise ValueError('Nonpositive marginal covariance')
    # Same body-x convention as the previous voxel-plane analysis. Includes the
    # installation pitch; it is an odometry diagnostic, not wheel ground truth.
    forward = np.einsum('ij,ij->i', axis, velocity)
    dv = correction[:,6:9]
    prior_velocity = velocity-dv
    prediction_acc = np.full_like(velocity, np.nan)
    prediction_acc[1:] = (prior_velocity[1:]-velocity[:-1])/np.diff(time)[:,None]
    pose_forward = np.full(len(time), np.nan)
    pose_forward[1:] = np.einsum('ij,ij->i', axis[1:], np.diff(position,axis=0))/np.diff(time)
    a_forward = np.einsum('ij,ij->i', axis, prediction_acc)
    update_forward = np.einsum('ij,ij->i', axis, dv)
    pitch = np.rad2deg(np.arcsin(np.clip(-axis[:,2],-1,1)))
    statuses = {s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    intervals = []
    for start,end in [(3,60),(60,140),(140,180),(180,220),(220,280),(280,310),(310,315),(315,320),(320,325),(325,336),(336,360)]:
        mask=(time>=start)&(time<end)
        if not mask.any():
            continue
        ids=np.flatnonzero(mask)
        intervals.append(dict(start=start,end=end,scans=int(mask.sum()),last_time=float(time[ids[-1]]),
            forward_mean=float(forward[mask].mean()),forward_min=float(forward[mask].min()),
            output_pose_forward_mean=float(np.nanmean(pose_forward[mask])),
            output_pose_forward_min=float(np.nanmin(pose_forward[mask])),
            prediction_acc_forward_mean=float(np.nanmean(a_forward[mask])),
            lidar_dv_forward_mean=float(update_forward[mask].mean()),pitch_mean=float(pitch[mask].mean()),
            accel_bias_mean=ba[mask].mean(axis=0).tolist(),gyro_bias_mean=bg[mask].mean(axis=0).tolist(),
            matches_median=float(np.median(column('matches')[mask])),
            velocity_std_mean=np.sqrt(cov[mask,6:9,6:9].diagonal(axis1=1,axis2=2)).mean(axis=0).tolist()))
    late=np.flatnonzero((time>=318)&(forward<-.05))
    runs=np.split(late,np.flatnonzero(np.diff(late)>1)+1) if len(late) else []
    nominal_dt=float(np.median(np.diff(time))) if len(time)>1 else 0.0
    longest_negative=max((float(time[r[-1]]-time[r[0]])+nominal_dt for r in runs),default=0.0)
    result=dict(scans=len(rows),start=float(time[0]),end=float(time[-1]),status_counts=statuses,
        max_nodes=int(column('nodes').max()),max_span=float(column('span').max()),
        solve_ms_quantiles=dict(zip(['p50','p95','max'],np.quantile(column('solve_ms'),[.5,.95,1]).tolist())),
        max_imu_step_sec=float(column('max_imu_gap').max()),min_covariance_eigenvalue=minimum_eigenvalue,
        covariance_information_identity_max=float(np.max(np.abs(cov@info-np.eye(15)))),
        max_speed=float(np.linalg.norm(velocity,axis=1).max()),
        first_negative_after_318_sec=float(time[late[0]]) if len(late) else None,
        longest_negative_after_318_duration_sec=longest_negative,
        windows=intervals,final=dict(position=position[-1].tolist(),velocity=velocity[-1].tolist(),
                                  gyro_bias=bg[-1].tolist(),accel_bias=ba[-1].tolist(),pitch=float(pitch[-1])))
    arrays=dict(time=time,position=position,velocity=velocity,ba=ba,bg=bg,pitch=pitch,forward=forward,
                output_pose_forward=pose_forward,
                prediction_acc_forward=a_forward,update_forward=update_forward,covariance=cov,
                solve_ms=column('solve_ms'),matches=column('matches'))
    return result,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result,data=analyze(args.csv)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'dynamics.npz',**data)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
        baseline=np.load(args.baseline) if args.baseline else None
        for ax,key,label in zip(axes,['forward','pitch','ba','prediction_acc_forward'],
            ['Body-x velocity (m/s)','Pitch (deg)','Accelerometer bias (m/s²)','IMU predicted body-x acceleration (m/s²)']):
            ax.plot(data['time'],data[key],label=['temporal bx','temporal by','temporal bz'] if key=='ba' else 'temporal')
            oldkey='aforward' if key=='prediction_acc_forward' else key
            if baseline is not None and key!='ba':
                values=np.rad2deg(baseline[oldkey]) if key=='pitch' else baseline[oldkey]
                ax.plot(baseline['time'],values,alpha=.65,label='previous voxel-plane')
            elif baseline is not None:
                for k,name in enumerate(['bx','by','bz']):
                    ax.plot(baseline['time'],baseline['ba'][:,k],ls='--',color=f'C{k}',alpha=.7,label='previous '+name)
            ax.set_ylabel(label);ax.grid(alpha=.3);ax.axvline(320,color='grey',ls='--');ax.legend()
        axes[-1].set_ylim(-3,3)
        axes[-1].set_ylabel('IMU predicted body-x acceleration\n(m/s²; display limited to ±3)')
        axes[-1].set_xlabel('Time from first LiDAR (s)')
        fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))


if __name__=='__main__':
    main()
