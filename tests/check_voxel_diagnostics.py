#!/usr/bin/env python3
"""Validate the actual voxel matcher/filter fixtures, including corrupt weights."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_lio_diagnostics import analyze, Table
from analyze_lio_geometry import analyze as geometry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('fixtures',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();runs={}
    for name in ['baseline','xicp','empty-baseline','empty-xicp']:
        run,=list((args.fixtures/name).glob('run-*'));runs[name]=run
        metadata=json.loads((run/'metadata.json').read_text())
        assert metadata['schema_version']==4 and metadata['correspondence_mode']=='voxel_plane'
        assert analyze(run,args.output/name/'filter')
        assert geometry(run,args.output/name/'geometry')
        g=Table(run/'geometry.csv')
        assert 'neighbor_count' not in g.columns, 'voxel records must not fabricate 5-NN data'
        if name.startswith('empty'):
            assert set(g['status'])=={7}
        else:
            assert set(g['status'])=={4,5,7,8,9,10}, 'fixture must exercise rejected as well as accepted planes'
            used=g['status']==5
            assert used.sum()==75 and np.all(g['update_variance'][used]>.001)
            assert set(g['update_enabled'][used])=={0,1}
    with tempfile.TemporaryDirectory(prefix='voxel-schema-') as directory:
        bad=Path(directory)/'bad';shutil.copytree(runs['baseline'],bad)
        path=bad/'geometry.csv';lines=path.read_text().splitlines();names=lines[0].split(',')
        for i in range(1,len(lines)):
            fields=lines[i].split(',')
            if fields[names.index('status')]=='5':
                fields[names.index('update_variance')]=str(float(fields[names.index('update_variance')])*2)
                lines[i]=','.join(fields);break
        path.write_text('\n'.join(lines)+'\n')
        assert not geometry(bad,args.output/'corrupt-weight'), 'incorrect residual variance must fail validation'
        shutil.copyfile(runs['baseline']/'geometry.csv',path)
        lines=path.read_text().splitlines()
        for i in range(1,len(lines)):
            fields=lines[i].split(',')
            if fields[names.index('status')]=='5':
                fields[names.index('gate_variance')]='0';lines[i]=','.join(fields);break
        path.write_text('\n'.join(lines)+'\n')
        try:geometry(bad,args.output/'corrupt-gate')
        except ValueError:pass
        else:raise AssertionError('zero accepted gate variance must fail')
    print('Voxel-plane schema, covariance/gain, terminal rejection, and corruption checks PASS')


if __name__=='__main__':main()
