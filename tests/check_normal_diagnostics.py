#!/usr/bin/env python3
"""Check integration fixtures, legacy schema compatibility, and corrupt gate decisions.

Usage: check_normal_diagnostics.py <NORMAL_TEST_OUTPUT directory> <output directory>
"""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_lio_diagnostics import analyze as analyze_filter
from analyze_lio_geometry import analyze as analyze_geometry


def trim_csv(path, excluded):
    with path.open() as source:
        reader = csv.DictReader(source)
        fields = [n for n in reader.fieldnames if n not in excluded and not any(n.startswith(p) for p in excluded if p.endswith('_'))]
        rows = [{key: row[key] for key in fields} for row in reader]
    with path.open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main(root, output):
    runs = sorted(root.glob('*/run-*'))
    if len(runs) < 4:
        raise RuntimeError('Expected ON/OFF x XICP/baseline integration fixtures')
    for run in runs:
        target = output / run.parent.name / run.name
        assert analyze_geometry(run, target / 'geometry')
        assert analyze_filter(run, target / 'filter')
    off = next(run for run in runs if run.parent.name == 'off-baseline')
    on = next(run for run in runs if run.parent.name == 'on-baseline')
    with tempfile.TemporaryDirectory(prefix='normal-schema-') as temporary:
        temporary = Path(temporary)
        legacy = temporary / 'v2'
        shutil.copytree(off, legacy)
        metadata = json.loads((legacy / 'metadata.json').read_text())
        metadata['schema_version'] = 2
        (legacy / 'metadata.json').write_text(json.dumps(metadata))
        trim_csv(legacy / 'geometry.csv', ['source_normal_', 'source_eigenvalues_', 'source_usable', 'source_quality',
            'source_neighbor_count', 'source_max_neighbor_distance', 'normal_angle_valid', 'normal_angle_deg'])
        trim_csv(legacy / 'scans.csv', ['source_normal_', 'normal_comparisons', 'normal_rejected'])
        assert analyze_geometry(legacy, output / 'legacy-v2/geometry')
        assert analyze_filter(legacy, output / 'legacy-v2/filter')
        metadata['schema_version'] = 1
        metadata['geometry_enabled'] = False
        (legacy / 'metadata.json').write_text(json.dumps(metadata))
        trim_csv(legacy / 'scans.csv', ['geometry_logging_ms'])
        assert analyze_filter(legacy, output / 'legacy-v1/filter')
        corrupted = temporary / 'corrupted'
        shutil.copytree(on, corrupted)
        with (corrupted / 'geometry.csv').open() as source:
            reader=csv.DictReader(source); fields=reader.fieldnames; rows=list(reader)
        rejected=next(row for row in rows if row['status']=='6')
        rejected['normal_angle_deg']='0'
        with (corrupted / 'geometry.csv').open('w') as destination:
            writer=csv.DictWriter(destination,fieldnames=fields); writer.writeheader();writer.writerows(rows)
        try:
            analyze_geometry(corrupted, output / 'corrupted')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted angle decision was accepted')
    print('Schema-v3 roundtrip, filtered-set joins/gains, v1/v2 compatibility, corruption detection: PASS')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
